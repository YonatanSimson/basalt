#include <basalt/camera/generic_camera.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using Camera = basalt::GenericCamera<double>;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat4 = Eigen::Matrix4d;

Camera make_camera(
    const std::string& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        params) {
  auto p = params.unchecked<1>();

  Camera cam = Camera::fromString(model);
  const Eigen::VectorXd cur = cam.getParam();

  if (static_cast<ssize_t>(cur.size()) != p.shape(0)) {
    throw std::invalid_argument("parameter count mismatch for model " + model);
  }

  Eigen::VectorXd target(cur.size());
  for (ssize_t i = 0; i < p.shape(0); ++i) {
    target[i] = p(i);
  }

  cam.applyInc(target - cur);
  return cam;
}

py::tuple project_points(
    const std::string& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        params,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        points3d) {
  auto pts = points3d.unchecked<2>();
  if (pts.shape(1) != 3) {
    throw std::invalid_argument("points3d must have shape (N,3)");
  }

  Camera cam = make_camera(model, params);

  const ssize_t n = pts.shape(0);
  py::array_t<double> uv(std::vector<py::ssize_t>{n, 2});
  py::array_t<bool> valid(std::vector<py::ssize_t>{n});
  auto uv_m = uv.mutable_unchecked<2>();
  auto val_m = valid.mutable_unchecked<1>();

  for (ssize_t i = 0; i < n; ++i) {
    Vec3 p3(pts(i, 0), pts(i, 1), pts(i, 2));
    Vec2 px;
    const bool ok = cam.project(p3, px);
    uv_m(i, 0) = px[0];
    uv_m(i, 1) = px[1];
    val_m(i) = ok;
  }

  return py::make_tuple(uv, valid);
}

py::tuple unproject_points(
    const std::string& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        params,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        pixels) {
  auto uv = pixels.unchecked<2>();
  if (uv.shape(1) != 2) {
    throw std::invalid_argument("pixels must have shape (N,2)");
  }

  Camera cam = make_camera(model, params);

  const ssize_t n = uv.shape(0);
  py::array_t<double> rays(std::vector<py::ssize_t>{n, 3});
  py::array_t<bool> valid(std::vector<py::ssize_t>{n});
  auto rays_m = rays.mutable_unchecked<2>();
  auto val_m = valid.mutable_unchecked<1>();

  for (ssize_t i = 0; i < n; ++i) {
    Vec2 px(uv(i, 0), uv(i, 1));
    Vec4 ray4;
    const bool ok = cam.unproject(px, ray4);
    Vec3 ray = ray4.head<3>();
    const double norm = ray.norm();
    if (norm > 1e-12) {
      ray /= norm;
    }
    rays_m(i, 0) = ray[0];
    rays_m(i, 1) = ray[1];
    rays_m(i, 2) = ray[2];
    val_m(i) = ok;
  }

  return py::make_tuple(rays, valid);
}

py::dict estimate_pose_ransac(
    const std::string& model,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        params,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        points3d,
    const py::array_t<double, py::array::c_style | py::array::forcecast>&
        pixels) {
  auto pts = points3d.unchecked<2>();
  auto uv = pixels.unchecked<2>();

  if (pts.shape(1) != 3 || uv.shape(1) != 2 || pts.shape(0) != uv.shape(0)) {
    throw std::invalid_argument(
        "points3d must be (N,3), pixels must be (N,2), and N must match");
  }

  Camera cam = make_camera(model, params);

  opengv::bearingVectors_t bearing_vectors;
  opengv::points_t world_points;
  bearing_vectors.reserve(pts.shape(0));
  world_points.reserve(pts.shape(0));

  for (ssize_t i = 0; i < pts.shape(0); ++i) {
    Vec2 px(uv(i, 0), uv(i, 1));
    Vec4 ray4;
    if (!cam.unproject(px, ray4)) {
      continue;
    }

    Vec3 b = ray4.head<3>();
    const double bn = b.norm();
    if (bn < 1e-12) {
      continue;
    }
    b /= bn;

    bearing_vectors.emplace_back(b);
    world_points.emplace_back(pts(i, 0), pts(i, 1), pts(i, 2));
  }

  py::dict out;
  out["num_correspondences"] = static_cast<int>(bearing_vectors.size());

  if (bearing_vectors.size() < 8) {
    out["success"] = false;
    out["num_inliers"] = 0;
    return out;
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors,
                                                        world_points);

  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  auto problem = std::make_shared<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>(
      adapter,
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP);
  ransac.sac_model_ = problem;

  const auto p = params.unchecked<1>();
  const double fx = p(0);
  ransac.threshold_ = 1.0 - std::cos(std::atan(std::sqrt(2.0) / fx));
  ransac.max_iterations_ = 50;

  const bool ok = ransac.computeModel();
  if (!ok || ransac.inliers_.empty()) {
    out["success"] = false;
    out["num_inliers"] = static_cast<int>(ransac.inliers_.size());
    return out;
  }

  Mat4 T = Mat4::Identity();
  T.topLeftCorner<3, 3>() = ransac.model_coefficients_.topLeftCorner<3, 3>();
  T.topRightCorner<3, 1>() = ransac.model_coefficients_.topRightCorner<3, 1>();

  // OpenGV returns T_camera_target; invert to get T_target_camera
  // (world→camera)
  Mat4 T_inv = Mat4::Identity();
  T_inv.topLeftCorner<3, 3>() = T.topLeftCorner<3, 3>().transpose();
  T_inv.topRightCorner<3, 1>() =
      -T.topLeftCorner<3, 3>().transpose() * T.topRightCorner<3, 1>();

  py::array_t<double> T_np(std::vector<py::ssize_t>{4, 4});
  auto T_m = T_np.mutable_unchecked<2>();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      T_m(r, c) = T_inv(r, c);
    }
  }

  out["success"] = true;
  out["num_inliers"] = static_cast<int>(ransac.inliers_.size());
  out["T_target_camera"] = T_np;
  return out;
}

}  // namespace

PYBIND11_MODULE(basalt, m) {
  m.doc() = "Basalt camera-model projection utilities and absolute pose init";

  m.def(
      "project_points", &project_points, py::arg("model"), py::arg("params"),
      py::arg("points3d"),
      "Project 3D camera-frame points (N,3) into pixels. Returns (uv, valid).");

  m.def("unproject_points", &unproject_points, py::arg("model"),
        py::arg("params"), py::arg("pixels"),
        "Unproject pixels (N,2) into unit rays. Returns (rays, valid).");

  m.def("estimate_pose_ransac", &estimate_pose_ransac, py::arg("model"),
        py::arg("params"), py::arg("points3d"), py::arg("pixels"),
        "Estimate T_target_camera from 2D-3D correspondences using "
        "Basalt-style OpenGV RANSAC.");
}
