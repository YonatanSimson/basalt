import math

import numpy as np

import basalt_pybind


def _axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def test_ds_project_unproject_direction_consistency() -> None:
    rng = np.random.default_rng(0)

    # fx, fy, cx, cy, xi, alpha
    K = np.array([300.0, 300.0, 256.0, 256.0, 0.5, 0.55], dtype=np.float64)

    pts = rng.normal(size=(2000, 3))
    pts[:, 2] = rng.uniform(0.6, 3.0, size=pts.shape[0])

    uv, valid_proj = basalt_pybind.project_points("ds", K, pts)
    rays, valid_unproj = basalt_pybind.unproject_points("ds", K, uv)

    valid = valid_proj & valid_unproj
    assert int(valid.sum()) > 1000

    dirs = pts[valid] / np.linalg.norm(pts[valid], axis=1, keepdims=True)
    dots = np.einsum("ij,ij->i", dirs, rays[valid])
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.arccos(dots)

    assert float(np.median(ang)) < 1e-6
    assert float(np.percentile(ang, 99)) < 1e-4


def test_ds_pose_estimation_ransac_succeeds() -> None:
    rng = np.random.default_rng(1)

    # fx, fy, cx, cy, xi, alpha
    K = np.array([320.0, 318.0, 256.0, 256.0, 0.55, 0.58], dtype=np.float64)

    pts_target = rng.uniform([-0.5, -0.4, 0.0], [0.5, 0.4, 0.0], size=(120, 3)).astype(np.float64)

    R_gt = _axis_angle_to_rot(np.array([0.1, -0.2, 0.3], dtype=np.float64), 0.25)
    t_gt = np.array([0.03, -0.02, 1.4], dtype=np.float64)

    pts_cam = (R_gt @ pts_target.T).T + t_gt
    uv, valid = basalt_pybind.project_points("ds", K, pts_cam)

    pts = pts_target[valid]
    uv = uv[valid]

    pose = basalt_pybind.estimate_pose_ransac("ds", K, pts, uv)
    assert pose["success"]
    assert pose["num_inliers"] >= 20

    T = np.array(pose["T_target_camera"], dtype=np.float64)
    R_est = T[:3, :3]
    t_est = T[:3, 3]

    rot_err = np.linalg.norm(R_est - R_gt)
    trans_err = np.linalg.norm(t_est - t_gt)

    assert rot_err < 5e-2
    assert trans_err < 5e-2
