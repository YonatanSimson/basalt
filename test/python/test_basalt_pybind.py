import math

import numpy as np
import pytest

import basalt


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

    uv, valid_proj = basalt.project_points("ds", K, pts)
    rays, valid_unproj = basalt.unproject_points("ds", K, uv)

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
    uv, valid = basalt.project_points("ds", K, pts_cam)

    pts = pts_target[valid]
    uv = uv[valid]

    pose = basalt.estimate_pose_ransac("ds", K, pts, uv)
    assert pose["success"]
    assert pose["num_inliers"] >= 20

    T = np.array(pose["T_target_camera"], dtype=np.float64)
    R_est = T[:3, :3]
    t_est = T[:3, 3]

    rot_err = np.linalg.norm(R_est - R_gt)
    trans_err = np.linalg.norm(t_est - t_gt)

    assert rot_err < 5e-2
    assert trans_err < 5e-2


def _render_aprilgrid(tag_px: int, rows: int, cols: int) -> tuple[np.ndarray, dict]:
    """Kalibr-style AprilGrid: tag36h11 with a 2-cell black border, black squares at every tag
    corner (spacing 0.3), slightly blurred -- the detector needs soft edges, like a real image.

    Returns the image and, per tag id, its 4 outer corners in Basalt's corner order k.
    """
    cv2 = pytest.importorskip("cv2")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    cell = tag_px // 10
    sq = int(round(0.3 * tag_px))
    step = tag_px + sq
    margin = 2 * sq
    img = np.full((2 * margin + rows * step - sq, 2 * margin + cols * step - sq), 255, np.uint8)
    corners = {}
    for r in range(rows):
        for c in range(cols):
            tag_id = r * cols + c
            x0, y0 = margin + c * step, margin + (rows - 1 - r) * step
            inner = cv2.aruco.generateImageMarker(dictionary, tag_id, 8 * cell, borderBits=1)
            img[y0 : y0 + tag_px, x0 : x0 + tag_px] = np.pad(inner, cell, constant_values=0)
            x1, y1 = x0 + tag_px, y0 + tag_px
            corners[tag_id] = np.array([[x1, y0], [x0, y0], [x0, y1], [x1, y1]], float) - 0.5
    for r in range(rows + 1):
        for c in range(cols + 1):
            x, y = margin + c * step - sq, margin + (rows - r) * step - sq
            img[max(y, 0) : y + sq, max(x, 0) : x + sq] = 0
    return cv2.GaussianBlur(img, (0, 0), 1.0), corners


def test_detect_aprilgrid_finds_every_tag_at_its_corners() -> None:
    img, truth = _render_aprilgrid(tag_px=100, rows=3, cols=4)

    for image in (img, img.astype(np.uint16) << 8):
        det = basalt.detect_aprilgrid(image, num_tags=12)
        ids = np.asarray(det["corner_ids"])
        uv = np.asarray(det["corners"])

        assert uv.shape == (ids.size, 2)
        assert ids.size == 48 and set((ids // 4).tolist()) == set(truth)
        err = [np.linalg.norm(truth[i // 4][i % 4] - p) for i, p in zip(ids, uv)]
        assert max(err) < 0.05


def test_detect_aprilgrid_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        basalt.detect_aprilgrid(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        basalt.detect_aprilgrid(np.zeros((4, 4), dtype=np.float32))
