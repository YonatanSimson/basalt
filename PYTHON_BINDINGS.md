# Basalt Python Bindings Setup

The basalt C++ library now has proper Python bindings that support fisheye camera models (EUCM, DS, KB4, Pinhole) natively, avoiding OpenCV limitations.

## Quick Setup

### 1. Build basalt with CMake

```bash
cd /path/to/basalt
cmake --preset relwithdebinfo
cmake --build build/relwithdebinfo
```

### 2. Install basalt Python package

```bash
# Option A: Use setup.py (recommended for development)
cd /path/to/basalt
pip install -e .

# Option B: With custom build directory
CMAKE_BINARY_DIR=/path/to/build pip install -e .
```

### 3. Verify installation

```bash
python3 -c "import basalt; print(basalt.project_points, basalt.unproject_points, basalt.estimate_pose_ransac)"
```

## What Changed

### Key Functions

```python
import basalt
import numpy as np

# Camera intrinsics (e.g., Double Sphere)
K = np.array([fx, fy, cx, cy, xi, alpha], dtype=np.float64)

# Project 3D points to image
uv, valid = basalt.project_points("ds", K, points_3d)

# Unproject pixels to rays
rays, valid = basalt.unproject_points("ds", K, pixels)

# Estimate camera pose via RANSAC (handles fisheye!)
pose_result = basalt.estimate_pose_ransac("ds", K, points_3d, pixels)
if pose_result["success"]:
    T_w_c = np.array(pose_result["T_target_camera"], dtype=np.float64)
```

### Supported Camera Models
- `"eucm"`: Enhanced Unified Camera Model (fisheye)
- `"ds"`: Double Sphere (fisheye)  
- `"kb4"`: Kannala-Brandt 4 (fisheye)
- `"pinhole"`: Standard pinhole

## File Changes

- **basalt/setup.py** (new): Packages pre-built .so file for distribution
- **map_3d/evaluate_fisheye_calibration_basalt.py**: Now uses `basalt.estimate_pose_ransac` + `basalt.project_points`
- **map_3d/eucm_calib.py**: Added `BasaltCameraModel` wrapper class
- **map_3d/eucm_calib.py**: Renamed to better reflect multi-model support

## Usage Example

```bash
# Activate venv with basalt installed
source .venv/basalt/bin/activate

# Run evaluation
python3 evaluate_fisheye_calibration_basalt.py \
  --calibration /path/to/calibration.json \
  --corners     /path/to/detected_corners.cereal \
  --aprilgrid   /path/to/aprilgrid.json \
  --output-dir  ./results
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'basalt'`

**Solution**: 
1. Make sure basalt is built: `cmake --build build/relwithdebinfo`
2. Install it: `pip install -e .`
3. Verify: `python3 -c "import basalt"`

**Issue**: Incorrect build directory found

**Solution**: 
```bash
CMAKE_BINARY_DIR=/correct/path pip install -e .
```

## Benefits

1. **Fisheye support**: Handles EUCM, DS, KB4 natively without OpenCV limitations
2. **Accurate poses**: Uses RANSAC pose estimation optimized for bearing vectors
3. **Robust**: Filters invalid projections automatically
4. **Fast**: Leverages basalt's OpenGV integration

## Architecture

```
setup.py → copies basalt*.so → site-packages/basalt.so
        ↓
eucm_calib.py ← load_basalt_cameras() → creates BasaltCameraModel
        ↓
evaluate_fisheye_calibration_basalt.py ← uses basalt.estimate_pose_ransac + basalt.project_points
```
