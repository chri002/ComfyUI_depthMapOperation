# ComfyUI_depthMapOperation

# Depth Map Operation Nodes Documentation

---

## Image To Points (Torch)
![Empty Image Placeholder]()
### Description
GPU-accelerated version using PyTorch tensors. Maintains gradient flow and supports automatic device placement.

**Input Parameters**:
- `image`: Input RGB/RGBA image
- `depth_image`: Depth map image
- `depth`: Z-axis scaling factor (1-1024)
- `quality`: Downsampling quality (1=1 point:1 pixel , 16=16 interpolate points every 2 pixel)


**Output**:
- `Points3D`: XYZ coordinates + RGB colors

---

## Transform Points
![Empty Image Placeholder]()
### Description
Applies 3D transformations to point clouds (rotation, translation, scaling).

**Input Parameters**:
- `points`: Input point cloud (Points3D)
- `rot_x/y/z`: Euler angles in degrees
- `trl_x/y/z`: Translation offsets
- `scale_x/y/z`: Axis-specific scaling factors

**Output**:
- `Points3D`: Transformed point cloud (XYZ coordinates + RGB colors)

---

## Points To Image (Orthographic)
![Empty Image placeholder]()
### Description
Renders 3D points to 2D image using orthographic projection.

**Input Parameters**:
- `images`: Template for output dimensions
- `points`: Point cloud to render (Points3D)
- `color`: Enable RGB coloring

**Output**:
- `IMAGE`: Rendered grayscale/RGB image

---

## Points To Image (Projection)
![Empty Image Placeholder]()
### Description
Perspective projection renderer with customizable FOV.

**Input Parameters**:
- `images`: Template for output dimensions
- `points`: Point cloud to render (Points3D)
- `color`: Enable RGB coloring
- `fov`: Field of View in degrees (1-2000)

**Output**: 
- `IMAGE`: Rendered grayscale/RGB image

---

## Cube Limit
![Empty Image Placeholder]()
### Description
Filters points within relative cube dimensions (0-100% of original bounds).

**Input Parameters**:
- `points`: Point cloud to render (Points3D)
- 6 axis range parameters (x_min-x_max, etc.)

**Output**:
- `Points3D`: Subset of points within cube (XYZ coordinates + RGB colors)

---

## Clean Points (KDTree)
![Empty Image Placeholder]()
### Description
Removes outliers using KDTree neighborhood analysis.

**Parameters**:
- `points`: Point cloud to render (Points3D)
- `k`: Minimum neighbors required
- `m`: Max neighbor distance threshold

**Output**:
- `Points3D`: Cleaned point cloud (XYZ coordinates + RGB colors)

---

## InterpolatePointsCKDTree
![Empty Image Placeholder]()
### Description
Generates new points through neighborhood-based interpolation using KDTree. Enhances point cloud density in sparse regions by creating intermediate points between existing neighbors.

**Input Parameters**:
- `points`: Input 3D point cloud
- `value`: (0-1) Blend ratio for new points (0=keep original, 1=full interpolation)
- `n`: Number of nearest neighbors to consider (0-32)

**Output**:
- `Points3D`: Point cloud with added interpolated points (XYZ coordinates + RGB colors)

---

## Export To PLY
![Empty Image Placeholder]()
### Description
Exports point cloud to PLY format (ASCII/binary).

**Input Parameters**:
- `points`: Point cloud to render (Points3D)
- `multiple_files`: Split XYZ/RGB data
- `format_out`: File encoding format

---

## Import PLY
![Empty Image Placeholder]()
### Description
Import PLY point cloud files into compatible Point3D format.
(Sperimental)

**Input Parameter**:
- `.ply` file selection

**Output**:
- `Points3D`: Loaded point cloud data (XYZ coordinates + RGB colors)

---

## Cloud Points Info
![Empty Image Placeholder]()
### Description
Displays point cloud statistics and coordinate ranges.

**Output**:
- `STRING`: Formatted summary text
