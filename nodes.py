import numpy as np
from math import cos, sin
import torchvision.transforms as transforms
import math as m
import cv2
import torch
import os
from scipy.spatial import cKDTree
import folder_paths
import pandas as pd
import struct
import hashlib

def rxPoints(points,arrPixel,colorPix, ic,ir,r,p,p_,maxZ, quality):
  if (ic>0) and p!=r[ic-1] :
    d = float(1/(quality))
    rd = abs(p-r[ic-1])
    for pt in (range(0,quality+1)):
      pt_ = -(r[ic-1]+(rd*(pt if p>r[ic-1] else -pt)/quality))/256*maxZ
  
      points.append((ic-1+(d*(pt)),ir,pt_, colorPix[int(ir),int(ic),0],colorPix[int(ir),int(ic),1],colorPix[int(ir),int(ic),2]))
      
      ryPoints(points,arrPixel,colorPix, ic-1+(d*(pt)),ir,r,p,p_,maxZ,quality)

  elif(ic>0):
    for ict in (range(1,quality+1)):
      
      points.append((ic-1+ict/quality,ir,p_, colorPix[int(ir),int(ic),0],colorPix[int(ir),int(ic),1],colorPix[int(ir),int(ic),2]))
      ryPoints(points,arrPixel,colorPix, ic-1+ict/quality,ir,r,p,p_,maxZ,quality)

def ryPoints(points,arrPixel,colorPix,ic,ir,r,p,p_,maxZ,quality):
  if (ir>0) and p!=arrPixel[ir-1][m.floor(ic)] :
    d = float(1/(quality))
    rd = abs(p-arrPixel[ir-1][m.floor(ic)])
    for pt in (range(0,quality+1)):
      pt_ = -(arrPixel[ir-1][m.floor(ic)]+(rd*(pt if p>arrPixel[ir-1][m.floor(ic)] else -pt)/quality))/256*maxZ
      
      points.append((ic,ir-1+(d*(pt)),pt_, colorPix[int(ir),int(ic),0],colorPix[int(ir),int(ic),1],colorPix[int(ir),int(ic),2]))
  elif(ir>0):
    for irt in (range(1,quality+1)):
      
      points.append((ic,irt/quality+ir-1,p_, colorPix[int(ir),int(ic),0],colorPix[int(ir),int(ic),1],colorPix[int(ir),int(ic),2]))

def from2Dto3D(arrPixel, colorPix, maxZ, quality=1, lim_min=1):
  points = []
  for ir,r in enumerate(arrPixel):
    
    for ic,p in enumerate(r):
    
      p_ = -(p/255)*maxZ
      if p_<=-lim_min:
          points.append((ic,ir,p_, colorPix[ir,ic,0],colorPix[ir,ic,1],colorPix[ir,ic,2]))

          rxPoints(points,arrPixel,colorPix,ic,ir,r,p,p_,maxZ,quality)
      
      
  return points
  
def export_PLY(points, name_file="model.ply", multiple_files=False, format_ascii="ascii"):
    
    # Write PLY header
    header = [
        "ply",
        f"format {(format_ascii)} 1.0",
        f"element vertex {(points.shape[0])}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header\n"
    ]
    
    if not(name_file.endswith(".ply")):
        name_file+=".ply"
        
    if "ascii" in format_ascii:
        header_cons = "\n".join(header)
        if not(multiple_files):
            with open(name_file, 'wb') as f:
                np.savetxt(f, points, fmt='%g %g %g %d %d %d', header=header_cons, comments='')
            with open(name_file, 'rb+') as fout:  
                fout.seek(-1, os.SEEK_END)
                fout.truncate()
        else:
            def split_array(arr, n):
                return [arr[i:i+n] for i in range(0, len(arr), n)]
            for idx,points_ck in enumerate(split_array(points, 2097152)):
                header = [
                    "ply",
                    f"format {(format_ascii)} 1.0",
                    f"element vertex {(points_ck.shape[0])}",
                    "property float x",
                    "property float y",
                    "property float z",
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                    "end_header\n"
                ]
                name_file_t = name_file.split(".ply")[0]+"_"+str(idx)+".ply"
                with open(name_file_t, 'wb') as f:
                    np.savetxt(f, points_ck, fmt='%g %g %g %d %d %d', header=header_cons, comments='')
                with open(name_file_t, 'rb+') as fout:  
                    fout.seek(-1, os.SEEK_END)
                    fout.truncate()
            
    else:
        xyz = points[:, :3].astype(np.float32)
        rgb = points[:, 3:6].astype(np.uint8)
        
        
        
        if (multiple_files):
            def split_array(arr, n):
                return [arr[i:i+n] for i in range(0, len(arr), n)]
            xyz = split_array(xyz, 2097152)
            rgb = split_array(rgb, 2097152)
        else:
            xyz =[xyz]
            rgb =[rgb]
            
        for idx,xyz_ in enumerate(xyz):
            header = [
                "ply",
                f"format {(format_ascii)} 1.0",
                f"element vertex {(xyz_.shape[0])}",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header"
            ]
            structured_array = np.zeros(xyz_.shape[0], dtype=[
                ('x', '<f4'),  # Little-endian float32
                ('y', '<f4'),
                ('z', '<f4'),
                ('red', 'u1'),  # Unsigned byte (0-255)
                ('green', 'u1'),
                ('blue', 'u1')
            ])
            rgb_ = rgb[idx]
            name_file_t = (name_file.split(".ply")[0]+"_"+str(idx)+".ply" if multiple_files else name_file)
            structured_array['x'] = xyz_[:, 0]
            structured_array['y'] = xyz_[:, 1]
            structured_array['z'] = xyz_[:, 2]
            structured_array['red'] = rgb_[:, 0]
            structured_array['green'] = rgb_[:, 1]
            structured_array['blue'] = rgb_[:, 2]
            with open(name_file_t, 'wb') as f:
                f.write('\n'.join(header).encode('utf-8'))
                f.write(b'\n')  # Header ends with a newline
                structured_array.tofile(f)
  
def read_ply_ascii(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # Parse header
    header_end = lines.index('end_header')
    header = lines[:header_end+1]
    data = lines[header_end+1:]

    # Get vertex count
    vertex_line = next(line for line in header if line.startswith('element vertex'))
    num_vertices = int(vertex_line.split()[2])

    # Get property names and indices (x, y, z, r, g, b)
    prop_names = []
    for line in header:
        if line.startswith('property'):
            parts = line.split()
            prop_names.append(parts[-1])  # e.g., 'x', 'red', etc.

    # Map property names to target components
    component_map = {
        'x': ['x'],
        'y': ['y'],
        'z': ['z'],
        'r': ['r', 'red', 'diffuse_red'],
        'g': ['g', 'green', 'diffuse_green'],
        'b': ['b', 'blue', 'diffuse_blue']
    }
    indices = {}
    for target, aliases in component_map.items():
        for alias in aliases:
            if alias in prop_names:
                indices[target] = prop_names.index(alias)
                break
        else:
            raise ValueError(f"Missing required property: {target}")

    # Parse data
    point_cloud = []
    for line in data[:num_vertices]:
        parts = line.split()
        if len(parts)<6:
            continue
        x = float(parts[indices['x']])
        y = float(parts[indices['y']])
        z = float(parts[indices['z']])
        r = int(parts[indices['r']])  # Assume uint8 (0-255)
        g = int(parts[indices['g']])
        b = int(parts[indices['b']])
        point_cloud.append([x, y, z, r, g, b])

    return np.array(point_cloud)
  
def read_ply_binary(file_path):
    # Mapping from PLY data types to struct format characters and their sizes
    ply_type_to_struct = {
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uint': ('I', 4),
        'uchar': ('B', 1),
        'ushort': ('H', 2),
        'short': ('h', 2),
        # Add other PLY types as necessary
    }
    
    with open(file_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            if line == 'end_header':
                break
            header.append(line)
        
        # Extract vertex count
        vertex_line = next(line for line in header if line.startswith('element vertex'))
        num_vertices = int(vertex_line.split()[2])
        
        # Parse vertex properties
        vertex_properties = []
        for line in header:
            if line.startswith('property'):
                parts = line.split()
                dtype = parts[1]
                name = parts[2]
                vertex_properties.append((dtype, name))
        
        # Build struct format and calculate bytes per vertex
        struct_format = '<'  # Little-endian
        total_bytes = 0
        prop_info = []
        for dtype, name in vertex_properties:
            if dtype not in ply_type_to_struct:
                raise ValueError(f"Unsupported data type: {dtype}")
            fmt_char, size = ply_type_to_struct[dtype]
            struct_format += fmt_char
            total_bytes += size
            prop_info.append((name, fmt_char))
        
        # Read vertex data
        data = []
        for _ in range(num_vertices):
            buffer = f.read(total_bytes)
            if len(buffer) != total_bytes:
                raise ValueError("Unexpected end of file")
            unpacked = struct.unpack(struct_format, buffer)
            
            # Initialize default values
            x, y, z = 0.0, 0.0, 0.0
            r, g, b = 0, 0, 0
            
            # Extract desired properties based on names
            for i, (name, fmt_char) in enumerate(prop_info):
                value = unpacked[i]
                if name == 'x':
                    x = value
                elif name == 'y':
                    y = value
                elif name == 'z':
                    z = value
                elif name in ['r', 'red']:
                    r = int(value)
                elif name in ['g', 'green']:
                    g = int(value)
                elif name in ['b', 'blue']:
                    b = int(value)
            
            data.append([x, y, z, r, g, b])
        
        return np.array(data)
  
def from2Dto3D_vectorized_torch(arrayDepth, arrayPixel, mask, maxZ, quality=1):
    # Ensure inputs are PyTorch tensors
    assert isinstance(arrayDepth, torch.Tensor), "arrayDepth must be a torch.Tensor"
    assert isinstance(arrayPixel, torch.Tensor), "arrayPixel must be a torch.Tensor"
    
    # Check shape compatibility
    
    assert arrayPixel.shape[:2] == arrayDepth.shape, "Pixel and Depth map shapes do not match"
    W, H = arrayDepth.shape  # Original dimensions
    
    if quality > 1:
        # Calculate new dimensions
        new_W = (W - 1) * quality + 1
        new_H = (H - 1) * quality + 1
        
        # Generate coordinates 
        new_X = torch.linspace(0, W-1, new_W, device=arrayDepth.device)
        new_Y = torch.linspace(0, H-1, new_H, device=arrayDepth.device)
        
        # Create meshgrid 
        xi, yi = torch.meshgrid(new_X, new_Y, indexing='ij')
        
        # Prepare indices 
        x0 = torch.floor(xi).long()
        x1 = torch.clamp(x0 + 1, 0, W-1)
        y0 = torch.floor(yi).long()
        y1 = torch.clamp(y0 + 1, 0, H-1)
        
        dx = xi - x0
        dy = yi - y0
        
        # Interpolate depth 
        tl = arrayDepth[x0, y0]
        tr = arrayDepth[x0, y1]
        bl = arrayDepth[x1, y0]
        br = arrayDepth[x1, y1]
        interpolated_depth = (
            (1 - dx) * (1 - dy) * tl +
            dx * (1 - dy) * tr +
            (1 - dx) * dy * bl +
            dx * dy * br
        )
        
        # Interpolate pixel (
        interpolated_pixel = torch.zeros((new_W, new_H, 3), 
                                      dtype=arrayPixel.dtype,
                                      device=arrayPixel.device)
        for c in range(3):
            tl_c = arrayPixel[x0, y0, c]
            tr_c = arrayPixel[x0, y1, c]
            bl_c = arrayPixel[x1, y0, c]
            br_c = arrayPixel[x1, y1, c]
            interpolated_c = (
                (1 - dx) * (1 - dy) * tl_c +
                dx * (1 - dy) * tr_c +
                (1 - dx) * dy * bl_c +
                dx * dy * br_c
            )
            interpolated_pixel[:, :, c] = interpolated_c.to(arrayPixel.dtype)
        
        # Interpolate mask using nearest neighbor
        x_idx = torch.round(xi).long().clamp(0, W-1)
        y_idx = torch.round(yi).long().clamp(0, H-1)
        interpolated_mask = mask[x_idx, y_idx]
        mask = interpolated_mask
        
        W, H = new_W, new_H
        arrayDepth = interpolated_depth
        arrayPixel = interpolated_pixel
        final_X, final_Y = xi, yi
    else:
        # Original coordinates (CHANGED: torch.arange)
        x = torch.arange(W, device=arrayDepth.device)
        y = torch.arange(H, device=arrayDepth.device)
        final_X, final_Y = torch.meshgrid(x, y, indexing='ij')
    
    # Calculate Z (CHANGED: Tensor operations)
    d_min = torch.min(arrayDepth)
    d_max = torch.max(arrayDepth)
    if (d_max - d_min).item() > 1e-9:  # CHANGED: .item() for scalar comparison
        Z = -(arrayDepth - d_min) / (d_max - d_min) * maxZ
    else:
        Z = torch.zeros_like(arrayDepth)
    
    
    mask_ = mask==1
    # Extract RGB 
    R = arrayPixel[:, :, 0]
    G = arrayPixel[:, :, 1]
    B = arrayPixel[:, :, 2]
    
    
    # Stack components 
    points = torch.stack([
        final_Y.permute(1, 0),  # Equivalent to .T in NumPy
        final_X.permute(1, 0),
        Z.permute(1, 0),
        R.permute(1, 0),
        G.permute(1, 0),
        B.permute(1, 0)
    ], dim=-1)
    
    # Apply mask to select valid points
    mask_bool = (mask == 1).contiguous().view(-1)  # Flatten the mask
    points = points.view(-1, 6)       # Reshape points to (H*W, 6)
    points = points[mask_bool]        # Filter points using the mask
    
    return points

def from2Dto3D_vectorized(arrayDepth, arrayPixel, maxZ, quality=1):
    # Check if the input shapes are compatible
    assert arrayPixel.shape[:2] == arrayDepth.shape, "Pixel and Depth map shapes do not match"
    W, H = arrayDepth.shape  # Original dimensions
    
    if quality > 1:
        # Calculate new dimensions based on quality
        new_W = (W - 1) * quality + 1
        new_H = (H - 1) * quality + 1
        
        # Generate new coordinates using linspace
        new_X = np.linspace(0, W-1, new_W)
        new_Y = np.linspace(0, H-1, new_H)
        
        # Create meshgrid for new coordinates
        xi, yi = np.meshgrid(new_X, new_Y, indexing='ij')  # shapes (new_W, new_H)
        
        # Prepare indices for interpolation
        x0 = np.floor(xi).astype(int)
        x1 = np.clip(x0 + 1, 0, W-1)
        y0 = np.floor(yi).astype(int)
        y1 = np.clip(y0 + 1, 0, H-1)
        
        dx = xi - x0
        dy = yi - y0
        
        # Interpolate arrayDepth
        tl = arrayDepth[x0, y0]
        tr = arrayDepth[x0, y1]
        bl = arrayDepth[x1, y0]
        br = arrayDepth[x1, y1]
        interpolated_depth = (
            (1 - dx) * (1 - dy) * tl +
            dx * (1 - dy) * tr +
            (1 - dx) * dy * bl +
            dx * dy * br
        )
        
        # Interpolate arrayPixel for each channel
        interpolated_pixel = np.zeros((new_W, new_H, 3), dtype=arrayPixel.dtype)
        for c in range(3):
            tl_c = arrayPixel[x0, y0, c]
            tr_c = arrayPixel[x0, y1, c]
            bl_c = arrayPixel[x1, y0, c]
            br_c = arrayPixel[x1, y1, c]
            interpolated_c = (
                (1 - dx) * (1 - dy) * tl_c +
                dx * (1 - dy) * tr_c +
                (1 - dx) * dy * bl_c +
                dx * dy * br_c
            )
            interpolated_pixel[:, :, c] = interpolated_c.astype(arrayPixel.dtype)
        
        # Update variables to use interpolated arrays
        W, H = new_W, new_H
        arrayDepth = interpolated_depth
        arrayPixel = interpolated_pixel
        # Use new_X and new_Y for coordinates
        final_X, final_Y = np.meshgrid(new_X, new_Y, indexing='ij')
    else:
        # Original coordinates
        x = np.arange(W)
        y = np.arange(H)
        final_X, final_Y = np.meshgrid(x, y, indexing='ij')  # shapes (W, H)
    
    # Calculate scaled Z values
    d_min = np.min(arrayDepth)
    d_max = np.max(arrayDepth)
    if d_max - d_min > 1e-9:
        Z = -(arrayDepth - d_min) / (d_max - d_min) * maxZ
    else:
        Z = np.zeros_like(arrayDepth, dtype=np.float64)
    
    # Extract RGB components
    R = arrayPixel[:, :, 0]
    G = arrayPixel[:, :, 1]
    B = arrayPixel[:, :, 2]
    
    # Stack all components and reshape to Nx6
    # Transpose to shape (H, W) for correct ordering when using 'ij' indexing
    points = np.stack([
        final_Y.T, final_X.T, Z.T,
        R.T, G.T, B.T
    ], axis=-1)
    points = points.reshape(-1, 6)
    
    return points

def transform_points(points, translate=(0, 0, 0), rotate=(0, 0, 0), scale=(1,1,1)):
    """Apply 3D transformations to points (rotation first, then translation)"""
    # Convert rotation angles to radians if needed (modify if using degrees)
    rx, ry, rz = rotate
    
    
    # Create rotation matrices
    # X-axis rotation
    rot_x = np.array([
        [1, 0, 0],
        [0, cos(rx), -sin(rx)],
        [0, sin(rx), cos(rx)]
    ])
    
    # Y-axis rotation
    rot_y = np.array([
        [cos(ry), 0, sin(ry)],
        [0, 1, 0],
        [-sin(ry), 0, cos(ry)]
    ])
    
    # Z-axis rotation
    rot_z = np.array([
        [cos(rz), -sin(rz), 0],
        [sin(rz), cos(rz), 0],
        [0, 0, 1]
    ])
    
    
    # Combined rotation matrix (Z-Y-X order)
    rotation_matrix = rot_z @ rot_y @ rot_x
    
    points[:,0:3] *= np.array(scale)
    
    # Apply rotation
    points[:,0:3] = points[:,0:3] @ rotation_matrix.T
    
    # Apply translation
    points[:,0:3] += np.array(translate)
    
    return points

def project_points(points, fov=60, rotation=None, translation=None, scale=None, aspect_ratio=1, img_size=(1000,1000)):
    """Fast 3D->2D projection with perspective correction"""
    if rotation is None:
        rotation = np.zeros(3)
    else:
        rotation = np.radians(rotation)  # Convert rotation from degrees to radians
    if translation is None:
        translation = np.zeros(3)
    if scale is None:
        scale = np.ones(3)
        
    # Ensure points are in float32
    points = np.asarray(points, dtype=np.float32)
    
    # Camera parameters
    fov_rad = m.radians(fov)
    focal_length = img_size[1] / (2.0 * m.tan(fov_rad / 2.0))

    # Apply transformations 
    points = transform_points(points, translate=translation, rotate=rotation, scale=scale)

    # Select only points in front of the camera
    valid = points[:, 2] < 0
    points_t = points[valid]  # Uncommented to filter invalid points
    
    if len(points_t) == 0:
        return np.empty((0, 2)), np.array([]), np.empty((0, 3))

    # Project coordinates
    z = -points_t[:, 2]  # Distance from camera ù
    x_proj = (points_t[:, 0] * focal_length) / (z * aspect_ratio)
    y_proj = (points_t[:, 1] * focal_length) / z

    # Convert to pixel coordinates with rounding
    pixel_x = np.round(x_proj + img_size[0]/2).astype(int)
    pixel_y = np.round(y_proj + img_size[1]/2).astype(int)

    # Clip to image boundaries
    pixel_x = np.clip(pixel_x, 0, img_size[0]-1)
    pixel_y = np.clip(pixel_y, 0, img_size[1]-1)

    # Depth normalization 
    z_min = z.min()
    z_max = z.max()
    z_range = z_max - z_min + 1e-5  
    z_norm = (z - z_min)/z_range

    return np.column_stack([pixel_x, pixel_y]), z_norm, points_t[:, 3:6]
def project_points_ortho(points, rotation=None, translation=None, scale=None):
    """Orthographic 3D->2D projection"""
    if rotation is None:
        rotation = np.zeros(3)
    if translation is None:
        translation = np.zeros(3)
    if scale is None:
        scale = np.ones(3)
        
    
    # Apply transformations
    points = transform_points(points, translate=translation, rotate=rotation, scale=scale)
    
    # Orthographic projection
    x_proj = points[:, 0]  # Simple scaling + centering offset
    y_proj = points[:, 1]
    
    # Normalize depth (similar to perspective version)
    z = points[:, 2]
    z_min = z.min()
    z_max = z.max()
    z_range = z_max - z_min + 1e-5  
    z_norm = (z - z_min) / z_range
    
    colors = points[:,3:6]

    return np.column_stack([x_proj, y_proj]), z_norm, colors

def render_points_fast(points, rotation=None, translation=None, scale=None, 
                      img_size=(1000, 1000), color=False, cameraType=0, fov=60, correction=False, ksize=(2,2), thresh=2, TEST=False):
    """Ultra-fast rendering using pure NumPy and PIL"""
    # Project points to 2D
    if cameraType==0:
        proj, depth, colors = project_points_ortho(points, rotation=rotation, translation=translation)
    else:
        proj, depth, colors = project_points(points, fov=fov, aspect_ratio=1, rotation=rotation, translation=translation, scale=scale, img_size=img_size)
    
            
    # Convert to integer coordinates and filter valid points
    proj = proj.astype(np.int32)
    valid = (proj[:, 0] >= 0) & (proj[:, 0] < img_size[0]) & \
            (proj[:, 1] >= 0) & (proj[:, 1] < img_size[1]) 
    
    proj = proj[valid]
    depth = depth[valid]
    colors = colors[valid]
    
    # Grayscale rendering based on depth
    max_depth = depth.max() if depth.size > 0 else 1
    min_depth = depth.min() if depth.size > 0 else 0
    if max_depth - min_depth > 0:
        depth_normalized = (depth - min_depth) / (max_depth - min_depth)
    else:
        depth_normalized = np.zeros_like(depth)
    intensities = (255 - (depth_normalized * 255)).astype(np.uint8)
    
    
    # Sort points by descending intensity to prioritize closer points
    sorted_indices = np.argsort(-intensities)
    proj_sorted = proj[sorted_indices]
    y_coords = proj_sorted[:, 1]
    x_coords = proj_sorted[:, 0]
    
    if TEST:
        dtype = [('y', y_coords.dtype), ('x', x_coords.dtype)]
        structured = np.zeros(len(y_coords), dtype=dtype)
        structured['y'] = y_coords
        structured['x'] = x_coords
            
        _, unique_indices = np.unique(structured, return_index=True)
    else:
        # Use panda 'unique' that is more faster than numpy
        df = pd.DataFrame({'y': y_coords, 'x': x_coords})
        unique_indices = df.drop_duplicates().index.to_numpy()
    
    
    # Extract unique coordinates and colors
    unique_y = y_coords[unique_indices]
    unique_x = x_coords[unique_indices]
    
    
    if color:
        # Extract colors from valid points (assuming points are Nx[x,y,z,r,g,b])
        point_colors = colors.astype(np.uint8)
        colors_sorted = point_colors[sorted_indices]
        
        unique_colors = colors_sorted[unique_indices]
        
        # Create color buffer and assign colors
        color_buffer = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        color_buffer[unique_y, unique_x] = unique_colors
        
        img = color_buffer
    else:
        point_colors = intensities
        colors_sorted = point_colors[sorted_indices]
        
        unique_colors = colors_sorted[unique_indices]
        
        # Create color buffer and assign colors
        color_buffer = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        color_buffer[unique_y, unique_x] = unique_colors
        
        img = color_buffer
    if correction:
        point_colors = intensities
        colors_sorted = point_colors[sorted_indices]
        
        unique_colors = colors_sorted[unique_indices]
        
        # Create color buffer and assign colors
        depth_buffer = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        depth_buffer[unique_y, unique_x] = unique_colors
        
        img_depth = depth_buffer
        
        img = np.array(img)
        img_dep = np.array(img_depth)
                       
        img_blur = blur_image_excluding_black(img, ksize, thresh)
        
        mask = img_dep < thresh
        img[mask] = img_blur[mask]
        
    return tensor_im(img)

def tensor_im(img):
    img = img.squeeze()
    if len(img.shape)==3:
        return img
    elif len(img.shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    else:
        raise ValueError("Image shape not right")

def clean_points(points, k, m):
    points = np.asarray(points)
    if points.size == 0:
        return points.copy()
    if k <= 0:
        raise ValueError("k must be a positive integer")
    n = len(points)
    if k > n - 1:
        return np.empty((0, points.shape[1]))
    tree = cKDTree(points, balanced_tree=False)
    # Query k+1 to include the point itself, then select the k-th neighbor
    distances, _ = tree.query(points, k=k+1, workers=-1)
    kth_distances = distances[:, k]  # k-th neighbor after excluding self
    mask = kth_distances <= m
    return points[mask]

def interpolate_points(points, alpha=0.5, n_c=3):
    """
    Interpolate n_c new points for each original point, positioned between the original and its n_c nearest neighbors.
    
    Parameters:
    points (numpy.ndarray): Input array of shape (N, 6) where each row is (x, y, z, r, g, b).
    alpha (float): Interpolation factor (0.0 = original point, 1.0 = neighbor). Default is 0.5 (midpoint).
    
    Returns:
    numpy.ndarray: Array of interpolated points with shape (n_c*N, 6).
    """
    # Extract coordinates and colors
    coords = points[:, :3]
    colors = points[:, 3:]
    N = coords.shape[0]
    
    # Build KDTree for efficient neighbor lookup
    tree = cKDTree(coords, balanced_tree=False)
    
    # Query for n_c+1 nearest neighbors (including self), then exclude self
    _, indices = tree.query(coords, k=n_c+1, workers=-1)
    neighbor_indices = indices[:, 1:(n_c+1)]  # Shape (N, n_c)
    
    # Prepare indices for vectorized operations
    original_indices = np.repeat(np.arange(N), n_c)
    neighbors_flat = neighbor_indices.ravel()
    
    # Gather original and neighbor data
    original_coords = coords[original_indices]
    neighbor_coords = coords[neighbors_flat]
    
    original_colors = colors[original_indices]
    neighbor_colors = colors[neighbors_flat]
    
    # Interpolate coordinates and colors
    interpolated_coords = (1 - alpha) * original_coords + alpha * neighbor_coords
    interpolated_colors = (1 - alpha) * original_colors + alpha * neighbor_colors
    
    # Combine into new points array
    new_points = np.hstack((interpolated_coords, interpolated_colors))
    
    combined_points = np.vstack((points, new_points))
    
    return combined_points

def blur_image_excluding_black(image, kernel_size=(4,4), threshold=2):
    # Create mask for non-black pixels (all channels zero)
    if image.ndim == 3:
        mask = np.any(image > threshold, axis=-1).astype(float)
    else:
        mask = (image > threshold).astype(float)
    
    kernel = np.ones(kernel_size)
    
    if image.ndim == 3:
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            masked_channel = channel * mask
            sum_matrix = convolve2d_np(masked_channel, kernel, mode='same')
            count_matrix = convolve2d_np(mask, kernel, mode='same')
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_matrix = sum_matrix / count_matrix
                # Where count is zero, use original pixel if non-black, else 0
                blurred_channel = np.where(count_matrix > 0, mean_matrix, masked_channel)
            blurred[:, :, c] = blurred_channel
    else:
        masked_image = image * mask
        sum_matrix = convolve2d_np(masked_image, kernel, mode='same')
        count_matrix = convolve2d_np(mask, kernel, mode='same')
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_matrix = sum_matrix / count_matrix
            blurred = np.where(count_matrix > 0, mean_matrix, masked_image)
    
    return blurred
    
def convolve2d_np(image, kernel, mode='same'):
    kernel = np.flipud(np.fliplr(kernel))
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape
    
    if mode=='same':
        pad_top = (k_h-1)//2
        pad_bottom = (k_h-1)-pad_top
        pad_left = (k_w-1)//2
        pad_right = (k_w-1)-pad_left
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    elif mode=='valid':
        padded_image = image
    elif mode=='full':
        padded_image = np.pad(image, ((k_h-1,k_h-1),(k_w-1,k_w-1)), mode='constant')
    else:
        raise ValueError("Mode must be 'same', 'valid' or 'full'")
        
    #gen sliding windows
    windows = np.lib.stride_tricks.sliding_window_view(padded_image, (k_h, k_w))
    
    #perform convolution by summing element-wise multiplication
    result = np.sum(windows * kernel.reshape(1,1,k_h,k_w), axis=(-2,-1))
    
    return result
    
    

############################

class ImageToPoints:
                
    def __init__(self, device="cpu"):
        self.device = device
    def imageTo3Dpoints(self, image, depth_image,depth, quality, mask=None):
        returns = None
        
        
        if (image.cpu().numpy().shape[0])>1:
            raise Exception("Batch not work")
        
        
        for img, cols in zip(depth_image.cpu().numpy(), image.cpu().numpy()):
        
            if cols.shape[2]==4:
                cols = cv2.cvtColor(cols, cv2.COLOR_RGBA2RGB)
                
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            
            
            w,h = img.shape[0:2]
            
            cols = cv2.resize(cols, (h,w)) 
            points = from2Dto3D(img*255,cols*255,depth,quality, lim_min)
            
            
            
            if returns == None:
                returns = torch.from_numpy(np.array(points)[np.newaxis,:,:])
        
        return (returns)
            
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",{}),
                "depth_image": ("IMAGE",{}),
                "depth" : ("INT", {"default": 1, "min":1, "max":1024}),
                "quality" : ("INT", {"default": 1, "min":1, "max":16}),
            },
        }

    RETURN_TYPES = ("Points3D",)
    FUNCTION = "imageTo3Dpoints"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"
    
class ImageToPointsTorch:
                
    def __init__(self, device="cpu"):
        self.device = device
    def imageTo3Dpoints(self, image, depth_image, depth, quality, mask=None):
        # Validate batch size
        if image.shape[0] > 1 or depth_image.shape[0] > 1:
            raise ValueError("Batch processing not supported in this version")
        
        # Ensure tensors are on same device
        device = image.device
        depth_image = depth_image.to(device)
        
        # Remove batch dimension while keeping gradient
        depth_image = depth_image.squeeze(0)  # [H, W, 4]
        image = image.squeeze(0)             # [H, W, 4]
        alpha = torch.ones_like(image[:,:,0])[...,None]
        
        # Convert RGBA to RGB/RGRAY using tensor operations
        def rgba_to_rgb(tensor):
            return tensor[..., :3], tensor[..., 3]  # Simple alpha channel removal
        
        def rgba_to_grayscale(tensor):
            # Use luminance weights (same as cv2.COLOR_RGBA2GRAY coefficients)
            return (tensor[..., 0] * 0.299 + 
                    tensor[..., 1] * 0.587 + 
                    tensor[..., 2] * 0.114)
        
        # Process colors
        if image.shape[-1] == 4:
            image,alpha = rgba_to_rgb(image)
        if mask!=None:
            alpha = mask.squeeze()[..., None]
        
        # Process depth image (convert to grayscale)
        if image.shape[-1] >= 3:
            depth_gray = rgba_to_grayscale(depth_image)
        else:
            depth_gray = depth_image
        
        # Resize operations using torch (preserve gradients)
        def resize_tensor(tensor, size):
            # Input: [H, W, C], Output: [H_new, W_new, C]
            return torch.nn.functional.interpolate(
                tensor.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
                size=size,
                mode='bilinear' if tensor.dtype.is_floating_point else 'nearest'
            ).squeeze(0).permute(1, 2, 0)
        
        # Get original dimensions
        h, w = depth_gray.shape[:2]
        
        
        # Resize color image to match depth dimensions
        if image.shape[:2] != (h, w):
            image = resize_tensor(image, (h, w))
            alpha = resize_tensor(alpha, (h, w))
       
        
        # Convert to float and normalize if needed
        def prepare_tensor(tensor, normalize=True):
            if normalize and tensor.dtype == torch.uint8:
                return tensor.int() 
            return (tensor*255).int()
        
        depth_tensor = prepare_tensor(depth_gray)  # [H, W]
        color_tensor = prepare_tensor(image)      # [H, W, 3]
                
        # Add batch dimension for processing
        points = from2Dto3D_vectorized_torch(
            arrayDepth=depth_tensor,
            arrayPixel=color_tensor,
            maxZ=depth,
            quality=quality, 
            mask=alpha
        )  # [N, 6]
        
        
        
        # Add batch dimension to output
        if len(points.shape)>2:
            points = points.squeeze()
        return points[None, ...]
            
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",{}),
                "depth_image": ("IMAGE",{}),
                "depth" : ("INT", {"default": 1, "min":1, "max":1024}),
                "quality" : ("INT", {"default": 1, "min":1, "max":16}),
            },
        }

    RETURN_TYPES = ("Points3D",)
    FUNCTION = "imageTo3Dpoints"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"

class ImageToPointsTest:
                
    def __init__(self, device="cpu"):
        self.device = device
    def imageTo3Dpoints(self, image, depth_image,depth, quality, mask=None):
        returns = None
        
        if (image.cpu().numpy().shape[0])>1:
            raise Exception("Batch not work")
        
        for img, cols in zip(depth_image.cpu().numpy(), image.cpu().numpy()):
        
            if cols.shape[2]==4:
                cols = cv2.cvtColor(cols, cv2.COLOR_RGBA2RGB)
                
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            if mask!=None:
                mask = mask.detach().cpu().numpy()
            
            w,h = img.shape[0:2]
            
            cols = cv2.resize(cols, (h,w)) 
            
            
            points = from2Dto3D_vectorized(img*255,cols*255,depth,quality)
            
            
            
            if returns == None:
                returns = torch.from_numpy(np.array(points)[np.newaxis,:,:])
            
        
        return (returns)
            
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",{}),
                "depth_image": ("IMAGE",{}),
                "depth" : ("INT", {"default": 1, "min":1, "max":1024}),
                "quality" : ("INT", {"default": 1, "min":1, "max":16}),
            },
        }

    RETURN_TYPES = ("Points3D",)
    FUNCTION = "imageTo3Dpoints"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"

class TransformPoints:
                
    def __init__(self, device="cpu"):
        self.device = device
    def rotatePoints(self, points,rot_x,rot_y,rot_z,trl_x,trl_y,trl_z, scale_x, scale_y, scale_z):
        returns = None 
        rot_x = rot_x/180*m.pi
        rot_y = rot_y/180*m.pi
        rot_z = rot_z/180*m.pi
        
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,point in enumerate(points.detach().clone().cpu().numpy()[np.newaxis,:,:]):
            
            rotation = np.array([rot_x,rot_y,rot_z])
            translation=np.array([trl_x,trl_y,trl_z])
            scale = np.array([scale_x, scale_y, scale_z])
            
            points_rot = transform_points(point, translate=translation, rotate=rotation, scale=scale)
            
            if returns == None:
                returns = torch.from_numpy(np.array(points_rot)[np.newaxis,:,:])
        
        return (returns,)
            
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": ("Points3D",{}),
                "rot_x" : ("FLOAT", {"default": 0, "min": -360, "max": 360, "step":0.1}),
                "rot_y" : ("FLOAT", {"default": 0, "min": -360, "max": 360, "step":0.1}),
                "rot_z" : ("FLOAT", {"default": 0, "min": -360, "max": 360, "step":0.1}),
                "trl_x" : ("INT", {"default": 0, "min": -2048, "max": 2048, "step":1}),
                "trl_y" : ("INT", {"default": 0, "min": -2048, "max": 2048, "step":1}),
                "trl_z" : ("INT", {"default": 0, "min": -2048, "max": 2048, "step":1}),
                "scale_x" : ("FLOAT", {"default": 1, "min": -500, "max": 500, "step":0.01}),
                "scale_y" : ("FLOAT", {"default": 1, "min": -500, "max": 500, "step":0.01}),
                "scale_z" : ("FLOAT", {"default": 1, "min": -500, "max": 500, "step":0.01}),
            },
        }

    RETURN_TYPES = ("Points3D",)
    FUNCTION = "rotatePoints"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"
           
class PointsToImage_ortho:
                
    def __init__(self, device="cpu"):
        self.device = device
    def points2Img(self, images, points, color=False):
        returns = None 
           
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,points_rot in enumerate(points.detach().cpu().numpy()[np.newaxis,:,:]):
        
            
            img = cv2.cvtColor(images.cpu().numpy()[idx1], cv2.COLOR_RGBA2GRAY)
            h,w = img.shape
                           
            img_pil = render_points_fast(points_rot.copy(), img_size=(w,h), color=color)  
            
            img = np.array(img_pil)
            
                        
            if not(isinstance(returns, torch.Tensor)):
              returns=transform(img)
            else:
              returns = torch.stack((returns,transform(img)))
              
            
        out= returns if len(returns.shape)==4 else returns[None,:,:,:]
        out = out.permute(0,2,3,1)
        return (out,)
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",{}),
                "points": ("Points3D",{}),
                "color" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "points2Img"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"
         
class PointsToImage_proj:
                
    def __init__(self, device="cpu"):
        self.device = device
    def points2Img(self, images, points, color=False, fov=35):
        returns = None 
           
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,points_rot in enumerate(points.detach().cpu().numpy()[np.newaxis,:,:]):
            
            
            img = cv2.cvtColor(images.cpu().numpy()[idx1], cv2.COLOR_RGBA2GRAY)
            h,w = img.shape
                        
            translation = np.array([-w/2,-h/2,0])
            scale = np.array([1,1,-1])
            
            points_cop = transform_points(points_rot.copy(), translate=translation, scale=scale)
            translation = np.array([0,0,-max(w,h)/m.sin(m.radians(fov))*1.15])
            points_cop = transform_points(points_cop, translate=translation)
            
            
            
            img_pil = render_points_fast(points_cop, img_size=(w,h), color=color, fov=fov, cameraType=1)
            
            img = np.array(img_pil)
            
                        
            if not(isinstance(returns, torch.Tensor)):
              returns=transform(img)
            else:
              returns = torch.stack((returns,transform(img)))
              
        #print("ended")
            
        out= returns if len(returns.shape)==4 else returns[None,:,:,:]
        out = out.permute(0,2,3,1)
        return (out,)
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",{}),
                "points": ("Points3D",{}),
                "color" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "fov"   : ("FLOAT", {"default":35, "min":1, "max":2000, "step":0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "points2Img"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"
   
class PointsToImage_ortho_A:
                
    def __init__(self, device="cpu"):
        self.device = device
    def points2Img(self, images, points, color=False, correct=False, ksize=2, threshold=2):
        returns = None 
           
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,points_rot in enumerate(points.detach().cpu().numpy()[np.newaxis,:,:]):
        
            
            img = cv2.cvtColor(images.cpu().numpy()[idx1], cv2.COLOR_RGBA2GRAY)
            h,w = img.shape
            
            img_pil = render_points_fast(points_rot.copy(), img_size=(w,h), color=color, correction=correct, ksize=(ksize,ksize), thresh=threshold) 
            
            img = np.array(img_pil)
            
            
            
            if not(isinstance(returns, torch.Tensor)):
              returns=transform(img)
            else:
              returns = torch.stack((returns,transform(img)))
              
            
        out= returns if len(returns.shape)==4 else returns[None,:,:,:]
        out = out.permute(0,2,3,1)
        return (out,)
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",{}),
                "points": ("Points3D",{}),
                "color" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "correct" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "ksize": ("INT", {"default":2, "min":2,"max":128}),
                "threshold": ("INT", {"default":2, "min":0,"max":256}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "points2Img"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"
         
class PointsToImage_proj_A:
                
    def __init__(self, device="cpu"):
        self.device = device
    def points2Img(self, images, points, color=False, fov=35, correct=False, ksize=2, threshold=2):
        returns = None 
           
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,points_rot in enumerate(points.detach().cpu().numpy()[np.newaxis,:,:]):
            
            
            img = cv2.cvtColor(images.cpu().numpy()[idx1], cv2.COLOR_RGBA2GRAY)
            h,w = img.shape
            
            
            translation = np.array([-w/2,-h/2,0])
            scale = np.array([1,1,-1])
            
            points_cop = transform_points(points_rot.copy(), translate=translation, scale=scale)
            translation = np.array([0,0,-max(w,h)/m.sin(m.radians(fov))*1.15])
            points_cop = transform_points(points_cop, translate=translation)
            
            
            
            img_pil = render_points_fast(points_cop.copy(), img_size=(w,h), color=color, fov=fov, cameraType=1, correction=correct, ksize=(ksize,ksize), thresh=threshold)  
            
            img = np.array(img_pil)
                           
                        
            if not(isinstance(returns, torch.Tensor)):
              returns=transform(img)
            else:
              returns = torch.stack((returns,transform(img)))
            
            
        out= returns if len(returns.shape)==4 else returns[None,:,:,:]
        out = out.permute(0,2,3,1)
        
        return (out,)
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",{}),
                "points": ("Points3D",{}),
                "color" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "fov"   : ("FLOAT", {"default":35, "min":1, "max":2000, "step":0.1}),
                "correct" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "ksize": ("INT", {"default":2, "min":2,"max":128}),
                "threshold": ("INT", {"default":2, "min":0,"max":256}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "points2Img"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"

class PointsToImage_TEST:
                
    def __init__(self, device="cpu"):
        self.device = device
    def points2Img(self, images, points, color=False, fov=35, correct=False, ksize=2, threshold=2, camera = "Projection", panda=True):
        returns = None 
           
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,points_rot in enumerate(points.detach().cpu().numpy()[np.newaxis,:,:]):
            
            
            img = cv2.cvtColor(images.cpu().numpy()[idx1], cv2.COLOR_RGBA2GRAY)
            h,w = img.shape
            
            
            translation = np.array([-w/2,-h/2,0])
            scale = np.array([1,1,-1])
            
            points_cop = transform_points(points_rot.copy(), translate=translation, scale=scale)
            translation = np.array([0,0,-max(w,h)/m.sin(m.radians(fov))*1.15])
            points_cop = transform_points(points_cop, translate=translation)
            
            
            
            img_pil = render_points_fast(points_cop.copy(), img_size=(w,h), color=color, fov=fov, cameraType=1 if "Pro" in camera else 0, correction=correct, ksize=(ksize,ksize), thresh=threshold, TEST=panda)  
            
            img = np.array(img_pil)
                           
                        
            if not(isinstance(returns, torch.Tensor)):
              returns=transform(img)
            else:
              returns = torch.stack((returns,transform(img)))
            
            
        out= returns if len(returns.shape)==4 else returns[None,:,:,:]
        out = out.permute(0,2,3,1)
        
        return (out,)
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",{}),
                "points": ("Points3D",{}),
                "color" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "fov"   : ("FLOAT", {"default":35, "min":1, "max":2000, "step":0.1}),
                "correct" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "ksize": ("INT", {"default":2, "min":2,"max":128}),
                "threshold": ("INT", {"default":2, "min":0,"max":256}),
                "camera" : (["Orthographic","Projection"],),
                "panda" : ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "points2Img"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"

        
class CubeCut:
    def __init__(self, device="cpu"):
        self.device = device
    def function_do(self, points, x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100):
        returns = None 
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,point in enumerate(points.detach().clone().cpu().numpy()[np.newaxis,:,:]):
            
            xm,xM, ym,yM, zm,zM = (point[:,0].min(),point[:,0].max(), point[:,1].min(),point[:,1].max(), point[:,2].min(),point[:,2].max())
            p_x, p_y, p_z = ((xM-xm)/100, (yM-ym)/100, (zM-zm)/100)
            
                        
            mask = (point[:,0]>=(x_min*p_x)+xm) & (point[:,0]<=(x_max*p_x)+xm) & (point[:,1]>=(y_min*p_y)+ym) & (point[:,1]<=(y_max*p_y)-ym) & (point[:,2]>=(z_min*p_z)+zm) & (point[:,2]<=(z_max*p_z)+zm)
            
            points_clean = point[mask]
            
            if returns == None:
                returns = torch.from_numpy(np.array(points_clean)[np.newaxis,:,:])
                
        return (returns,)
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": ("Points3D",{}),
                "x_min" : ("FLOAT", {"default": 0, "min": 0, "max": 100, "step":0.001}),
                "x_max" : ("FLOAT", {"default": 100, "min": 0, "max": 100, "step":0.001}),
                "y_min" : ("FLOAT", {"default": 0, "min": 0, "max": 100, "step":0.001}),
                "y_max" : ("FLOAT", {"default": 100, "min": 0, "max": 100, "step":0.001}),
                "z_min" : ("FLOAT", {"default": 0, "min": 0, "max": 100, "step":0.001}),
                "z_max" : ("FLOAT", {"default": 100, "min": 0, "max": 100, "step":0.001}),
            },
        }
    
    RETURN_TYPES = ("Points3D",)
    FUNCTION = "function_do"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"
 
class CleanPointsKDTree:
    def __init__(self, device="cpu"):
        self.device = device
    def function_do(self, points, k, m):
        returns = None 
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,point in enumerate(points.detach().clone().cpu().numpy()[np.newaxis,:,:]):
            
            points_clean = clean_points(point, k, m)
            
            
            if returns == None:
                returns = torch.from_numpy(np.array(points_clean)[np.newaxis,:,:])
        
        
        return (returns,)
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": ("Points3D",{}),
                "k" : ("INT", {"default": 20, "min": 0, "max": 32, "step":1}),
                "m" : ("FLOAT", {"default": 16, "min": 0, "max": 1024, "step":0.01}),
            },
        }
    
    RETURN_TYPES = ("Points3D",)
    FUNCTION = "function_do"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"

class InterpolatePointsCKDTree:
    def __init__(self, device="cpu"):
        self.device = device
    def function_do(self, points, value, n):
        returns = None 
        
        if len(points.shape)>2:
            points = points.squeeze()
        
        for idx1,point in enumerate(points.detach().clone().cpu().numpy()[np.newaxis,:,:]):
            
            points_int = interpolate_points(point, value, n)
            
            if returns == None:
                returns = torch.from_numpy(np.array(points_int)[np.newaxis,:,:])
        
        return (returns,)
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": ("Points3D",{}),
                "value" : ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step":0.01}),
                "n" : ("INT", {"default": 3, "min": 0, "max": 32, "step":1}),
            },
        }
    
    RETURN_TYPES = ("Points3D",)
    FUNCTION = "function_do"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"

class exportToPLY:
    def __init__(self, device="cpu"):
        self.device = device
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        
    def function_do(self, points, multiple_files=False, format_out="ascii", filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        if filename_prefix.endswith(".ply"):
            filename_prefix = filename_prefix.split(".ply")[0]
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        
    
        if len(points.shape)>2:
            points = points.squeeze()
        
        file = f"{filename}_{counter:05}_.ply"

        results: list[FileLocator] = []
        
        file = os.path.join(full_output_folder, file)
        
        for idx1,point in enumerate(points.detach().clone().cpu().numpy()[np.newaxis,:,:]):
            str_out = export_PLY(point, file, multiple_files=multiple_files, format_ascii = format_out)
            
        return {"":""}
        
        
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points": ("Points3D",{}),
                "multiple_files": ("BOOLEAN", {"default": False, "label_off": "OFF", "label_on": "ON"}),
                "format_out": (["ascii","binary_little_endian"],),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "function_do"
    OUTPUT_NODE = True

    CATEGORY = "depthMapOperation"
    
class importPLY:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".ply")]
        return {"required":
                    {"points": (sorted(files), {"points_upload": True})},
                }

    CATEGORY = "depthMapOperation"

    RETURN_TYPES = ("Points3D",)
    FUNCTION = "do_imp"
    def do_imp(self, points):
        points_path = folder_paths.get_annotated_filepath(points)
        
        type_ = "ascii"
                
        with open(points_path, "r", errors="ignore") as f:
            for line in f:
                try:
                    line=line.rstrip()
                except:
                    break
                if "format" in line:
                    type_ = line.split(" ")[1]
                    break
                elif "end_header" in line:
                    break
        if "ascii" in type_:
            points_cloud = read_ply_ascii(points_path)
        elif "little_endian" in type_:
            points_cloud = read_ply_binary(points_path)
        else:
            raise ValueError("PLY format file error, it should be 'ascii' or 'binary_little_endian'")
            
        mx, Mx, my, My, mz, Mz = (points_cloud[:,0].min(), points_cloud[:,0].max(), points_cloud[:,1].min(), points_cloud[:,1].max(), points_cloud[:,2].min(), points_cloud[:,2].max())
        
        tx = -mx
        ty = -my
        
        points_cloud = transform_points(points_cloud, translate=np.array([tx,ty,0]))
        
        
        return torch.from_numpy(np.array(points_cloud)[np.newaxis,:,:])
        

    @classmethod
    def IS_CHANGED(s, points):
        return points

    @classmethod
    def VALIDATE_INPUTS(s, points):
        if not folder_paths.exists_annotated_filepath(points):
            return "Invalid points file: {}".format(points)

        return True
    
class CloudPointsInfo:
    def __init__(self, device="cpu"):
        self.device = device
        
    def function_do(self, points):
        points = points.squeeze()
        print(points.shape)
        str_ = f"Points : {points.shape[1]}\nX : [{points[:,0].min()}, {points[:,0].max()}]\nY : [{points[:,1].min()}, {points[:,1].max()}]\nX : [{points[:,2].min()}, {points[:,2].max()}]"
        
        return {"ui": {"text": (str_,)}, "result": (str_,)}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "points": ("Points3D",{}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "function_do"
    OUTPUT_NODE = True
    CATEGORY = "depthMapOperation"
    
NODE_CLASS_MAPPINGS = {
    #"ImageToPoints (Legacy)" :ImageToPoints,       #old for loop
    #"ImageToPoints" :ImageToPointsTest,            #old numpy only
    "CloudPointsInfo" : CloudPointsInfo,
    "ImageToPoints (Torch)" :ImageToPointsTorch,
    "Export to PLY":exportToPLY,
    "Import PLY":importPLY,
    "CubeLimit":CubeCut,
    "CleanPoints (KDTree)":CleanPointsKDTree,
    "InterpolatePoints (KDTree)":InterpolatePointsCKDTree,
    "TransformPoints":TransformPoints,
    "PointsToImage (Orthographic)":PointsToImage_ortho,
    "PointsToImage (Projection)":PointsToImage_proj,
    "PointsToImage advance (Orthographic)":PointsToImage_ortho_A,
    "PointsToImage advance (Projection)":PointsToImage_proj_A,
    "PointsToImage advance (DEBUG)":PointsToImage_TEST,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CloudPointsInfo" : "Cloud Points Info",
    "ImageToPoints (Torch)" : "Image To Points (Torch)",
    "Export to PLY": "Export to PLY",
    "Import PLY": "Import from PLY",
    "CubeLimit": "Cube Limit",
    "CleanPoints (KDTree)": "Clean Points (KDTree)",
    "InterpolatePoints (KDTree)":"Interpolate Points (KDTree)",
    "TransformPoints":"Transform Points",
    "PointsToImage (Orthographic)": "Points To Image (Orthographic)",
    "PointsToImage (Projection)":"Points To Image (Projection)",
    "PointsToImage advance (Orthographic)":"Points To Image advance (Orthographic)",
    "PointsToImage advance (Projection)":"Points To Image advance (Projection)",
    "PointsToImage advance (DEBUG)":"Points To Image (DEBUG)"
    
}