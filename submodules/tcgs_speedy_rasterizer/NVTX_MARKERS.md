# NVTX Markers for TCGS (Tensor-Core Gaussian Splatting) Profiling

This document describes the NVTX markers added to the tcgs_speedy_rasterizer module for profiling with nsys.

## File Modified

- `cuda_rasterizer/rasterizer_impl.cu`

## NVTX Markers

### 1. Preprocessing
**Marker:** `Preprocessing`

**Location:** Around `FORWARD::preprocess()` call

**Function:** Performs per-Gaussian operations before rasterization:
- Transform 3D Gaussians to 2D screen space
- Compute 3D covariance matrices from scale/rotation
- Project to 2D covariance matrices
- Frustum culling (remove Gaussians outside view)
- Convert spherical harmonics to RGB colors
- Compute exact tile overlap using tight culling

### 2. TileBinning
**Marker:** `TileBinning`

**Location:** Around prefix sum and `duplicateWithKeys()` calls

**Function:** Assigns Gaussians to screen tiles:
- Compute prefix sum of tiles touched per Gaussian
- Generate tile-Gaussian key-value pairs with tight bounds
- Uses per-Gaussian conic for precise tile overlap detection

### 3. Sorting
**Marker:** `Sorting`

**Location:** Around `cub::DeviceRadixSort::SortPairs()` and `identifyTileRanges()` calls

**Function:** Sorts Gaussians for front-to-back rendering:
- Radix sort by tile ID and depth
- Identify start/end ranges for each tile's Gaussian list

### 4. AlphaBlending
**Marker:** `AlphaBlending`

**Location:** Around `TCGS::renderCUDA_Forward()` call (or `FORWARD::render()` if USE_TCGS=0)

**Function:** TCGS tensor-core accelerated alpha compositing:
- Converts Gaussian primitives to vectorized format
- Uses tensor cores (MMA instructions) for batch exponent computation
- Processes 16 Gaussians in parallel per warp
- Combined culling and alpha blending in `culling_and_blending()`
- FP16 accumulation for colors

## TCGS-Specific Operations (inside AlphaBlending)

The AlphaBlending marker includes TCGS-specific operations:
- `transform_coefs()`: Preprocess conics for log-space computation
- `gs2vec()`: Convert Gaussians to vectorized matrix format
- `pix2vec()`: Convert pixels to vectorized matrix format
- MMA operations: `mma_16x8x8_f16_f16` for batch exponent calculation
- `culling_and_blending()`: Per-pixel alpha test and accumulation

## Usage with nsys

```bash
nsys profile --trace=cuda,nvtx -o profile_output python render.py ...
```

Then view in Nsight Systems GUI to see timing breakdown by marker.

