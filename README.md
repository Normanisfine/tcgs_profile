# TC-GS Profiling Fork

This is a modified fork of [TC-GS (Tensor-Core Gaussian Splatting)](https://github.com/DeepLink-org/3DGSTensorCore) with **NVTX profiling markers** and **CUDA 11.6 compatibility fixes** for performance analysis with NVIDIA Nsight Systems.

## What I Modified

### 1. NVTX Profiling Markers

Added NVTX markers to the CUDA rasterizer to measure per-stage GPU kernel time:

| Stage | Marker | What It Measures |
|-------|--------|------------------|
| **Preprocessing** | `Preprocessing` | 3D→2D projection, tight tile culling (SnugBox) |
| **TileBinning** | `TileBinning` | Prefix sum, ellipse-tile intersection |
| **Sorting** | `Sorting` | Radix sort by tile ID and depth |
| **AlphaBlending** | `AlphaBlending` | Tensor-core accelerated alpha compositing |

**File Modified**: `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/rasterizer_impl.cu`

### 2. CUDA 11.6 Compatibility Fix

Fixed compilation error with CUDA 11.6 where `__hmin` intrinsic is undefined (introduced in CUDA 12.0).

**Solution**: Added compatibility macro in `tcgs_utils.h`:
```cuda
#if (!defined(CUDA_VERSION) || CUDA_VERSION < 12000)
  #ifndef __hmin
    #define __hmin(a, b) ((__hlt((a), (b))) ? (a) : (b))
  #endif
#endif
```

**File Modified**: `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/tcgs/tcgs_utils.h`

## Key Files

| File | Description |
|------|-------------|
| [`CHANGELOG.md`](CHANGELOG.md) | Documents the CUDA 11.6 compatibility fix |
| [`submodules/tcgs_speedy_rasterizer/NVTX_MARKERS.md`](submodules/tcgs_speedy_rasterizer/NVTX_MARKERS.md) | Documentation of NVTX markers |
| [`submodules/tcgs_speedy_rasterizer/cuda_rasterizer/rasterizer_impl.cu`](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/rasterizer_impl.cu) | Modified CUDA code with NVTX markers |
| [`submodules/tcgs_speedy_rasterizer/cuda_rasterizer/tcgs/tcgs_utils.h`](submodules/tcgs_speedy_rasterizer/cuda_rasterizer/tcgs/tcgs_utils.h) | CUDA 11.6 compatibility fix |

## Usage

### Run Rendering
```bash
python render.py -m <path_to_trained_model> --iteration 30000
```

### Profile with Nsight Systems
```bash
nsys profile --trace=cuda,nvtx -o profile_output \
    python render.py -m <path_to_trained_model> --iteration 30000
```

Then open the `.nsys-rep` file in Nsight Systems GUI to see timing breakdown by NVTX marker.

## Profiling Results (Garden Scene, 1036×1600)

| Stage | GPU Time |
|-------|----------|
| Preprocessing | 1.17 ms |
| TileBinning | 2.94 ms |
| Sorting | 0.12 ms |
| AlphaBlending | 6.18 ms |
| **Total** | **~10.4 ms** (~124 FPS) |

### Speedup vs Original 3DGS

| Stage | Speedup | Key Optimization |
|-------|---------|------------------|
| Preprocessing | 2.3× | Tighter culling |
| Sorting | **35×** | SnugBox reduces tile-Gaussian pairs by 30-50% |
| AlphaBlending | 1.9× | Tensor Core MMA (16 Gaussians batched) |
| **Total** | **1.9×** | GPU kernel time |
| **End-to-End** | **3.5×** | FPS (includes pipeline efficiency gains) |

---

## Original Repository

This fork is based on TC-GS:
- **Paper**: [TC-GS: A Faster Gaussian Splatting Module Utilizing Tensor Cores](https://arxiv.org/abs/2505.24796)
- **Original Repo**: https://github.com/DeepLink-org/3DGSTensorCore

```bibtex
@article{tcgs2025,
  title={TC-GS: A Faster Gaussian Splatting Module Utilizing Tensor Cores},
  author={TC-GS Team},
  journal={arXiv preprint arXiv:2505.24796},
  year={2025}
}
```
