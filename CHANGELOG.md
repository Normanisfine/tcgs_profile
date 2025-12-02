# Changelog

## [2025-01-XX] - CUDA 11.6 Compatibility Fix

### Problem
Compilation of `tcgs_speedy_rasterizer` failed with CUDA 11.6 with the following error:
```
cuda_rasterizer/tcgs/tcgs_forward.cu(134): error: identifier "__hmin" is undefined
cuda_rasterizer/tcgs/tcgs_forward.cu(145): error: identifier "__hmin" is undefined
```

### Root Cause
The `__hmin` intrinsic function was introduced in CUDA 12.0 and is not available in CUDA 11.6. The code in `tcgs_forward.cu` uses `__hmin` to compute the minimum of two half-precision values, but this function doesn't exist in CUDA 11.6.

### Solution
Added a compatibility wrapper in `cuda_rasterizer/tcgs/tcgs_utils.h` that defines `__hmin` as a macro for CUDA versions < 12.0:

```cuda
// Compatibility wrapper for __hmin (not available in CUDA 11.6, introduced in CUDA 12.0)
// Define as macro to avoid function redefinition conflicts
#if (!defined(CUDA_VERSION) || CUDA_VERSION < 12000)
  #ifndef __hmin
    #define __hmin(a, b) ((__hlt((a), (b))) ? (a) : (b))
  #endif
#endif
```

### Implementation Details
- **Location**: `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/tcgs/tcgs_utils.h`
- **Approach**: Macro-based implementation using `__hlt` (half less-than) intrinsic, which is available in CUDA 11.6
- **Behavior**: Functionally equivalent to the native `__hmin` - returns the minimum of two half-precision values
- **Compatibility**: Works with CUDA 11.6 and doesn't interfere with CUDA 12.0+ where `__hmin` is natively available

### Testing
- Compilation should now succeed with CUDA 11.6.124
- Functionality remains unchanged - the macro computes `min(a, b)` correctly

### Files Modified
- `submodules/tcgs_speedy_rasterizer/cuda_rasterizer/tcgs/tcgs_utils.h`

---

