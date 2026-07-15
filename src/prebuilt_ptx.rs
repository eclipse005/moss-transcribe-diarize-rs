//! Precompiled multi-arch PTX selection (scheme B).
//!
//! GPU kernels are compiled offline with `nvcc -ptx` (see `scripts/compile-ptx.ps1`)
//! and embedded here. At runtime we pick the highest prebuilt arch that is
//! ≤ the device's compute capability. This removes any need for NVRTC / CUDA
//! Toolkit on end-user machines.

/// Supported prebuilt compute capabilities, ascending.
pub const PREBUILT_SMS: &[u32] = &[61, 70, 75, 80, 86, 89, 90];

/// Minimum supported device CC (must match first entry of [`PREBUILT_SMS`]).
pub const MIN_SM: u32 = 61;

/// Encode `(major, minor)` as a single integer, e.g. `(8, 6) -> 86`.
#[inline]
pub fn sm_code(major: i32, minor: i32) -> u32 {
    (major as u32) * 10 + (minor as u32)
}

/// Choose the highest prebuilt SM ≤ `device_sm`.
///
/// Returns `None` if the device is below [`MIN_SM`].
pub fn select_prebuilt_sm(device_sm: u32) -> Option<u32> {
    let mut best = None;
    for &sm in PREBUILT_SMS {
        if sm <= device_sm {
            best = Some(sm);
        } else {
            break;
        }
    }
    best
}

/// Embedded PTX source for the given prebuilt SM code, or `None` if unknown.
pub fn ptx_for_sm(sm: u32) -> Option<&'static str> {
    match sm {
        61 => Some(include_str!("../ptx/kernels_sm61.ptx")),
        70 => Some(include_str!("../ptx/kernels_sm70.ptx")),
        75 => Some(include_str!("../ptx/kernels_sm75.ptx")),
        80 => Some(include_str!("../ptx/kernels_sm80.ptx")),
        86 => Some(include_str!("../ptx/kernels_sm86.ptx")),
        89 => Some(include_str!("../ptx/kernels_sm89.ptx")),
        90 => Some(include_str!("../ptx/kernels_sm90.ptx")),
        _ => None,
    }
}

/// Resolve PTX text for a live device `(major, minor)`.
pub fn resolve_ptx_for_device(major: i32, minor: i32) -> Result<(&'static str, u32), String> {
    let device_sm = sm_code(major, minor);
    let selected = select_prebuilt_sm(device_sm).ok_or_else(|| {
        format!(
            "GPU compute capability sm_{device_sm} is below the minimum supported sm_{MIN_SM}"
        )
    })?;
    let ptx = ptx_for_sm(selected).ok_or_else(|| {
        format!("internal error: missing embedded PTX for sm_{selected}")
    })?;
    Ok((ptx, selected))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_exact_and_floor() {
        assert_eq!(select_prebuilt_sm(61), Some(61));
        assert_eq!(select_prebuilt_sm(86), Some(86));
        assert_eq!(select_prebuilt_sm(89), Some(89));
        assert_eq!(select_prebuilt_sm(87), Some(86)); // between 86 and 89
        assert_eq!(select_prebuilt_sm(75), Some(75));
        assert_eq!(select_prebuilt_sm(100), Some(90)); // newer than list → highest
    }

    #[test]
    fn rejects_below_minimum() {
        assert_eq!(select_prebuilt_sm(60), None);
        assert_eq!(select_prebuilt_sm(50), None);
        assert!(resolve_ptx_for_device(5, 0).is_err());
    }

    #[test]
    fn embeds_nonempty_ptx_for_all_prebuilt() {
        for &sm in PREBUILT_SMS {
            let ptx = ptx_for_sm(sm).unwrap_or_else(|| panic!("missing PTX sm_{sm}"));
            assert!(
                ptx.len() > 100 && ptx.contains(".version"),
                "PTX for sm_{sm} looks empty/invalid (len={})",
                ptx.len()
            );
        }
    }

    #[test]
    fn resolve_p104_and_3060ti() {
        let (_, sm61) = resolve_ptx_for_device(6, 1).unwrap();
        assert_eq!(sm61, 61);
        let (_, sm86) = resolve_ptx_for_device(8, 6).unwrap();
        assert_eq!(sm86, 86);
    }

    /// GPU smoke: load prebuilt PTX via the same path as engine init.
    #[test]
    fn load_prebuilt_ptx_on_device_if_available() {
        use cudarc::driver::CudaContext;
        use cudarc::nvrtc::Ptx;
        use std::sync::Arc;

        let ctx = match CudaContext::new(0) {
            Ok(c) => Arc::new(c),
            Err(e) => {
                eprintln!("skip GPU PTX load smoke (no device): {e:?}");
                return;
            }
        };
        let (major, minor) = ctx.compute_capability().expect("compute_capability");
        let (ptx_src, selected) =
            resolve_ptx_for_device(major, minor).expect("resolve PTX for live device");
        eprintln!("device sm_{major}{minor} -> prebuilt sm_{selected}");
        let module = ctx
            .load_module(Ptx::from_src(ptx_src))
            .expect("load_module prebuilt PTX");
        let _ = module
            .load_function("rms_norm_f16")
            .or_else(|_| module.load_function("rms_norm_bf16"))
            .or_else(|_| module.load_function("layer_norm_f16"))
            .expect("load a known kernel from prebuilt PTX");
    }
}
