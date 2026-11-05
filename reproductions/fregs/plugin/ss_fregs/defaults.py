"""Every FreGS hyper-parameter default lives here (single place to edit after calibration).

Status of each value (see README and plan.md section 3/4/6):
  paper       - stated in arXiv 2403.06908
  assumed     - not given by the paper; written default
  PLACEHOLDER - not given; to be replaced by the calibration scripts
                (calibrate_weights.py for w_l/w_h, calibrate_threshold.py for grow_grad2d)
"""

DEFAULTS = {
    "density": {
        "absgrad": False,  # paper: 3DGS signed view-space gradient
        # CALIBRATED: evidence/threshold_calibration.json (count-matched to gsplat at step 15000,
        # same 15k sweep, bicycle/garden/room; interpolated between 0.0003 and 0.0004)
        "grow_grad2d": 0.00035,
    },
    "frequency": {
        # CALIBRATED: evidence/weight_calib_v2/merged.json (median over bicycle/garden/room)
        "w_l": 3.400926541615607e-06,  # Fig. 2 means2d ratio 1.7 at step 1, train views
        "w_h": 5.568719487276853e-07,  # w_l * median(M_low/M_high), t=T, 15k train-view ref
        "d0_frac": 0.10,  # assumed: low-pass radius D0 = d0_frac * D
        "t0": 3000,  # assumed: T0, start of the high-frequency band
        "t_end": 15000,  # assumed (bounded by paper): T, band reaches D
        "stop_iter": 15000,  # paper Sec. 4.1: FR stops when densification ends
        "anneal": True,  # paper Sec. 3.3 (FA); False = Base+FR ablation
        "mask": "ideal_circular",  # assumed
        "radius_units": "index",  # assumed: origin (H/2, W/2), index units, D = corner
        "channels": "per_channel_sum",  # assumed
        "phase": "wrapped",  # iteration 1: Eq. 6 with wrapped difference angle(F_hat conj F); "raw" = i0
        "phase_eps": 1e-6,  # assumed numerical safeguard (not in paper)
        "normalization": "paper",  # paper Eq. 5/6: (1/sqrt(HW)) * sum, unnormalized DFT
    },
}

# Flip to True when the calibration result has been written into DEFAULTS above.
CALIBRATED = {"w_l": True, "w_h": True, "grow_grad2d": True}  # weight_calib_v2/merged.json; threshold_calibration.json

# Threshold calibration grid (plan 6.2); each value gets an entry point fregs-calib-gXXXX.
THRESHOLD_GRID = (0.0003, 0.0004, 0.0005, 0.0006)

# calibrate_weights.py (statistic: per-Gaussian screen-space gradient norm, densification's grad2d).
# w_l: Fig. 2 iteration-0 ratio 1.7 (both regions alike), step-1 state.
# w_h: per-band unit-gradient equalization on the 15k train-view reference at t = T (review-2):
#      w_h = w_l * M_low / M_high. The resulting whole-image t = T ratio is compared with the
#      Fig. 2 12k-15k envelope (well ~1.15 .. over ~2.3); a miss is a recorded deviation.
#      Whole-image targets 1.15/1.5/2.0 are diagnostics only.
WEIGHT_TARGETS = {
    "w_l": 1.7,
    "w_h_rule": "equalize_band_screen_gradient",
    "fig2_envelope_tT": (1.15, 2.3),
    "w_h_sensitivity": (1.15, 1.5, 2.0),
}
