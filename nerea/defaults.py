DEFAULT_BIN_KWARGS = {"bins": None, "smooth": True, 'renormalize': True}
DEFAULT_MAX_KWARGS = {"fst_ch": None}
DEFAULT_SMOOTH_KWARGS = {"smoothing_method": 'savgol_filter', "window_length": 10, "polyorder": 4, 'renormalize': True}
DEFAULT_DTC_KWARGS = {"tau_p": 88e-9, "tau_np": 108e-9}
DEFAULT_AC_KWARGS = {"t_left": 3e-2, "t_right": 1e-2, "smooth_kwargs": DEFAULT_SMOOTH_KWARGS, "dtc_kwargs": DEFAULT_DTC_KWARGS}
