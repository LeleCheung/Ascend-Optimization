"""Device visibility variable names shared by wire validation and KGS runtime."""

BACKEND_VISIBILITY_VARIABLES: dict[str, tuple[str, ...]] = {
    "cuda": ("CUDA_VISIBLE_DEVICES",),
    "npu": ("ASCEND_RT_VISIBLE_DEVICES",),
    "musa": ("MUSA_VISIBLE_DEVICES", "MTHREADS_VISIBLE_DEVICES"),
    "mlu": ("MLU_VISIBLE_DEVICES",),
    "hygon": ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"),
    # MetaX honors both; narrow every configured variable for child isolation.
    "metax": ("CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES"),
    "iluvatar": ("CUDA_VISIBLE_DEVICES",),
    "kunlunxin": ("CUDA_VISIBLE_DEVICES",),
    "thead": ("CUDA_VISIBLE_DEVICES",),
    "txda": ("TXDA_VISIBLE_DEVICES",),
    "enflame": ("TOPS_VISIBLE_DEVICES",),
}

ALL_VISIBILITY_VARIABLES = frozenset(
    variable
    for variables in BACKEND_VISIBILITY_VARIABLES.values()
    for variable in variables
)
