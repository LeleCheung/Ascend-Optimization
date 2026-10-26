# 独立进程复核相邻行复用候选

scan_launch：SUCCEEDED，job cbbf99e6c04a4d478a1c5133fd174c72。GRAYLIST [cumulative-trapezoid-cann-launch-deterministic] 17 hit(s): graylist Tensor method: main.py:84 .data_ptr; graylist Tensor method: main.py:85 .stride; graylist Tensor method: main.py:86 .stride; graylist Tensor method: main.py:144 .is_contiguous; graylist Tensor method: main.py:144 .storage_offset; graylist Tensor method: main.py:145 .dim; graylist Tensor method: main.py:146 .stride; graylist Tensor method: main.py:153 .data_ptr; graylist Tensor method: main.py:166 .numel; graylist Tensor method: main.py:171 .is_contiguous; graylist Tensor method: main.py:173 .numel; graylist Tensor method: main.py:174 .is_contiguous; graylist Tensor method: main.py:175 .numel; graylist Tensor method: main.py:176 .numel; graylist Tensor method: main.py:178 .numel; graylist Tensor method: main.py:179 .stride; graylist Tensor method: main.py:204 .data_ptr
/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/npu/__init__.py:539: UserWarning: You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. The return value of get_device_capability is only for compatibility with PyTorch and does not represent the actual capability of the NPU device.
  warnings.warn(f"You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. {warning_str}")
/data/hanle/ascend-optimization/Ascend-Optimization/third_party/FlagGems/src/flag_gems/utils/device_info.py:58: UserWarning: [device_info] torch_device_fn.get_device_capability returned None for device_id=0, fallback to (0, 0).
  warnings.warn(


