# 当前运行时接口事实

原件SHA256：ad1f4116ad7c26194e4600f95a9a5dedba047110de7e3bd6f1c6938e2578dd01

## triton.backends.ascend.driver
来源：/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/driver.py；SHA256：dfa7d353e4d847abcb2348a6cb9f616dba7d4b9b62f833d691f14ff0fe27b74c
```python
218:         else:
219:             arch = self.utils.get_arch()
220:         warp_size = 0
221:         return GPUTarget(backend, arch, warp_size)
222: 
223:     def get_current_device(self):
224:         """
225:         Get current device
226:         """
227:         return get_backend_func("get_current_device")
228: 
229:     def get_active_torch_device(self):
230:         import torch
231:         import torch_npu
232:         return torch.device("npu", self.get_current_device())
233: 
234:     def set_current_device(self, device):
235:         """
236:         Set current device as the given device
237:         """
238:         return get_backend_func("set_current_device", device)
239: 
240:     def get_current_stream(self, device: Optional[int] = None) -> int:
241:         """
242:         Get stream for current device
243:         """
244:         # According to torch_npu, the content of a torch.npu.Stream is essentilly an rtStream_t
245:         # TODO: use CANN API instead of torchnpu
246:         return get_backend_func("get_current_stream", device)
247: 
248:     def get_benchmarker(self):
249:         from triton.testing import do_bench
250:         return do_bench
251: 
252:     def get_device_interface(self):
253:         return get_backend_func("get_device_interface")
254: 
255:     def get_empty_cache_for_benchmark(self):
256:         cache_size = 192 * 1024 * 1024
257:         return get_backend_func("get_empty_tensor", cache_size // 4)
258: 
259:     def clear_cache(self, cache):
260:         cache.zero_()
261: 
262: 
263: def _precompile_npu_ext_with_lock(header_src, enable_precompile):
410:         read_header('internal/debug_tunnel/tunnel_impl.h')
411:     ])
412: 
413: 
414: def generate_npu_header_src():
415:     enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')
416:     return f"""
417: /*
418:  * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
419:  * Copyright 2018-2020 Philippe Tillet
420:  * Copyright 2020-2022 OpenAI
421:  *
422:  * Permission is hereby granted, free of charge, to any person obtaining a copy
423:  * of this software and associated documentation files (the "Software"), to deal
424:  * in the Software without restriction, including without limitation the rights
425:  * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
426:  * copies of the Software, and to permit persons to whom the Software is
427:  * furnished to do so, subject to the following conditions:
428:  *
429:  * The above copyright notice and this permission notice shall be included in
430:  * all copies or substantial portions of the Software.
431:  *
432:  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
433:  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
434:  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
435:  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
436:  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
437:  * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
438:  * THE SOFTWARE.
480:         "fp64": "double",
481:     }[ty]
482: 
483: 
484: # the template is from triton-adapter HEAD. Wrapping the generated kernel binary into a python module
485: def make_launcher(constants, signature, metadata):
486:     import os
487:     workspace_size = int(metadata.workspace_size) \
488:                           if hasattr(metadata, 'workspace_size') else -1
489:     lock_init_value = int(metadata.lock_init_value) \
490:                           if hasattr(metadata, 'lock_init_value') else 0
491:     lock_num = int(metadata.lock_num) \
492:                           if hasattr(metadata, 'lock_num') else -1
493:     bs_task_type = metadata.bs_task_type if hasattr(metadata, 'bs_task_type') else 0
494:     mix_mode = metadata.mix_mode
495:     compile_on_910_95 = metadata.compile_on_910_95
496:     parallel_mode = metadata.parallel_mode
497:     enable_simt = ("simt" in parallel_mode) or metadata.force_simt_only
498: 
499:     def _serialize_signature(sig):
500:         if isinstance(sig, tuple):
501:             return ','.join(map(_serialize_signature, sig))
502:         return sig
503: 
504:     def _extracted_type(ty):
505:         if isinstance(ty, tuple):
506:             val = ','.join(map(_extracted_type, ty))
507:             return f"[{val}]"
508:         if ty[0] == '*':
509:             return "PyObject*"
510:         if ty in ("constexpr"):
511:             return "PyObject*"
512:         return ty_to_cpp(ty)
513: 
514:     def format_of(ty):
515:         if isinstance(ty, tuple):
516:             val = ''.join(map(format_of, ty))
517:             return f"({val})"
518:         if ty[0] == '*':
519:             return "O"
520:         if ty in ("constexpr"):
521:             return "O"
522:         if ty == "void*":
523:             return "O"
524:         return {
525:             "float": "f",
526:             "double": "d",
527:             "long": "l",
528:             "int8_t": "b",
529:             "int16_t": "h",
530:             "int32_t": "i",
531:             "int64_t": "L",
532:             "uint8_t": "B",
533:             "uint16_t": "H",
534:             "uint32_t": "I",
535:             "uint64_t": "K",
536:         }[ty_to_cpp(ty)]
537: 
538:     def _format_of_msprof_task_type_ratio(bs_task_type, mix_mode):
539:         # Default fallback based on mix_mode
540:         default_task_type = "MSPROF_GE_TASK_TYPE_AIV" if mix_mode == "aiv" else "MSPROF_GE_TASK_TYPE_AI_CORE"
541: 
542:         if not bs_task_type:
543:             return default_task_type, 0
562:         PyObject* launch_enter_hook, *launch_exit_hook;
563:         *args_expand
564:     """
565:     debug_abi = _DebuggerHiddenArgABI.from_metadata(metadata)  # FlagPrism
566: 
567:     args_format = ''.join([format_of(ty) for ty in signature.values()])
568:     format = "iiiKKOOOO" + args_format + debug_abi.parse_format  # FlagPrism
569:     signature = ','.join(map(_serialize_signature, signature.values()))
570:     signature = list(filter(bool, signature.split(',')))
571:     signature = {i: s for i, s in enumerate(signature)}
572:     args_list = ', ' + ', '.join(f"&_arg{i}" for i, ty in signature.items()) if len(signature) > 0 else ''
573:     args_list += debug_abi.parse_argument  # FlagPrism
574:     # Record the end of regular arguments;
575:     # subsequent arguments are architecture-specific descriptors.
576:     arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items() if ty != "constexpr")
577:     internal_args_list = []
578:     for i, ty in signature.items():
579:         if ty[0] == "*":
580:             internal_args_list.append(f"ptr_info{i}.dev_ptr")
581:         elif ty != "constexpr":
582:             internal_args_list.append(f"_arg{i}")
583: 
584:     # generate glue code
585:     newline = '\n  '
586:     ptr_decls = [
587:         f"DevicePtrInfo ptr_info{i} = getPointer(_arg{i}, {i}); if (!ptr_info{i}.valid) return NULL;"
588:         for i, ty in signature.items()
589:         if ty[0] == "*"
590:     ]
593:     # TODO: automatically check if gather load ops are used.
594: 
595:     arch = get_ascend_arch_from_env()
596:     target_support_ffts = is_ffts_supported(arch) and (not force_disable_ffts())
597:     enable_device_print = os.getenv("TRITON_DEVICE_PRINT", 'false').lower() in ('true', '1')
598:     enable_taskqueue = os.getenv("TRITON_ENABLE_TASKQUEUE", 'true').lower() in ('true', '1')
599:     enable_grid_warn_print = os.getenv("TRITON_GRID_WARN_PRINT", 'false').lower() in ('true', '1')
600:     # Per-kernel metadata wins; fall back to the env var when unset.
601:     enable_auto_map_parallel_blocks = getattr(metadata, "enable_auto_blockify", None)
602:     if enable_auto_map_parallel_blocks is None:
603:         enable_auto_map_parallel_blocks = _is_auto_map_parallel_blocks_enabled()
604:     npu_utils = NPUUtils()
605:     num_physical_blocks = npu_utils.get_aivector_core_num() if mix_mode == "aiv" else npu_utils.get_aicore_num()
606:     task_type, mix_block_dim_ratio = _format_of_msprof_task_type_ratio(bs_task_type, mix_mode)
607:     is_mix_task_type = "true" if ("MIX" in task_type) else "false"
608:     LINE_CHANGE_CHAR = chr(10)  # it is \n
609:     alloc_success_code = 'return 1;'
610:     sync_lock_fail_code = 'fprintf(stderr, "Error: syncBlockLock allocation failed\\n"); return;'
611:     workspace_fail_code = 'fprintf(stderr, "Error: workspace allocation failed\\n"); return;'
612: 
613:     cpp_device_pointer = """
614: typedef struct _DevicePtrInfo {
615:   void *dev_ptr;
616:   bool valid;
617: } DevicePtrInfo;
618: 
619: static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
620:   DevicePtrInfo ptr_info;
621:   ptr_info.dev_ptr = 0;
622:   ptr_info.valid = true;
623:   if (PyLong_Check(obj)) {
624:     ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
625:     return ptr_info;
626:   }
627:   if (obj == Py_None) {
628:     // valid nullptr
629:     return ptr_info;
630:   }
631:   PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
632:   if(ptr){
633:     PyObject *empty_tuple = PyTuple_New(0);
634:     PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
635:     Py_DECREF(empty_tuple);
636:     Py_DECREF(ptr);
637:     if (!PyLong_Check(ret)) {
638:       PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
639:       ptr_info.valid = false;
640:       return ptr_info;
641:     }
642:     ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
643:     if(!ptr_info.dev_ptr)
644:       return ptr_info;
645:     aclrtPtrAttributes attributes;
646:     aclError status = aclrtPointerGetAttributes(ptr_info.dev_ptr, &attributes);
647: 
648:     if (status == ACL_SUCCESS) {
649:       if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE && attributes.location.type != 4) {
650:         Py_DECREF(ret);
651:         PyErr_Format(PyExc_ValueError,
652:                      "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
653:         ptr_info.valid = false;
654:         return ptr_info;
655:       }
656:     } else {
657:       Py_DECREF(ret);
658:       PyErr_Format(PyExc_RuntimeError,
659:                    "Failed to query pointer attributes at argument %d. "
660:                    "Error code: %d. This may indicate invalid memory address "
661:                    "or NPU device error.",
662:                    idx, status);
663:       ptr_info.valid = false;
664:       return ptr_info;
665:       }
666:     Py_DECREF(ret);
667:     return ptr_info;
668:   }
669:   PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
670:   ptr_info.valid = false;
671:   return ptr_info;
672: }
673: """
674: 
675:     cpp_msprof_extern = """
676: extern "C" {
677:   typedef int (* callback)(unsigned int type, void* data, unsigned int len);
678:   extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
679:   extern unsigned long int  MsprofSysCycleTime();
680:   extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
681:   static unsigned int __MsprofFlagL0  = 0;
682:   static unsigned int __MsprofFlagL1  = 0;
683: 
```

## triton.backends.ascend.backend_register
来源：/usr/local/python3.11.15/lib/python3.11/site-packages/triton/backends/ascend/backend_register.py；SHA256：5650b0a2fb3dbbde830415396936b00fd93191a324aeeefb6602091b2e19cf6c
```python
228:         ]
229:     return cc_cmd
230: 
231: 
232: @backend_strategy_registry.register("mindspore", "get_current_device")
233: def get_current_device():
234:     import mindspore
235:     return mindspore.get_current_device().device_id
236: 
237: 
238: @backend_strategy_registry.register("torch_npu", "get_current_device")
239: def get_current_device():
240:     import torch
241:     import torch_npu
242:     return torch.npu.current_device()
243: 
244: 
245: @backend_strategy_registry.register("mindspore", "set_current_device")
246: def set_current_device(device_id):
247:     import mindspore
248:     return mindspore.set_device("Ascend", device_id)
249: 
250: 
251: @backend_strategy_registry.register("torch_npu", "set_current_device")
252: def set_current_device(device_id):
253:     import torch
254:     import torch_npu
255:     return torch.npu.set_device(device_id)
256: 
257: 
258: @backend_strategy_registry.register("mindspore", "get_current_stream")
259: def get_current_stream(device):
260:     import mindspore
261:     try:
262:         return mindspore.current_stream().stream_ptr()
263:     except Exception:
264:         return mindspore.current_stream().id
265: 
266: 
267: @backend_strategy_registry.register("torch_npu", "get_current_stream")
268: def get_current_stream(device):
269:     import torch
270:     import torch_npu
271:     if device is None:
272:         device = torch.npu.current_device()
273:     if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
274:         from torch_npu._C import _npu_getCurrentRawStreamNoWait
275:         return _npu_getCurrentRawStreamNoWait(device)
276:     else:
277:         from torch_npu._C import _npu_getCurrentRawStream
278:         return _npu_getCurrentRawStream(device)
279: 
280: 
281: @backend_strategy_registry.register("mindspore", "header_file")
282: def header_file(enable_taskqueue):
283:     return f'''#include "include/utils/device_manager_conf.h"
284: #include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
285: #include "include/mindspore/ops/kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
286: {'#include "include/pynative/utils/runtime/op_executor.h"' if {enable_taskqueue} else ''}
287: {'#include "include/runtime/pipeline/pipeline.h"' if {enable_taskqueue} else ''}'''
288: 
289: 
290: @backend_strategy_registry.register("torch_npu", "header_file")
291: def header_file(enable_taskqueue):
292:     return f'''#include <ATen/ATen.h>
293: #include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
294: {'#include <torch_npu/csrc/framework/OpCommand.h>' if {enable_taskqueue} else ''}'''
295: 
296: 
297: @backend_strategy_registry.register("mindspore", "allocate_memory")
298: def allocate_memory(size, stream):
299:     return f'''auto work_ptr = std::make_shared<mindspore::kernel::pyboost::MemBlock>(device_context, {size}, reinterpret_cast<uint64_t>({stream}));
300:     workspace_addr_ptr = work_ptr->ptr_;'''
301: 
302: 
303: @backend_strategy_registry.register("torch_npu", "allocate_memory")
304: def allocate_memory(size, stream):
305:     return f"workspace_addr_ptr = const_cast<void *>(at::empty({size}, at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte)).storage().data());"
306: 
307: 
308: @backend_strategy_registry.register("mindspore", "allocate_sync_block_lock")
309: def allocate_sync_block_lock(size, stream):
310:     return f'''auto sync_ptr = std::make_shared<mindspore::kernel::pyboost::MemBlock>(device_context, {size}, reinterpret_cast<uint64_t>({stream}));
311:     syncBlockLock_ptr = work_ptr->ptr_;'''
312: 
313: 
314: @backend_strategy_registry.register("torch_npu", "allocate_sync_block_lock")
315: def allocate_sync_block_lock(size, stream):
316:     return f"syncBlockLock_ptr = const_cast<void *>(at_npu::native::allocate_workspace({size}, {stream}).storage().data());"
317: 
335:     return f'''mindspore::runtime::OpExecutor::DispatchLaunchTask({func});'''
336: 
337: 
338: @backend_strategy_registry.register("torch_npu", "async_launch")
339: def async_launch(func):
340:     return f'''at_npu::native::OpCommand cmd;
341:     cmd.Name(name.c_str()).SetCustomHandler({func}).Run();'''
```

## triton.compiler.compiler
来源：/usr/local/python3.11.15/lib/python3.11/site-packages/triton/spec/ascend/compiler/compiler.py；SHA256：31ea8760a43db36ef59d8622d74761a147fbad1cf05e1a760ec341d8a374308b
```python
457:         target = metadata['target']
458:         metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
459:         KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
460:         self.metadata = KernelMetadata(**metadata)
461:         backend = make_backend(self.metadata.target)
462:         self.packed_metadata = backend.pack_metadata(self.metadata)
463:         self.src = src
464:         self.hash = hash
465:         self.name = self.metadata.name
466:         # stores the text of each level of IR that was generated during compilation
467:         asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
468:         binary_ext = backend.binary_ext
469:         binary_extensions = getattr(backend, 'binary_extensions', {binary_ext})
470:         self.asm = AsmDict({
471:             file.suffix[1:]:
472:             file.read_bytes() if file.suffix[1:] in binary_extensions else file.read_text()
473:             for file in asm_files
474:         })
475:         self.metadata_group = metadata_group
476:         self.kernel = self.asm[binary_ext]
477:         # binaries are lazily initialized
478:         # because it involves doing runtime things
479:         # (e.g., checking amount of shared memory on current device)
480:         self.module = None
481:         self.function = None
482:         self._run = None
483: 
484:     def _init_handles(self):
485:         if self.module is not None:
486:             return
487: 
488:         def raise_(err):
489:             # clone the exception object so that the one saved in the closure
490:             # of the partial function below doesn't get assigned a stack trace
491:             # after the subsequent raise. otherwise, the CompiledKernel instance
492:             # saved in the (global) kernel cache will keep references to all the
493:             # locals in the traceback via the exception instance in the closure.
494:             cloned_err = copy.deepcopy(err)
495:             self._run = functools.partial(_raise_error, cloned_err)
496:             raise err
497: 
498:         device = driver.active.get_current_device()
499:         # create launcher
500:         self._run = driver.active.launcher_cls(self.src, self.metadata)
501:         # not enough shared memory to run the kernel
502:         max_shared = max_shared_mem(device)
503:         if self.metadata.shared > max_shared:
504:             raise_(OutOfResources(self.metadata.shared, max_shared, "shared memory"))
505:         if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
506:             # Use blackwell max tmem size for now, this should be moved in device properties
507:             max_tmem_size = 512  # tmem size in number of columns
533:             return ret
534:         arg_dict = {name: arg for name, arg in zip(self.src.fn.arg_names, args)}
535:         ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
536:         return ret
537: 
538:     def __getitem__(self, grid):
539:         self._init_handles()
540: 
541:         def runner(*args, stream=None):
542:             if stream is None:
543:                 device = driver.active.get_current_device()
544:                 stream = driver.active.get_current_stream(device)
545:             launch_metadata = self.launch_metadata(grid, stream, *args)
546:             self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
547:                      knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)
548: 
549:         return runner
```

## NPULauncher
```python
class NPULauncher(object):

    def __init__(self, src, metadata):
        self.compile_only = os.getenv("TRITON_COMPILE_ONLY", 'false').lower() in ('true', '1')
        self.enable_msprof_register_tensor = os.getenv("TRITON_REGISTER_TENSOR_MSPROF",
                                                       'false').lower() in ('true', '1')
        debug_mode = metadata.debug
        header_src = generate_npu_header_src()
        constants = src.constants if hasattr(src, "constants") else dict()
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(key): value for key, value in constants.items()}
        signature = {cst_key(key): value for key, value in src.signature.items()}
        wrapper_src = make_launcher(constants, signature, metadata)
        so_launcher_path = make_npu_launcher_stub(header_src, wrapper_src, metadata.debug)
        self.metadata = metadata  # FlagPrism
        # setup for remote run
        # TODO: use a var to pack all vars required to run on a remote machine
        self.mix_mode = metadata.mix_mode
        self.shared = metadata.shared
        # initialize launcher
        import importlib.util
        spec = importlib.util.spec_from_file_location("__triton_launcher", so_launcher_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")

    def __call__(self, *args, **kwargs):
        if self.compile_only:
            cache_manager = get_cache_manager(args[5]['hash'])
            print("[INFO]: skip running kernel")
            print(f"[INFO]: The compiled kernel cache is in {cache_manager.cache_dir}")
        if self.enable_msprof_register_tensor:
            tensor_params_shape = get_backend_func("get_tensor_params_shape", *args)
            # args[5] must be the packed metadata.
            # Check the launch wrapper in which PyArg_ParseTuple specifies the ordering of args
            args[5]['tensor_params_shape'] = tensor_params_shape
        else:
            if self.compile_only:
                return
            debug_enabled = bool(getattr(self.metadata, "debug_enabled", False))
            if debug_enabled:
                # FlagPrism: use the backend-neutral launch gateway; this
                # backend only supplies its name and launch state.
                with _flagprism.debugger_launch_context(
                        "ascend",
                        self.metadata,
                        args[:3],
                        args[3],
                        args[6],
                        args[9:],
                ) as hidden_args:
                    profiler_registered = self.launch(*args, *hidden_args, **kwargs)
            else:
                profiler_registered = self.launch(*args, **kwargs)
            import triton
            triton.backends.ascend.utils.TRITON_PROFILER_REGISTERED = True if profiler_registered == 1 else False

```
## launch_enter_hook
类型：HookChain；活动回调数：0
```python
class HookChain(Generic[F]):
    """A chain of hooks of the same type F to be called in order.
    """

    def __init__(self, reversed: bool = False):
        self.calls: list[F] = []
        self.reversed = reversed

    def add(self, func: F) -> None:
        if func not in self.calls:
            self.calls.append(func)

    def remove(self, func: F) -> None:
        if func in self.calls:
            self.calls.remove(func)

    def __call__(self, *args, **kwargs):
        for call in self.calls if not self.reversed else reversed(self.calls):
            call(*args, **kwargs)

```
## launch_exit_hook
类型：HookChain；活动回调数：0
```python
class HookChain(Generic[F]):
    """A chain of hooks of the same type F to be called in order.
    """

    def __init__(self, reversed: bool = False):
        self.calls: list[F] = []
        self.reversed = reversed

    def add(self, func: F) -> None:
        if func not in self.calls:
            self.calls.append(func)

    def remove(self, func: F) -> None:
        if func in self.calls:
            self.calls.remove(func)

    def __call__(self, *args, **kwargs):
        for call in self.calls if not self.reversed else reversed(self.calls):
            call(*args, **kwargs)

```
