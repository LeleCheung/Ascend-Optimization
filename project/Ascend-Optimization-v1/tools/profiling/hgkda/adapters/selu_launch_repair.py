# 追加到原始第二轮源码之后；只替换入口，原始候选字节保留用于谱系核验。
from triton import knobs as _repair_knobs
from triton.runtime import jit as _repair_jit


@triton.jit
def _selu_values_repaired(values):
    fp = values.to(tl.float32)
    return (1.0507009873554805 * tl.where(
        fp > 0, fp, 1.6732632423543772 * (tl.exp(fp) - 1.0))).to(values.dtype)


@triton.jit
def _selu_dense_repaired(X, N: tl.constexpr, BLOCK: tl.constexpr, LOOP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if LOOP:
        for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
            index = tile * BLOCK + lane
            value = tl.load(X + index, index < N, other=0)
            tl.store(X + index, _selu_values_repaired(value), index < N)
    else:
        index = tl.program_id(0) * BLOCK + lane
        value = tl.load(X + index, index < N, other=0)
        tl.store(X + index, _selu_values_repaired(value), index < N)


@triton.jit
def _selu_strided_tile_repaired(X, index, N: tl.constexpr,
                                SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    remainder = index
    offset = tl.full(index.shape, 0, tl.int64)
    for axis in tl.static_range(len(SHAPE)-1, -1, -1):
        coordinate = remainder % SHAPE[axis]
        remainder = remainder // SHAPE[axis]
        offset += coordinate.to(tl.int64) * STRIDES[axis]
    value = tl.load(X + offset, index < N, other=0)
    tl.store(X + offset, _selu_values_repaired(value), index < N)


@triton.jit
def _selu_strided_repaired(X, N: tl.constexpr, SHAPE: tl.constexpr,
                           STRIDES: tl.constexpr, BLOCK: tl.constexpr, LOOP: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    if LOOP:
        for tile in range(tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)):
            _selu_strided_tile_repaired(X, tile * BLOCK + lane, N, SHAPE, STRIDES)
    else:
        _selu_strided_tile_repaired(X, tl.program_id(0) * BLOCK + lane, N, SHAPE, STRIDES)


def _selu_launch_repaired(kernel, args, meta, grid):
    # 与固定FlagTree的JITFunction.run构造同样的选项和specialization key。
    # 仅复用编译产物；每次使用实际输入参数、当前device和stream，不缓存Tensor。
    if kernel.pre_run_hooks:
        kernel[grid](*args, **meta)
        return
    try:
        options = dict(meta)
        options['debug'] = kernel.debug or _repair_knobs.runtime.debug
        _repair_jit._flagprism.apply_compile_options(options)
        device = _driver.active.get_current_device()
        cache, key_cache, _target, _backend, binder = kernel.device_caches[device]
        bound, specialization, bound_options = binder(*args, **options)
        key = _repair_jit.compute_cache_key(key_cache, specialization, bound_options)
        compiled = cache.get(key)
        if compiled is not None and hasattr(compiled, 'result'):
            compiled = compiled.result()
    except (AttributeError, KeyError, TypeError):
        # 固定内部接口不可用时，在提交计算前回到正常入口。
        kernel[grid](*args, **meta)
        return
    if compiled is None:
        kernel[grid](*args, **meta)
        return
    # 从这里开始不捕获launch异常后重试：原地算子不能在可能已提交后重复执行。
    stream = _driver.active.get_current_stream(device)
    metadata = compiled.launch_metadata(grid, stream, *bound.values())
    compiled.run(grid[0], 1, 1, stream, compiled.function, compiled.packed_metadata,
                 metadata, _repair_knobs.runtime.launch_enter_hook,
                 _repair_knobs.runtime.launch_exit_hook, *bound.values())


def run(A):
    count = A.numel()
    if not count:
        return A
    contiguous = A.is_contiguous()
    block = 8192 if contiguous and count >= 1048576 else 1024
    tiles = (count + block - 1) // block
    grid = (min(tiles, 65535), 1, 1)
    loop = tiles > grid[0]
    with torch.npu.device(A.device):
        if contiguous:
            _selu_launch_repaired(_selu_dense_repaired, (A,),
                {'N': count, 'BLOCK': block, 'LOOP': loop}, grid)
        else:
            _selu_launch_repaired(_selu_strided_repaired, (A,),
                {'N': count, 'SHAPE': tuple(A.shape), 'STRIDES': tuple(A.stride()),
                 'BLOCK': block, 'LOOP': loop}, grid)
    return A


selu_ = run
