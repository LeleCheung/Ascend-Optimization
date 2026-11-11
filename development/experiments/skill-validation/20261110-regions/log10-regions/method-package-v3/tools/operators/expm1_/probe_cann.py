"""独立进程探测官方旧图Expm1精度模式；仅参考侧，不产生性能成绩。"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re




def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['default', 'high_precision'], required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-device', type=int, required=True)
    args = parser.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == str(args.physical_device)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    out = args.output.resolve()
    assert not out.exists(), '同一探测身份不重发'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'started.json').write_text(json.dumps(dict(pid=os.getpid(), checker_sha256=sha(__file__))))
    # GE/TBE要求缓存目录预先存在；每个模式使用独立的本批目录。
    for name, folder in [('ASCEND_CACHE_PATH','vendor-cache'),('ASCEND_WORK_PATH','vendor-work')]:
        path=out/folder
        path.mkdir()
        os.environ[name]=str(path)
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    sdk = args.sdk.resolve()
    header = (sdk/'include/acl/acl_op_compiler.h').read_text()
    block = re.search(r'typedef enum\s*\{([^}]+)\}\s*aclCompileOpt', header, re.S)
    assert block, '无法核对本机官方编译选项枚举'
    entries = [part.strip() for part in re.sub(r'//[^\n]*', '', block[1]).split(',') if part.strip()]
    assert entries[:5] == ['ACL_PRECISION_MODE','ACL_AICORE_NUM','ACL_AUTO_TUNE_MODE',
                           'ACL_OP_SELECT_IMPL_MODE','ACL_OPTYPELIST_FOR_IMPLMODE'], '本机枚举布局与绑定不一致'
    acl = ct.CDLL(str(sdk / 'lib64/libascendcl.so'))
    compiler = ct.CDLL(str(sdk / 'lib64/libacl_op_compiler.so'))
    pointer = ct.c_void_p
    acl.aclCreateTensorDesc.argtypes = [ct.c_int, ct.c_int, ct.POINTER(ct.c_int64), ct.c_int]
    acl.aclCreateTensorDesc.restype = pointer
    acl.aclDestroyTensorDesc.argtypes = [pointer]
    acl.aclDestroyTensorDesc.restype = None
    acl.aclCreateDataBuffer.argtypes = [pointer, ct.c_size_t]
    acl.aclCreateDataBuffer.restype = pointer
    acl.aclDestroyDataBuffer.argtypes = [pointer]
    acl.aclDestroyDataBuffer.restype = ct.c_int
    compiler.aclSetCompileopt.argtypes = [ct.c_int, ct.c_char_p]
    compiler.aclSetCompileopt.restype = ct.c_int
    compiler.aclopCompileAndExecute.argtypes = [ct.c_char_p, ct.c_int, ct.POINTER(pointer), ct.POINTER(pointer),
        ct.c_int, ct.POINTER(pointer), ct.POINTER(pointer), pointer, ct.c_int, ct.c_int, ct.c_char_p, pointer]
    compiler.aclopCompileAndExecute.restype = ct.c_int
    acl.aclGetRecentErrMsg.argtypes = []
    acl.aclGetRecentErrMsg.restype = ct.c_char_p
    options = []
    if args.mode == 'high_precision':
        # 枚举值由本机冻结的acl_op_compiler.h核对；只修改本进程，退出即销毁。
        for number, name, value in [(3, 'ACL_OP_SELECT_IMPL_MODE', 'high_precision'),
                                     (4, 'ACL_OPTYPELIST_FOR_IMPLMODE', 'Expm1')]:
            rc = compiler.aclSetCompileopt(number, value.encode())
            options.append(dict(option=name, value=value, returncode=rc))
    rows = []
    samples = [0., -0., 1.e-8, -1.e-8, 1.e-6, -1.e-6, 1.e-4, -1.e-4,
               .006, -.006, .03, -.03, .125, -.125, 1., -1., float('nan'), float('inf'), -float('inf')]
    cases = [json.loads(line) for line in (args.catalog / 'correctness.jsonl').read_text().splitlines() if line.strip()]
    for dtype, datatype in [(torch.float16, 1), (torch.bfloat16, 27), (torch.float32, 0)]:
        row = dict(dtype=str(dtype), passed=False)
        descriptors, buffers = [], []
        try:
            assert all(o['returncode'] == 0 for o in options), '官方选项设置失败'
            cpu = torch.tensor(samples, dtype=dtype)
            expected = torch.expm1(cpu)
            x = cpu.to('npu:0')
            y = torch.empty_like(x)
            dims = (ct.c_int64 * 1)(x.numel())
            for tensor in [x, y]:
                descriptor = acl.aclCreateTensorDesc(datatype, 1, dims, 2)  # ACL_FORMAT_ND
                buffer = acl.aclCreateDataBuffer(tensor.data_ptr(), tensor.numel() * tensor.element_size())
                assert descriptor and buffer, '官方描述符构造失败'
                descriptors.append(descriptor)
                buffers.append(buffer)
            stream = torch.npu.current_stream(0).npu_stream
            rc = compiler.aclopCompileAndExecute(b'Expm1', 1, (pointer * 1)(descriptors[0]),
                (pointer * 1)(buffers[0]), 1, (pointer * 1)(descriptors[1]), (pointer * 1)(buffers[1]),
                None, 0, 0, None, stream)
            row['execute_returncode'] = rc
            if rc:
                row['vendor_error']=(acl.aclGetRecentErrMsg() or b'').decode(errors='replace')
            assert rc == 0, '官方Expm1执行失败：' + str(rc)
            torch.npu.synchronize()
            actual = y.cpu()
            row.update(input=[str(v) for v in cpu.tolist()], actual=[str(v) for v in actual.tolist()],
                       mathematical=[str(v) for v in expected.tolist()])
            tol = next(c['tolerance'] for c in cases if c['inputs']['spec']['dtype'] == str(dtype).split('.')[-1])
            torch.testing.assert_close(actual, expected, atol=0, rtol=tol['rtol'], equal_nan=True)
            zeros = expected == 0
            assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros])), '零符号不一致'
            row['passed'] = True
        except Exception as error:
            row['error'] = type(error).__name__ + ': ' + str(error)[:2500]
            row['vendor_error']=(acl.aclGetRecentErrMsg() or b'').decode(errors='replace')
        finally:
            # 提交成功后的张量读取已同步；错误路径先同步再释放API描述符。
            try:
                torch.npu.synchronize()
            except Exception:
                pass
            for buffer in buffers:
                acl.aclDestroyDataBuffer(buffer)
            for descriptor in descriptors:
                acl.aclDestroyTensorDesc(descriptor)
        rows.append(row)
    value = dict(status='probe_completed', mode=args.mode, options=options, cases=rows,
                 passed=sum(r['passed'] for r in rows), total=len(rows), qualified_CANN=False,
                 performance_measured=False, candidate_modified=False, checker_sha256=sha(__file__),
                 physical_device=args.physical_device, sdk=str(sdk),
                 catalog=str(args.catalog.resolve()), catalog_sha256=sha(args.catalog/'correctness.jsonl'),
                 compiler_header_sha256=sha(sdk / 'include/acl/acl_op_compiler.h'),
                 scope='独立进程官方Expm1旧图接口与官方high_precision选项局部资格探测；无自写数学、无精度分支篡改、不修改KGS或其他进程。')
    (out / 'result.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: value[k] for k in ['status', 'mode', 'passed', 'total']}))


if __name__ == '__main__':
    main()
