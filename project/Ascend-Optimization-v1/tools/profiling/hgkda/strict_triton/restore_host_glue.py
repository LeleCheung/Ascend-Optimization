"""恢复已审查的QKV C++宿主胶水；不改设备Triton，不重复已有完整请求。"""
from pathlib import Path
from common import BATCH,HERE,PREVIOUS,read,save,sha,load
from gate import inspect,admit


def execute():
    op='transform_bias_rescale_qkv'
    historical=next(r for r in read(PREVIOUS/'delivery.json')['rows'] if r['operator']==op)
    old=historical['selected_new'];origin=Path(old['source']);assert old['backend']=='Triton_with_Cpp_host'
    assert sha(origin)==old['source_sha256']
    index=read(BATCH/'candidate-index.json')
    directory=BATCH/'candidates'/op/'reviewed-cpp-host-v2';directory.mkdir(parents=True,exist_ok=True)
    source=directory/'main.py'
    if source.exists():assert source.read_bytes()==origin.read_bytes()
    else:source.write_bytes(origin.read_bytes())
    binary=origin.parent/'_hgkda_qkv_host_v1.so';cpp=origin.parent/'qkv_host.cpp'
    for path in [binary,cpp]:
        snapshot=directory/path.name
        if snapshot.exists():assert snapshot.read_bytes()==path.read_bytes()
        else:snapshot.write_bytes(path.read_bytes())
    review=dict(index[op],source=str(source),source_sha256=sha(source),origin=str(origin),origin_sha256=sha(origin),
        method='恢复历史合法C++宿主桥；设备算法和Triton核原字节保留',storage_rewrite=False,helper_sha256=None,
        review_note='逐函数检查execute：仅检查元数据、at::empty分配、select视图、获取stream、PyObject_CallObject调用Triton生成启动器；fallback也是已审查原Triton；没有CANN目标核、Memcpy或宿主Tensor计算',
        reviewed_host_bridges=[dict(binary=str(binary),binary_sha256=sha(binary),binary_snapshot=str(directory/binary.name),
            cpp_source=str(directory/cpp.name),cpp_source_sha256=sha(cpp),candidate_sha256=sha(source),status='reviewed_host_only',
            all_functions_reviewed=True,device_computation='Triton_only',
            functions=['execute','PYBIND11_MODULE'],note='源码与历史构建二进制身份固定；ATen分配和视图允许，设备核来自_qkv_tile')])
    save(directory/'review.json',review);save(directory/'static-audit.json',inspect(source));admit(source,directory/'review.json')
    index[op]=review;save(BATCH/'candidate-index.json',index)
    run=load('strict_host_restore_runner',HERE/'run.py')
    class RestoreRunner(run.Runner):
        def job(self,key,fn):return super().job(key+'@cpp-host-v2',fn)
        def formal(self,operator,src,leaf,out,label):
            result=old['pytorch' if label=='strict-PyTorch' else 'cann']
            assert result['passed']
            request=read(Path(result['result']).with_name(Path(result['result']).name.replace('.result.json','.request.json')))
            assert request['implementation']['sources'][0]['content']==src.read_text(encoding='utf8')
            assert request['settings']==dict(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1800)
            return dict(status='PASSED',score=result['score'],passed=result['complete'][0],total=result['complete'][1],
                is_hack=False,path=result['result'],source_sha256=sha(src),measurement='reused_exact_source_and_contract')
        def supplement(self,operator,src,leaf,out,label):
            from validation_status import supplement_status
            previous=old['supplementary' if label=='extra-pytorch' else 'cann_supplementary']
            value=read(previous['path']);assert supplement_status(operator,value,sha(src))=='passed'
            return dict(status='passed',passed=value['passed'],total=value['total'],path=previous['path'],
                source_sha256=sha(src),measurement='reused_exact_source_and_contract')
    runner=RestoreRunner();previous=runner.state['operators'].get(op,{})
    archive=BATCH/'revisions'/op/'restored-v1-state.json'
    if not archive.exists():save(archive,previous)
    if previous.get('source_sha256')!=sha(source):runner.state['operators'][op]={};runner.persist()
    runner.one(op);runner.state.update(status='completed',current_job=None);runner.persist()
    load('strict_host_restored_report',HERE/'report.py').main()
