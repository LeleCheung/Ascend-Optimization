"""对全部复用CANN资格重新核验原始源码、请求、完整结果和补测身份。仅做文件审查。"""
from pathlib import Path
from common import BATCH,PREVIOUS,read,save,sha
from validation_status import supplement_status


def main():
    old=read(PREVIOUS/'delivery.json');rows=[];files={}
    def checked(path):
        path=Path(path);files[str(path)]=sha(path);return path
    for row in old['rows']:
        op=row['operator'];qualified=bool(row.get('cann_reference_qualified'))
        if not qualified:
            rows.append(dict(operator=op,qualified=False,reason=row['cann_status']));continue
        source=checked(row['reference_source']);assert sha(source)==row['reference_source_sha256']
        formal=checked(row['reference_validation']['result']);result=read(formal)
        assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']>0 and not result.get('is_hack')
        assert all(x['status']=='PASSED' for x in result['per_workload'])
        request=checked(formal.with_name(formal.name.replace('.result.json','.request.json')));payload=read(request)
        assert payload['implementation']['sources'][0]['content']==source.read_text(encoding='utf8')
        assert payload['settings']==dict(warmup_ms=1000,benchmark_ms=100,num_trials=3,timeout_seconds=1800)
        extra=checked(row['reference_supplementary']['path']);supplement=read(extra)
        assert supplement_status(op,supplement,sha(source))=='passed'
        rows.append(dict(operator=op,qualified=True,kind=row['reference_kind'],source=str(source),source_sha256=sha(source),
            formal=str(formal),complete=[result['num_passed'],result['num_workloads']],supplementary=str(extra)))
    frozen=read(BATCH/'software-lock.json')['service_software'];actual=read(BATCH/'environment-recheck.json')
    for port in ['19655','19657','19659']:
        assert actual[port]['software']==frozen
        assert actual[port]['timing']=='walltime' and actual[port]['server_version']=='v6.5.0'
    output=BATCH/'reference-qualification-audit.json'
    save(output,dict(status='verified',qualified=sum(x['qualified'] for x in rows),total=len(rows),rows=rows,
        evidence_sha256=files,software_lock_verified=True,
        scope='复用参考资格重新核验，不重测、不重选高分；单接口和兼容/组合仍区分，5项原参考缺口未放宽'))
    print('REFERENCE_QUALIFICATION_AUDITED',sum(x['qualified'] for x in rows),len(rows))


if __name__=='__main__':main()
