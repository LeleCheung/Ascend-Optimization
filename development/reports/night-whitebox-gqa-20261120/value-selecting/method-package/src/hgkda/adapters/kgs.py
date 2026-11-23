"""通过 KGS 公开 HTTP 协议提交冻结请求；不导入其内部模块。"""
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.error
import urllib.request
import uuid

from ..artifacts import emit, identity, read_json
from ..gate import admit


def call(server, endpoint, payload=None, *, timeout=45, method=None, operation_id=None, content_type='application/json'):
    url = urllib.parse.urlsplit(server)
    if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password:
        raise ValueError('KGS 地址必须是无内嵌凭据的 HTTP(S) URL')
    data = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode('utf8') if payload is not None else None
    headers = {'Content-Type': content_type}
    if operation_id:
        headers['X-KernelGen-Operation-Id'] = operation_id
    request = urllib.request.Request(server.rstrip('/') + endpoint, data=data, method=method, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode('utf8', errors='replace')
        raise ValueError('KGS HTTP ' + str(error.code) + ': ' + detail) from error


def upload_bundle(server, archive):
    raw = Path(archive).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    result = call(server, '/operator-bundles/' + digest, raw, method='PUT', timeout=120,
                  content_type='application/vnd.kernelgen.operator-bundle.v1+tar')
    if result.get('bundle_id') != 'sha256:' + digest or result.get('sha256') != digest:
        raise ValueError('服务返回的合同包身份不符')
    return result


def evaluate(server, request_path, review_path, output_directory, *, timeout=1920):
    request = read_json(request_path)
    review = read_json(review_path)
    implementation = request['implementation']
    sources = implementation['sources']
    if len(sources) != 1 or implementation['entrypoint'] != sources[0]['path'] + '::run':
        raise ValueError('本适配器只提交单文件自包含候选；多文件候选需逐依赖审查适配')
    admission = admit(sources[0]['content'], review)
    status = call(server, '/status')
    if status.get('status') != 'ok' or status.get('api_version') != request.get('api_version'):
        raise ValueError('KGS 状态或协议版本与冻结请求不符')
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    op_id = uuid.uuid4().hex
    emit(request, output / 'request.json')
    emit(review, output / 'review.json')
    emit({'operation_id': op_id, 'server': server, 'request': identity(output / 'request.json'),
          'source_sha256': admission['source_sha256'], 'admission': admission, 'state': 'submitting'}, output / 'operation.json')
    try:
        result = call(server, '/evaluate', request, timeout=timeout, operation_id=op_id)
    except Exception as error:
        emit({'operation_id': op_id, 'state': 'unknown', 'error': str(error),
              'next_action': '使用 operation_id 查询服务状态，禁止自动重新提交'}, output / 'transport-error.json')
        raise
    emit(result, output / 'result.json')
    return {'operation_id': op_id, 'source_sha256': admission['source_sha256'],
            'result_path': str(output / 'result.json'), 'evaluation': result,
            'policy': '单次完整 KGS 请求；评测通过不自动改变发布验收统计'}
