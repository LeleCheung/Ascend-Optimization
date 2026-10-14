"""区分优化已结束与 KG 后续审核等待；不修改原生工作区状态。"""
import json


def exportable_state(workspace, allow_review_waiting=False):
    progress = json.loads((workspace / '.kernelgen/run-progress.json').read_text(encoding='utf-8'))
    state = progress.get('state')
    process = json.loads((workspace / '.kernelgen/run-process.json').read_text(encoding='utf-8'))
    if not process.get('finished_at'):
        raise ValueError('KG worker 尚未确认退出，不能归档或导出')
    if state in {'SUCCEEDED', 'FAILED', 'CANCELLED'}:
        return state
    if allow_review_waiting and progress.get('stage') == 'WAITING_CODE_REVIEW':
        receipt = json.loads((workspace / 'stages/optimize/attempts/01/result.json').read_text(encoding='utf-8'))
        if receipt.get('stage') == 'optimize' and receipt.get('result', {}).get('outcome') == 'COMPLETED':
            return 'OPTIMIZATION_COMPLETED_REVIEW_WAITING'
    raise ValueError('KG 优化尚未结束，不能导出最终候选')
