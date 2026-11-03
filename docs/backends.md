# 可选后端与运行边界

## 核心与 KG

HG-KDA 的 compare、diagnose、hotspots、regions、roofline、context 与 knowledge 使用 Python 标准库。它们不导入 KG/KGS，不连接模型或设备。skill 提供工作方法，模型由调用宿主提供。

KG 可读取 context 命令生成的 JSON，继续承担分析/Coder 角色、多轮执行与候选管理。当前版本不修改 KG 内部实现；上下文文件是明确的集成边界。无 KG 的宿主也能调用相同脚本。

## KGS 评测

适配器使用公开 HTTP 协议，保留冻结请求与服务返回的原始结果。服务地址由命令给出，不设个人机器默认值。

    hgkda kgs status --server http://127.0.0.1:19661
    hgkda kgs upload --server http://127.0.0.1:19661 --archive operator.tar
    hgkda kgs evaluate --server http://127.0.0.1:19661 --request request.json --review review.json --run-directory /path/out/new-run

评测适配器当前接受单文件自包含候选。请求沿用 KGS 的 binding、implementation 和 settings；合同包由固定版本的 KGS 客户端打包，上传阶段验证 SHA。默认只发起一次完整评测请求。KGS 的 num_trials 是同次请求内的计时参数，不等于多次独立复验。

提交前执行静态筛查，并核对全部路径/依赖的审查身份。超时或连接中断时保存 operation_id，查询原任务后再决定处理，禁止自动重发。静态筛查与评测结果均不能代替后端资格审查或正式交付准入。

## msprof 采集

collect 命令读取显式 JSON 配置：argv、source、source_sha256、可选 cwd、environment 和 timeout_seconds。argv 为 msprof op 参数数组；输出目录由 --run-directory 指定。采集输出与源码绑定，进程成功不自动等同证据完整。

运行环境负责 CANN 初始化、设备分配和互斥锁。不要用共享容器重启解决项目配置问题。设备采集与评测串行使用分配给项目的设备；运行环境、日志与编译缓存位于仓库外。

## 结果使用

KGS 完整调用计时用于正式性能；独立真机 profiling 和模拟器时间分别保存。CANN/Ascend C 是参考实现类别，Triton 是候选后端约束。PyTorch 参考与 CANN 参考成绩分别报告，不能因切换参考而宣称候选提速。
