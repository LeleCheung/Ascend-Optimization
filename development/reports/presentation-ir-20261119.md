# 主图、四项固定输入案例与IR研究

## 23:25完成状态

夜间IR目标3/3：BatchNorm 42.955→39.261μs（下降8.60%），上采样164.856→159.570μs（下降3.21%），scatter_add 1281.927→1262.722μs（下降1.50%）。三项均固定当前最佳Triton源码，通过完整KGS、必要补测、ABBA和修改后双分析，原／改IR与二进制身份分别保存。[案例、源码及证据](ir-performance-20261119/README.md)已落盘。

26项主图与四项固定输入曲线的原生身份、连续和单调检查通过，新scatter全合同／固定输入截图已检查。图面保留最佳进展、方法来源和实际测量桥接，原始下降、复测及失败继续保存。组会入口为[网页与四图](group-meeting-20261119/index.html)、[18页IR进展PPT](group-meeting-20261119/whitebox-simulator-ascend-20261119-ir-progress.pptx)和[讲稿](group-meeting-20261119/talk.md)；PPT逐形状边界通过，桌面PowerPoint原生渲染尚未检查。下文为此前阶段记录。

## 22:18更新：两项IR收益完成验收

BatchNorm与上采样的固定源码IR优化均已通过完整KGS、双方补测、ABBA及修改后双分析，当前2/3。目标耗时分别下降8.60%与3.21%；26项主图、四项固定输入曲线及[17页IR进展PPT](group-meeting-20261119/whitebox-simulator-ascend-20261119-ir-progress.pptx)已同步。上采样全合同的同口径复测轻微下降保留在详情，连接标注“同源码复测见详情”；固定输入复测节点保留真实数值。详细[案例与导出收据](ir-performance-20261119/README.md)已落盘。以下按时间保留首项验收及此前阶段历史。

## 21:05更新：首项IR收益完成验收

BatchNorm固定源码输出IR加宽后39.261332μs、0.717359×CANN，完整27/27、双方各36/36、ABBA及修改后双采集通过，进度1/3。主图已接入IR节点，测量变化用同源码复测灰色连接；26图与四图数据／连续／单调检查通过。详见[案例](ir-performance-20261119/batchnorm-ir004-case.md)和[夜间IR记录](night-ir-campaign-20261119.md)。下文18:50及排队状态为阶段历史。

## 18:50验收状态

26项主图及四项固定输入曲线已验收落盘。主图原生身份、连续连线、数值单调性、合同与测量身份已核对；26图截图检查与四图全尺寸检查通过，页面链接及文字溢出检查通过。四项知识／经验复用分支保留原供体与父版本，在图中使用已确认的成绩进展语义。近似重算节点保留≈标记。

组会材料入口为[group-meeting-20261119/index.html](group-meeting-20261119/index.html)，[15页PPT](group-meeting-20261119/whitebox-simulator-ascend-20261119.pptx)和[讲稿](group-meeting-20261119/talk.md)已保存。PPT结构、内容及图文边界检查通过，桌面PowerPoint原生渲染待检查。74份导出证据完成压缩包及逐文件SHA校验。

BatchNorm最佳源码本次27/27，固定输入48.775024μs、0.578988×CANN；历史44.912918μs、0.626948×冻结保留。scatter最佳本次18/18，固定输入1.281829804ms、2.680344×CANN。原始配对成绩、共同CANN锚点重算及全合同成绩分别保存。

图表阶段完成后已进入当前最佳IR研究。首轮采集的tuple键序列化故障已修复并保留原日志，恢复进程PID3844496已启动。当前IR新增收益0/3，详细状态见[夜间IR记录](night-ir-campaign-20261119.md)。下文复测排队与未启动描述保留为阶段历史。

## 当前执行顺序

1. 完成26项主图与四项固定输入曲线，保存数据、源码身份、连线说明、筛选映射及截图验收。
2. 以完成验收的材料准备下周大组会《白盒模拟器驱动的多层次优化：以昇腾算子优化为例》。
3. 开展模拟器指导的单算子IR研究；先固定机制假设和验证合同，再由CC／DeepSeek修改。

## 固定输入

| 算子 | 输入 | case |
| --- | --- | --- |
| upsample_bilinear2d | FP16 `[4,32,128,128]→[4,32,256,256]` | timing-1 |
| transform_bias_rescale_qkv | FP16 `[8,512,3072]`，16 heads | timing-2 |
| batch_norm_with_update_functional | FP32 NCHW `[8,64,64,64]` | timing-7 |
| scatter_add | FP32 `[2048,2048]` | timing-8 |

主曲线纵轴采用相对CANN的加速比，连续连接保留最佳成绩。原生未通过时采用无数值失败标记，后续有效版本须完整通过固定合同。下降探索保留在详情，筛选映射独立保存。原始性能和交付身份继续冻结。

## 同口径复测

远端独立目录为`/data/hanle/ascend-optimization/runs/presentation-replay-20261119`，专用端口19730，沿用物理卡7共享锁及同一容器、解释器、固定CANN合同和profiler计时参数。控制器为`development/tools/research/presentation_replay.py`。固定源码逐项核对SHA；复测目的为消除展示中的跨计时口径接续，并补齐固定输入的原生测量。

BatchNorm原生及四份历史固定源码已排队完整复测。历史原生205.979μs保留，当前profiler结果单独落盘。新结果绑定请求、合同、源文件及服务状态，避免将计时变化记为代码收益。

首份原生复测已完成27/27：目标81.775047μs，配对CANN28.881026μs、0.353177×。严格Triton起点复测27/27，目标81.436625μs。后续三份优化源码继续排队。原生记录已保存至`presentation-replay-20261119/batchnorm-native/`。

scatter在`presentation-scatter-replay-20261119`中复测原生、两份历史版本、严格Triton起点与当前最佳；两份早期版本另跑现有双方补测及分段独立性检查。直接加载原生模块的相对导入失败已保存，`presentation-scatter-native-adapter-20261119`使用独立package-context加载适配重测，原生字节保持一致。

scatter原生加载修复后已完整18/18通过，目标19.882584ms、配对CANN3.435842ms、0.172807×，全合同0.192623×。展示从该原生实测起步；旧的导入失败原件继续保留。首个历史版本目标19.447351ms，完整18/18、双方各21/21及分段独立性3/3通过。

## 当前恢复信息

- BatchNorm控制器PID3549308；scatter复测PID3560183；scatter原生加载修复PID3583404；历史补测PID3603309。核对`/proc/<PID>/cmdline`与各目录中的process文件后再恢复，避免重复提交。
- 专用服务19730的归属、命令与PID保存于`presentation-replay-20261119/service.json`。设备锁沿用`physical-card-7-campaign.lock`；其他服务保持原状态。
- 导出工具为`development/tools/research/export_presentation_replay.py`；每个完整run导出小型请求、源码、结果、状态与逐文件SHA收据，保留远端原始profiling。
- 导入命令：`python -m development.tools.traces.import_presentation_replay`。它保存更新前展示数据，按共同逐case CANN锚点生成新的全合同主图。
- 四图命令：`python -m development.tools.traces.build_group_meeting`，实际入口为`fixed_case_meeting.py`。数据为`group-meeting-20261119/cases.json`，纵轴单独使用固定case加速比。
- 总览同步：调用`development.tools.traces.progress_view.update_catalog(Path.cwd(), png=True)`，随后执行`python -m development.tools.traces.check_presentation`，并检查四张截图与26图的溢出报告。
- 图表筛选、连线与CANN重算相关8项测试通过。组会讲稿草稿见`group-meeting-20261119/talk.md`；IR协议见`ir-study-protocol-20261119.md`。当前尚未启动新IR候选。

用户后续将历史实现复用理解为知识／方法复用。主图按同一算子的最佳性能进展连续连接，方法来源归知识／经验；具体代码供体和父版本保留在详情，知识库实际读取另据日志确认。相关展示连接均保存原始关系及压缩路径。

## IR研究的选择原则

当前四项有效收益均来自Triton源码修改。IR主案例从四项当前最佳完整合格版本继续，优先BatchNorm；scatter历史串行扫描退步版本单列辅助机制研究。BatchNorm最佳版窗口中搬运和Vector等待占比较高，继续分析实际同步与数据搬运、重复整数运算及均匀计数状态。IR实验选择当前编译结果中具有可观测执行成本的最小区域，固定Triton源码，核对原IR覆盖及修改版完整合同与实际收益。

每轮固定原Triton源码及其编译配置，保存原IR、修改IR、补丁、编译入口、产物哈希、模型修改前决策和正式测量。先验证IR注入接口确实生效，再运行完整正确性、必要专项检查及KGS性能，并重新采集模拟器和profiling验证机制。修改层标记为单算子IR，独立于既有严格Triton交付成绩。通用pass只在多个算子出现同类问题后立项。

CC／DeepSeek编写目标IR补丁；Codex维护通用注入、取证和评测接口。当前阶段准备研究方案，正式IR实验在图表交付落盘后启动。
