# RX 9070 / Windows 原生 HIP 适配

开发基线：`1314521gjy/ninfer-fusion-kvmem@4e93bb8ab89b583653a9d16235582c688226ff05`。
主引擎位于 `src-tree/fusion-engine-src/`。保留其 KVMem 环、host-backed 复用和融合算子设计。

长期追踪检查点见 [upstream-tracking.json](upstream-tracking.json)。上游源码已审阅至 `9a5a920`，
原生采用基线仍为上面的 `4e93bb8`；新增 KV 回收改动须闭合接口和页引用所有权后再验收。
现有本地每日任务负责后续检查，该 JSON 记录本次公开检查点。

当前是适配开发分支。实际 public `.ninfer` Engine 与五个原产品已完成 AMD 构建，
Bonsai BF16 KV 的非推测和 MTP2 首次短推理均通过。两次均输出 `AMD native.`，
生成 token ID 相同。独立观测消费者随后也验证了两种模式的实际 HIP Graph 捕获、实例化
和推理阶段回放，`generation.wait()` 完成且仍生成相同四个 token；尚无端到端性能优势结论。
Dot 提供的原生 gfx1201 BF16 MMA 已接入原有生产 BF16 GEMM 并验证；
其他矩阵片段、四套 GDN、权重恢复/Graph、十一套完整 public Linear 原测试已通过。
完整 Attention 首次测试暴露 NVFP4 解码和 LDS 容量问题；局部修复后原 NVFP4 专项及
默认完整 Attention 均通过，其他独立注册模式仍须分别验证。
旧 `ninfer-all` HIP 实验的核心测试不能视作本基线的测试结果。

原始 Ops 测试已逐项接回：160 项注册中，159 项有原生目标/参数/环境映射；
另 1 项 NVIDIA TMA 专用 staging race 当前不受支持。现有 111 项原始注册实际通过，
另 4 个 Attention 入口可由相同函数和参数的默认测试证明覆盖，不能记作独立执行通过。
SparseMoE 与 GDN replay-fold 曾在 Graph 用例失败：默认流 D2D 初始化复制缺少完成依赖，
非阻塞 Graph 流可能提前读取或覆盖状态。两项夹具各增加一次检查返回值的完成屏障后，
完整原始注册均已实际通过；原算子、形状、容差、断言和回放次数不变，旧失败日志保留。
SparseMoE wide 及卷积 snapshot/record 的 legacy/unified 四个注册也已分别实际通过。
Attention/GDN 输入投影 small-T 的 legacy/unified 四个注册也已完整独立执行通过。
剩余 44 项尚未运行，
本轮因独占窗口结束而未启动，不计为失败或通过。
PackGQA、GDN FP16 state two-stage、E8 全码对及两套投影 small-T 原测试均已实际通过。
路线表变体仍分别记录，默认通过不会外推为所有候选或 `legacy`/`unified` 均通过。
完整词表转码要求至少 5.5 GiB 可用 RAM，目前因准入不足未运行；没有缩小词表或删除 Q4/Q6 测试域。

普通原产品冷进程测试中，默认 A8 许可配置的非推测/MTP2 均生成相同四个 token；
原 HTTP 产品的 A16 参考与默认 A8 冷请求均完成资源准入，实际日志分别确认
`prefill_a8=false/true`、19 token 完整输入与零前缀复用。HTTP 不导出生成 token ID，
因此该接口仅核验文本与 usage，不声称观察到 ID 相等。
独立 Graph 内核观测消费者又在 A16 非推测、默认 A8 非推测/MTP2 中完成了实际准入：
生成回放关联的真实 HIP 名称分别包含 `t2_small_t_v2` 与 `t2_small_t_i8`/量化内核。
这仅证明所捕获的 DecodeGraph 路线；未捕获的 prefill、耗时热点和完整质量均不能由此推断。
此前约 5 token/s 的短诊断显式关闭了 A8，不能代表默认整数路线的性能；默认 A8 激活精度
须另做质量验收。当前尚未证明相对 llama.cpp 的显著、可重复端到端优势。

显式 A8 评分工具现已实际构建：固定4目标与独立五组38目标均完成 A16/A8 两臂，
共12个实际进程通过资源检查；五组 logprob 差异 MAE 为0.00102823、最大0.018383。
这是固定参考续写的激活精度证据，不能替代任务、长文、生成稳定性或速度验收。
工具默认仍为 A16；`--prefill-a8` 才允许整数激活，`--resource-hold-ms` 默认0。
资源观测使用的显式8000ms驻留发生在评分后、真实 Engine 仍存活时，不计作性能样本。

非连续 KV 借用已有默认关闭的普通 HIP 显存替代后端：单一 payload、稳定设备页表、
有界 D2D 搬移和事件完成后归还。三个仓库 CMake 目标现均实际构建并运行通过，覆盖
21698项规划检查、字节/同实例 Graph，以及四个 NotReady 事件边界且无超时后备。
范围与复现方式见 [实验 KV 测试](memory/tests/README.md)。原 VMM 失败记录和生产门
保留。原分配器绑定与页表感知 BF16 append/Attention 候选现也已实际构建并通过六组功能测试：
18 次同实例 Graph 回放、1,259,520 项输出检查，独立 FP64、完整 KV 字节、跨 K/V 物理
搬移后位相等、每次回放的新写入和借用区隔离均通过。两候选开关默认关闭，尚未接入
生产 Engine；独立原分配器夹具已构建并通过链接图检查，尚未执行。范围见
[BF16 借页消费者](memory/segmented_bf16_attention_source.md)，尚无整模型显存节省或提速结论。

上游优化持续单独记录和审阅，当前已审阅至 `9a5a920`；新增 KV 回收补丁的接口、
页引用所有权与完整流同步仍待适配验收。实现基线仍为上述 `4e93bb8`，不会把上游 NVIDIA 验收数字
直接当作 AMD 结果。本地 llama.cpp/KVMem 优化按真实源码与质量/资源合同逐项迁入。

CLI 与 bench 的 `--lookup-ngram 0` 现可显式关闭上下文 lookup，符合原帮助与默认值语义。
两套原始 CPU 合约测试均实际通过，覆盖先设 5 再设 0、保留独立 MTP 配置，以及拒绝负数
和非整数；实际产品已重新链接。此修复不改变算子数学或推测解码算法。

候选选择器原完整测试曾只在 workspace scope/peak 断言失败：末尾 `I32[B]` 的实际高水位
与容量函数额外舍入到 256 字节不一致。适配层现返回 `peak_bytes(1)`，保留每个缓冲区的
256 字节起点对齐、全部数学/保护区/Graph 断言；修复后完整原测试实际通过。

## 已验证的硬件结果

2026-10-20，Windows 11 / RX 9070 gfx1201 wave32，显卡驱动 32.0.31041.1004，
本机 ROCm SDK 目录版本 10.0.0、AMD clang 23、运行时 API 版本 70260201、MSVC 14.44。

- 2,098 个矩阵样例，其中 2,050 个要求精确相等；另 48 个与独立 FP64 数学参考比较。
- 32 / 128 线程，每组连续累加 1 / 3 次，四组全部通过，共检查 1,074,176 个输出。
- 最大误差 / 允许误差界为 0.0898404337953；最大逐矩阵相对 L2 为 1.30929792175e-7。
- 输入未改写、输出保护区完好；同时拥有的两个设备缓冲区共 3,776,400 字节。
- CPU 独立布局及整数乘积检查通过 386 个样例。本机此次未重新连接 AMD 官方布局计算器。

这些结果只覆盖有限正常或零 BF16 输入的 MMA 原语；不代表生产 GEMM、attention、
KVMem、完整引擎、吞吐或端到端峰值显存已验收。原始候选 README 说明了精确边界。

## 构建与运行

本机执行 `ports\amd\build.cmd`，使用显式的 AMD 编译器与 MSVC 14.44 环境。
其他安装路径请修改脚本中的工具链位置。CMake 入口是本目录，独立于原有 CUDA 构建。
目标 `ninfer_amd_mma` 只含独立测试；CTest 超时 30 秒，77 是未运行。

加 `-DNINFER_AMD_CORE=ON` 编译本基线的真实 Core 和 14 个原有行为测试。
源代码由 `prepare_core.py` 从当前 CUDA 树和本机 HIPIFY 表生成到构建目录；
Graph API、VMM 字节地址、回调 ABI、Windows ETW 的语义修正明确写在脚本或 `core/` 中。
HIPIFY 原许可保留在生成目录。生成阶段不自动翻译 PTX；生产算子所需的指令适配由本目录显式实现。
其他机器需用 `-DNINFER_PYTHON=<Python 3.11 路径>` 显式选择 Python 环境。

加 `-DNINFER_AMD_BASIC_OPS=ON` 构建真实基础算子、其 GGUF 依赖和 30 个上游数值测试。
它需要同时启用 `NINFER_AMD_CORE`。这不是完整的 Engine 构建目标。

`NINFER_AMD_ATTENTION=ON` 接入全部原生产注意力/存储格式。context、sliding、packed、
KV append/materialize/stream 的独立测试已实际编译链接。packed、sliding、context、
KV stream、完整 KV append、BF16 KV materialize 及 selected-block 的下述数学测试通过。
完整 causal 库及原完整测试程序已实际 O3 编译链接；各独立模式仍须分别完成数值验收。
`NINFER_AMD_LINEAR=ON` 接入原 public Linear 分派与格式家族。140 个生产编译单元、共用依赖、
十一套原数学可执行文件及有界探针已实际 O3 编译链接。
两者都需要 Core 和 Basic Ops；不能借 HIP 的 major/minor 号选择 NVIDIA 调度表。

`NINFER_AMD_ARTIFACT=ON` 构建真实 v3 reader/binder/materializer；该库及三个原测试已编译链接，
读取/绑定 CPU、GPU materialize/permute 和 Python writer 单文件/分片互通测试均已通过。
`NINFER_AMD_RUNTIME_SUPPORT=ON` 接入原 Engine 资源、上下文成本和 profile 支持库；
它们是完整 Engine 的依赖，不是可运行模型的替代入口。

`NINFER_AMD_FUSED=ON` 接入原融合投影家族；`NINFER_AMD_AUXILIARY_OPS=ON` 接入
候选选择、分组卷积、KVMem 和稀疏 MoE。`NINFER_AMD_ENGINE=ON` 的目标图读取完整原有
public Engine、文本、媒体、模型 loading/runtime 和原测试清单，依赖上述真实生产库。
这些实际生产库、完整 Engine 与五个产品均已编译链接；整模型质量和性能仍待验收。

197 个原融合生产编译单元及八个对应静态库已实际完成 O3 编译与归档。
这轮构建曾在原 TopK 测试的 Graph API 签名适配处停止；类型安全适配修复后，Core Graph
原测试及 TopK 完整数值/Graph 套件已通过，Pair 分组边界探针也通过。
首批 28 个融合/辅助原测试程序现已全部链接；原 Q4/Q5/Q8 A16 及 NVFP4 LinearAdd、
Q5 small-T、Q8 LinearSwiGLU/LinearPair、GDN/Attention input 默认套件与部分辅助原测试已通过，独立 KVMem 探针也通过；
其余注册继续按参数和环境分别验收。combined GGUF 原测试补链接实际 SDK compiler-rt 后，
已完成真实链接与全套数学验证，原 oracle 未改。
原有超过 64 KiB LDS 的四处统一调度已通过实际编译，容量断言与数学 oracle 保留。
`NINFER_AMD_PRODUCTS=ON` 现可接入原 CLI、serve、calibrate、perplexity 和推理 benchmark；
其开关默认关闭，必须同时启用完整 Engine。五个真实产品均已构建，原 CLI 已完成首次模型推理。

原 Engine、模板和产品的 39 项 CPU 合约已全部通过；额外七项 AMD 环境/profile/help
合约也通过。CPU 专家保留原计算体，以函数局部 ISA 与 CPUID/OS 状态检查选择标量、
AVX2 和 AVXVNNI。本机三条已准入路径均通过独立整数编码/行计算与稀疏 FP64 forward
验证；该验证已作为 host-only CTest 接入，不给整个二进制强加 AVX ISA。

完整原 Ops 清单另有 87 个程序、160 项注册。覆盖审查补接了之前遗漏的两项 SmallTMma、
E8 root decode、词表转码和 CPU route coverage 原程序，以及 62 项此前只有外部队列或尚未
注册的原运行模式；参数、环境和时限保留。这些补接已配置，尚未完成新程序构建与逐模式
资格验证。NVIDIA TMA 专用 staging-race 测试单独标记为本卡原生机制不适用；不能把别的
AMD 路由运行成功说成该 TMA 机制通过。

可选 `NINFER_AMD_ALIGNMENT_TOOLS=ON`（同时启用 Engine 和 Products）接入两个研究工具。
`ninfer_amd_frontend_export` 复用真实生产 frontend，导出完整渲染文本与实际准备 token；
已构建并通过11项实际前端夹具，包括中英文、Unicode/控制字节、字面控制串、多轮、
thinking模式、stop去重及工具/媒体/词表补齐行的拒绝。
`ninfer_amd_score_tokens` 通过公开 CausalScoring API 导出指定 ID 序列的逐位置自然 logprob；
已构建并通过17项实际help/解析/默认计划检查，默认不构造Engine，显式 `--run` 才加载模型。
共28项检查串行持锁通过，最低可用RAM4.151GiB；回执在工作区
`research/ninfer-frontend-fixtures-native-first/receipt.json`。随后实际评分了完整23个输入IDs
中位置19–22的四个指定目标（含EOS）：BF16 KV/context512、FP32 state、A16、profile off，
exit0且资源检查通过，host guard最低3.382GiB。结果保存在
`research/ninfer-native-score-first/`；这只证明该小型评分路径实际执行，尚无配对品质或速度结论。
这两个工具用于相同输入和公共评分词域的对齐，不提供端到端优势结论。

原 artifact reader、admission、context cost、resource manager 和 prefix-cache index 的
五套 CPU 测试均已在实际 CMake/O3 构建中通过。修正了原 resource-manager 测试夹具缺失的
压力规划能力声明，保留其全部行为断言；这不代表完整 Engine 推理已通过。

## 生产算子本机资格验证

2026-10-20 首轮生产套件共 48 项：46 项通过、KV 借用隔离失败、完整推测解码测试触及
60 秒时间限制。30 个基础算子测试中 29 个通过，覆盖归一化、embedding、卷积、采样、
MTP round、位置、scatter、激活和输出 logprob。所有原有数学 oracle 和误差判据保留。
推测解码有大量 CPU oracle 工作，需进一步定位或按原有专项模式拆分验证；超时没有算作通过。

后续严格 oracle 复测中，原先因等锁未运行的五个基础套件均已通过。完整推测解码套件以
180 秒子进程上限重跑通过，保留全部 `K=1..31`、混合/重复历史、batch 和非有限值检查；
此前 60 秒时限不足。累计 30 个基础套件的原判据均已有通过结果。

十套完整原量化 Linear 数学套件通过：Q4/Q5/Q6/Q8 A16、T2 A16/A8、FP8 A16/A8、
NVFP4 A16/A4。保留全形状、原 FP64 oracle、输入保持、保护区、Graph 及策略分支。
七类 public Linear 有界探针也全部通过。Q6 探针初次误用了不合法的 token 数，已按原
四 token 对齐合约修正夹具，未改变生产准入或原数学测试。

完整原套件的 FP64 oracle 只物化 32 个采样行，输出也分块扫描，不持有整张 FP32 权重。
逐套按实际压缩权重/副本预算准入，额外留 1 GiB 运行时与 2 GiB 系统余量；本轮十套均实际
运行并通过，观测最低可用 RAM 2.902 GiB。BF16 原词表夹具的完整保护副本原先偏大，现已把
生成测试 helper 的 readback 改为至多 2 MiB 的全字节分块比较；全域和判据保留，完整 GPU
复测也已通过，包括 248,320×5,120 词表原域与 FP64/Graph 检查，最低可用 RAM 3.417 GiB。
至此十一套 public Linear 完整原数学套件均已有通过结果。

融合 RMSNormRoPE 的原完整测试也已通过，包括与分拆路径的逐位比较和原数学 oracle。
初测 19 处差异全部在 RoPE 后半通道；真实编译 IR 显示两条路径选择了不同的 FMA 收缩顺序。
融合路径现在显式保持该舍入边界，原 exact 断言未改。

制品的原 GPU 重排、materialization 和 Python writer 单文件/分片互通均通过。
writer 互通夹具补齐了当前 v3 格式要求的 identity 字段；全部父对象字节与逻辑视图断言保留。
原 device-profile 测试也已通过。Windows 没有 GNU `--wrap`，原 Windows 路径不执行
materializer 的链接注入故障测试，不能据此宣称该注入分支已验证。

生产 `bf16_gemm_mma_kernel` 已通过独立 FP64 参考检查：

- `N=256,K=5120` 的 13 个 token 长度，覆盖完整输出、非整 tile 尾部、Serial/Plain 与
  PingPong/Xor64 两种流水线及 Graph 捕获执行。
- `N=5120,K=6144` 的 `T=1,17,32`，数值参考检查均匀分布的 16 个输出行，
  同时扫描全部输出为有限值，并验证保护区及输入/权重不被改写。
- 采用上游 A16 Linear 判据，最终 BF16 存储之前计算 FP64 oracle。
  这是生产 kernel 验证，尚不能声称完整 Linear Op 或模型推理已经支持。

LDS 装载与零填充共 12,800 个精确检查通过；OCP FP8 E4M3/E5M2 的 512 个编码解码通过，
包含负零、非正规数、NaN/Inf；有限 FP32→BF16 的 326,400 个舍入边界样例通过。
HIPIFY 默认把 NVIDIA FP8 映射成 FNUZ，本适配显式改用 OCP 编码，保持原制品字节语义。

GGUF 的既有 GGML GPU 桥已使用其原生 HIP/RDNA4 WMMA 路径编译，未切换模型运行引擎。
恢复的 `vendors/hip.h` 来自同一固定上游提交 `a25c9865fe03c954c93fd755b5d79ae86ba99750`，
SHA256 为 `db9b1142727f3eaf037d888da133029cd21d9fbe15cfc707be7b9633ae080323`。
这消除了 embedding 的真实依赖缺口。GGUF Linear/LinearAdd/LinearSwiGLU 的原完整组合
数值套件现已通过；它不代表所有外部 GGUF 模型都满足本仓的架构和控制权重格式合约。

### Attention 与精确缓存编码

原 packed、sliding-window、context 和 KV-stream 独立套件通过，原 FP64 判据保留。
新增 selected-block public Op 测试覆盖 BF16/OCP FP8、非顺序稀疏选择、空集合、尾部、
Graph、保护区及输入保持，共 16 个样例，通过独立 FP64 oracle。

完整原 KV append 套件现已通过，包括 RK4V4/RK2V4 E8 的逐代码比较。初测 RK2V4 有
四个精确代码失败；原因是本机 HIP 的普通 `_rn` 包装允许乘减融合及近似平方根。
适配只在 E8 codec 上保留显式舍入边界：可链接的 OCML rounded 加减乘、禁止局部收缩的
除法，以及编译器正确舍入的除法/平方根模式。实际 O3 链接和全部原断言复测通过。
没有使用 SDK 头文件中声明但本机设备库缺失的 rounded sqrt/div 符号。

完整 causal attention 的数学资格仍待完成，以上不是端到端性能证据。
KV materialize 首次完整运行在动态共享内存设置处失败：原 Fused64 tile 需要 66,560 字节，
超过本卡 65,536 字节上限。原生 overlay 已把该路由的 K tile 从 128 改为 64，需求降为
33,280 字节，并增加编译期容量断言；全部 192 个 eager case 和 29 个 Graph 原断言保留，
该修正已实际 O3 重编与全套复测通过，另保留全部 87 次 Graph 重放，最低可用 RAM
6.271 GiB。合法的 49,664 字节路由继续使用真实 HIP 属性 API。该套件验证 BF16 K/V，
不能据此宣称其他缓存 codec 的 materialize 数学已验证。

原默认完整 Attention（causal、plain/packed、context）现已实际通过，保留原始 FP64
oracle、缓存字节、映射、边界和保护检查。首次 NVFP4 异常来自当前 SDK 的 FP4→half
软件分支：负零位码被数值转换成 32768。适配仅将实际 KV 的两处 FP16 扩展改走正确的
FP32 解码再精确转换为 FP16；SDK 文件和持久缓存表示均不改变。独立实际 GPU 测试覆盖
全部 256 个打包字节、127 个合法尺度，1,040,384 个 FP16/BF16 结果逐位通过，含负零。

另一失败点是大 RowTiles 的量化 small-T 调度，静态与动态 LDS 合计超过 64 KiB。
适配缩小其 key contraction，并保持 P 的原 swizzle stride；编译期检查计入所有固定数组
与动态 arena。336 个实际编译内核的资源元数据核对通过：FP8/NVFP4/K8V4 最大总 LDS
分别为 59,840/52,160/56,192 字节。修复后 NVFP4 原专项及默认完整套件均 exit0，
完整套件最低可用 RAM 3.566 GiB。DFlash2、PackGQA、wide 等独立注册模式尚未据此全部验收。

Flash-Next 原生产 NVFP4 MoE gate/up 的三种调度已限制到真实 64 KiB LDS 预算；九次真实
生产 phase 调用、124,800 个 BF16 输出通过独立 packed-weight FP64 oracle，含 expert/token
排列、尾部、保护区与输入保持。真实 HIP batch DMA 也通过 pageable/pinned 来源、两次更新和
逐字节保护检查。它们不代表完整 MoE、host-tiered staging 或整个模型已验收。

### GDN 的原生 FP32 路径

RDNA4 的 WMMA 没有 TF32 输入类型（见 [AMD 精度支持表](https://rocm.docs.amd.com/en/develop/reference/precision-support.html)
及 [RDNA4 WMMA 指令](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html)）。
GDN 原有 `m16n8k8` 片段接口现用 wave32 shuffle 和原生 FP32 FMA 计算完整 FP32 乘加，
保留 Op 的显式状态/存储转换；没有 BF16 降精度、TF32 位级等价或 Tensor Core 吞吐的承诺。
片段坐标依据 [NVIDIA PTX 的 m16n8k8 定义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688)。

该生产适配已用 1,088 个矩阵验证：1,024 个精确基底/累加样例、64 个保留全部 FP32 尾数位的
有限正常数样例，独立行列布局 FP64 oracle。32/128 线程、1/3 次累加、float/bits 两种调用共
8 组通过，检查 1,114,112 个输出；最大相对 L2 为 5.9051e-8，输入和保护区完好。
这里没有覆盖 NaN/Inf/非正规 FP32 输入，也不代表整个模型已验收。

加 `-DNINFER_AMD_GDN=ON`（同时启用 Core/Basic Ops）可编译原有 GDN 的完整家族，
包括 recurrent、WY/state-passing/output、two-stage、状态转换与 replay-record。
4 个原有数值测试已通过，包括 WY/recurrent、two-stage、FP16 状态和 replay-record。
保留了原始 FP64 oracle、长序列、状态连续性和误差判据；其中 two-stage 覆盖 T=16,385。
`test_gdn_replay_fold` 所依赖的融合输入投影库现已编译，原完整 replay-fold
测试已接入目标图并实际通过；独立 wide 注册仍待单独执行。
仅在 GDN 的 `FoldAccess` 上将 NVIDIA `__grid_constant__` 优化限定符改为 HIP const 值参数；
其方法不跨线程发布参数地址。其他 TMA 描述符未作同样转换。

### 其他矩阵片段

同一 RX 9070 / Windows / clang 23 环境中，以下生产片段适配已通过共享 GPU 锁下的独立测试：

| 输入与累加 | AMD 实现 | 数学证据 |
|---|---|---|
| FP16 → FP32 / FP16 | 原生 wave32 WMMA，保持两种不同的累加转换边界 | 2,180 个矩阵、8 组配置、2,232,320 个输出；基底、非正规数、正负零、溢出边界与 FP64 dense oracle |
| signed INT8 → INT32 | 两次原生 INT8 WMMA；关闭饱和以保留原接口溢出回绕 | 4,176 个矩阵、4 组配置、2,138,112 个精确输出；全坐标基底、随机有符号输入与溢出 |
| OCP FP8 E4M3 → FP32 | 两次原生 FP8 WMMA；保持 OCP 编码 | 4,706 个矩阵、4 组配置、2,409,472 个输出；全部 64,516 个有限编码对、FP32 非正规累加值与 FP64 oracle |
| NVFP4 E2M1 + UE4M3 → FP32 | 当前寄存器片段精确展开为 FP16，再用四次原生 WMMA | 8,420 个矩阵、4 组配置、4,311,040 个输出；全部 2,032 个合法码×scale 精确解码、16,129 个 scale 对、忽略元数据 lane 的毒化检查 |

以上均检查输入未修改和输出保护区。FP16 测试曾发现本机 clang 把半精度向量分量的
`bit_cast` 始终读取为第 0 项；现改为整体向量位转换后索引，原判据复测通过。
NVFP4 解码还发现本机编译器的融合半精度转换丢失负零符号；现显式保持结果符号位，
全部精确检查通过，原 oracle 未改。该实现没有全局转换缓冲、运行时权重重打包或原生 FP4
矩阵硬件的声明。这些片段结果也不能替代 public Linear、Attention 或完整 Engine 验收。

| NVIDIA 机制 | 当前 AMD 实现与边界 |
|---|---|
| BF16 Tensor Core MMA | gfx1201 原生 WMMA，已验证片段与上述生产 GEMM kernel |
| GDN TF32 MMA | 原生 FP32 SIMD FMA + wave32 shuffle；片段及上述四个原 GDN 数学套件通过 |
| FP16 / signed INT8 / OCP FP8 E4M3 MMA | gfx1201 原生 WMMA；上述片段测试通过，生产 Op 家族另行验收 |
| ldmatrix x2/x4、转置 | 原生 LDS 装载及 wave32 shuffle，精确布局测试通过 |
| CUB 排序 | hipCUB，采样与完整原推测解码长套件均已通过 |
| cp.async / pipeline copy | 先用逐线程完成的 HIP 全局→LDS 复制保持依赖；尚无异步重叠性能结论 |
| PDL 网格依赖 | HIP stream 顺序执行，保留数据依赖与 Graph 捕获语义，暂不实现网格重叠 |
| CUDA Graph | HIP Graph，Core、归一化和生产 BF16 GEMM 路径已验证 |
| NVTX | 原生 Windows ETW 范围事件；尚未验收 GPU 时间线关联 |
| 稀疏 VMM 借用 | 独立诊断出现数据隔离失败、停滞或运行时崩溃，不能作为已支持能力 |
| NVFP4 block-scale MMA | 寄存器解码加四次原生 FP16 WMMA，片段数学通过；gfx1201 无原生 FP4 WMMA |
| TMA 与 producer-only named barrier | 原生 LDS 装载和计数/代际屏障；独立屏障测试通过，完整 Op 数学另行验收 |

上述测试不是吞吐或峰值显存基准；该轮有另一会话的模型测试重叠，耗时不作为性能比较。

## 共享 GPU 测试所有权

全部 CTest GPU 入口通过 `test_locked.py` 获得共享锁的 byte 0，再启动测试。当前协作主机
显式配置 `-DNINFER_GPU_LOCK_FILE=D:/programfile/work/deepseek/cache/gpu-benchmark.lock`，
与本地模型部署会话共用。其他机器用该选项指定各项目共同的锁路径。
默认等候 300 秒；本协作主机为新队列配置 1,800 秒。锁忙且到期返回 77，表示未运行，不是通过。
测试完成释放锁后留 1.5 秒间隔，让另一会话能交替获取。

子测试位于 Windows kill-on-close job 中，超时或外层取消时只结束本次测试及其子进程，
先清理再释放锁，避免遗留 GPU 进程。已通过纯 CPU 的互斥、stdout 保留、超时、取消清理及释放检查。

## Core 与 VMM 当前状态

Core 静态库已经用原生 HIP 完整编译。首轮 14 个原有行为测试中 13 个通过，
包括 Graph、arena、多 rank、VMM/Graph 重映射、权重池、KV pool、磁盘桥和状态存储；
KV 借用隔离测试失败，保留了原断言。

后续独立 VMM 诊断暴露更深问题：64 KiB 映射在借出非连续页之后运行访问常驻页的 GPU 内核会
超时，预加载内核也未解决；改用 2 MiB 映射后出现 `hipErrorLaunchFailure`。
权重池测试也出现过 `hipMemSetAccess` 的 `hipErrorInvalidValue`，因此不能把首轮通过当作稳定支持。
诊断轮共 18 项，13 项通过、5 项失败（含 3 项新增诊断）；完整结果没有计作通过。

这些诊断通过 `-DNINFER_AMD_VMM_DIAGNOSTICS=ON` 显式启用，每项超时 30 秒。
默认关闭单独的驱动诊断；Core 模式仍会运行原有的失败用例，不会把它们过滤成通过。
另有一条失败与夹具对 NVIDIA 大页的假设有关：64 KiB 粒度下原来的 scale plane 也可借出，
原测试仍将其视作不可借出。现已把夹具的 scale 页调整为 128 字节，保留原来“不可借出”的测试意图；
原有常驻数据隔离断言未放宽；共享锁保护下单项复测仍有 4 条隔离断言失败。

AMD 上游也记录过尺寸相关的
[hipMemSetAccess 缺陷](https://github.com/ROCm/rocm-systems/issues/2516)，但尚未证实本机全部故障
与其同根因。后续优先确认可用的内存映射路径，必要时为借用机制提供显式的复制方案；
不能通过放宽隔离断言或忽略错误继续推理。

### 已验证的原生权重借用

权重池现用单次 `hipMalloc` 分配直接借出实际尾部，绕开故障 VMM alias/remap 路径。
借出和恢复不额外分配显存，固定权重地址保持不变；不同借出大小返回各自的当前尾部地址。
保留原来的 pinned-host 镜像，只恢复本次被借出的真实权重字节，其他窗口内修改保持原样。
原始 full-arena 逐字节测试已通过；新增测试也验证了四次 workspace Graph 写入、稳定权重
Graph 读取、不同/相同大小的地址语义、事务 move 与重复 close。
调用方在 `VisionOverlaySession::begin` 按当前地址重新绑定，不能把某一次借出地址当作通用固定 overlay。
具体契约与资源公式见 [memory/design.md](memory/design.md)。

KV 的任意非连续空闲页拼成连续 workspace 仍需要可靠映射，不能用权重尾部方法冒充等价实现。
生成的原生产 `make_kv_arena` 现在检查显式的非连续借用资格，当前为 false，因此沿用原有
普通 DeviceArena 与已验证权重尾部的回退路径。底层 VMM 诊断和原隔离失败断言仍保留；
完整 Vision/Engine 生命周期还需集成验证。
实验替代后端将语义 K/V 页映射到单一普通 HIP payload，以有界 D2D 搬移让出固定连续
workspace，稳定设备页表供 Graph 内核在执行时寻址。原分配器与六组 BF16 消费者功能已
通过；真实 Vision 借用方、工作线程交接与完整 Models 生命周期仍须接入验收，生产资格仍为 false。

本机兼容修正：原候选的 `-mwavefrontsize32` 改为编译器支持的 `-mno-wavefrontsize64`；
本机编译器没有 `__AMDGCN_WAVEFRONT_SIZE` 宏，改用设备 wavefront-size 检查，另保留
gfx1201 编译检查与运行时 wave32 设备检查。没有改动矩阵坐标映射或放宽数值判据。

## 后续接入边界

### 本机正式接入条件

用户要求：只有测试证明显著优于 llama.cpp，才与本地模型部署会话协作正式迁移已有工作。
该条件尚未满足。算子通过、单个 kernel 耗时和 NVIDIA 上游数据都不能替代这个结论。

完整 Engine 可运行后，应在同一 RX 9070 上交替测量同一模型、解码后相同的权重/量化语义、
相同上下文与 KV 正确性边界、并发和采样配置，保留构建参数及 GPU/RAM 预算。
分别记录 prefill、decode、首 token 延迟、稳态与峰值内存和失败率；重复交替运行以排除测量波动。
若量化、历史保留范围或输出质量不同，只能报告相应配置的取舍，不能直接声称后端更快。
确认主要工作负载的收益明显超过波动、质量/稳定性和资源约束满足要求后，再与部署会话接入
其现有调用和配置，并复测实际工作流。当前尚无端到端优势或已完成正式迁移的声明。

本机独立 CPU 转换环境已能运行原转换器；相关 GGUF/MTP/制品与 CLI 的 53 项测试通过。
原格式测试表补齐同一生产 GGML 已定义的 Q2_0（type 42、64 个元素/18 字节），未删除旧项。
现有本地 GGUF 不全满足原配方：Swift 的部分 GDN A/B 控制权重为 Q8，GSQ 部分为 F32，
而原 GDN gating projection 合约只接收 BF16。这两项仍是导入/算子限制，不能通过静默转 BF16
来宣称保持原权重语义。
这是可新增的格式支持，不是 AMD 硬件无解。后续可为小型控制矩阵增加 F32/Q8_0 权重
读取和原生 FP32 FMA 路由，保持控制输出为 FP32、独立 FP64 oracle 及零隐藏 workspace；
归一化控制分支不能借用先生成 BF16 激活或 BF16 logits 的替代路径。该方案尚未实施。

Bonsai 的 65-block 文件现可严格验证全部 866 个张量：识别 15 个已知 MTP 张量，保留主干
ternary 代码/尺度及 MTP 原 Q8_0 字节，未知张量仍拒绝。原转换入口也已补齐必需的 v3
model_id/weights_id 参数与报告；缺省权重身份每次唯一，显式标签保持原样。现有候选前端与
原 GGUF 的 248,077 个 token ID、247,587 条 BPE merges 全部匹配。

第一次真实制品转换触及 2 GiB RAM 保护而终止，没有计作成功。GGUF 源现按请求范围读取，
避免 Windows 整文件 mmap 累积驻留。修复后已实际生成 7,654,216,448 字节的 text+MTP
制品，783 个写入 job、790 个对象，转换时最低可用 RAM 5.112 GiB。它不是完整 Engine
推理或端到端优势的证据。

实际制品的 456 个 direct 控制参数已全量核对：norm、conv、控制矩阵及 dt_bias 在恢复
存储约定与重排后有效值完全一致。A_log 的 2,304 个标量有 3 个在逆 exp/FP32 重构后
产生舍入差异，最大绝对差约 1.91e-6；实际 GPU 数学与整模型质量仍需验证。
主 PQ2/T2 与 MTP Q8 的代表行编码/尺度抽检一致，不能据此宣称整个权重 payload 已逐值核对。
公平参考优先采用双方实际支持的相同 BF16 KV；NInfer Int8Group64 与 llama.cpp Q8_0
不是相同 codec，压缩模式需单列品质与性能取舍。冷启动测试还须关闭驻留提示词复用，
并确认缓存 token 为零；仅设置 `cache-ram=0` 不足以保证这一点。

后续仍须完成未验收的内存借用、算子/矩阵格式、Attention 模式、设备校准、Graph 与整模型
质量/性能验证。迁移收益须在质量、长文正确性和稳定性合格时，体现为同模型、有效权重、
真实历史范围和资源预算下可重复的提速，或显著节省专用/共享显存、物理 RAM/提交量、
SSD I/O，或扩展已验收的有效上下文；同时披露 TTFT/吞吐代价。当前没有正式迁移结论。
AMD gfx1201 不能当作 NVIDIA sm_120 使用，NVIDIA 路由表与性能数字也不适用于本卡。
KVMem 检索或环形缓存仍应遵守原仓对超池长提示词的正确性边界，不能视作全量注意力等价。

云端原始 ZIP、补丁、实现记录及本机接收回执保存在工作区 `research/` 中；原件保持不变。
