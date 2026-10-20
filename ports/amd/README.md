# RX 9070 / Windows 原生 HIP 适配

开发基线：`1314521gjy/ninfer-fusion-kvmem@4e93bb8ab89b583653a9d16235582c688226ff05`。
主引擎位于 `src-tree/fusion-engine-src/`。保留其 KVMem 环、host-backed 复用和融合算子设计。

当前是适配开发分支，完整 `.ninfer` Engine 尚未在 AMD 上编译或推理。
Dot 提供的原生 gfx1201 BF16 MMA 已接入原有生产 BF16 GEMM 内核并验证；
其他矩阵片段、四套 GDN 和原生权重恢复/Graph 也已通过下述独立测试；
完整 Linear Op、attention 和模型执行仍在接入。
旧 `ninfer-all` HIP 实验的核心测试不能视作本基线的测试结果。

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
KV append/materialize/stream 的独立测试已实际编译链接，算术 GPU 验证正在进行；
完整 causal 库仍待 O3 构建，不能把独立子库的链接当作全部注意力验收。
`NINFER_AMD_LINEAR=ON` 接入原 public Linear 分派与格式家族，实际 O3 构建和数学资格进行中。
两者都需要 Core 和 Basic Ops；不能借 HIP 的 major/minor 号选择 NVIDIA 调度表。

`NINFER_AMD_ARTIFACT=ON` 构建真实 v3 reader/binder/materializer；该库及三个原测试已编译链接，
读取/绑定 CPU 测试通过，GPU materialize/permute 和 Python writer 互通尚待运行。
`NINFER_AMD_RUNTIME_SUPPORT=ON` 接入原 Engine 资源、上下文成本和 profile 支持库；
它们是完整 Engine 的依赖，不是可运行模型的替代入口。

## 生产算子本机资格验证

2026-10-20 首轮生产套件共 48 项：46 项通过、KV 借用隔离失败、完整推测解码测试触及
60 秒时间限制。30 个基础算子测试中 29 个通过，覆盖归一化、embedding、卷积、采样、
MTP round、位置、scatter、激活和输出 logprob。所有原有数学 oracle 和误差判据保留。
推测解码有大量 CPU oracle 工作，需进一步定位或按原有专项模式拆分验证；超时没有算作通过。

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
这消除了 embedding 的真实依赖缺口；GGUF 各量化格式的矩阵数值验证尚未完成。

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
`test_gdn_replay_fold` 还依赖尚未迁移的
输入投影融合家族，保留在上游，需随该依赖一起验收，不能计作通过。
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
| CUB 排序 | hipCUB，采样原测试通过；完整推测解码长套件尚未通过 |
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
建议先采用可证明完全空闲的物理连续 KV 段，无法满足时借用已验证的权重尾部；另一个方案是
启动时预留并明确计入容量预算的 workspace。目前未实现这两项 KV 策略，也不宣称原非连续借用可用。

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

后续必须完成内存借用、其他算子/矩阵格式、attention、AMD 调度和完整 Engine 的接入。
AMD gfx1201 不能当作 NVIDIA sm_120 使用，NVIDIA 路由表与性能数字也不适用于本卡。
KVMem 检索或环形缓存仍应遵守原仓对超池长提示词的正确性边界，不能视作全量注意力等价。

云端原始 ZIP、补丁、实现记录及本机接收回执保存在工作区 `research/` 中；原件保持不变。
