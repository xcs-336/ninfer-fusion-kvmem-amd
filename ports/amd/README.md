# RX 9070 / Windows 原生 HIP 适配

开发基线：`1314521gjy/ninfer-fusion-kvmem@4e93bb8ab89b583653a9d16235582c688226ff05`。
主引擎位于 `src-tree/fusion-engine-src/`。保留其 KVMem 环、host-backed 复用和融合算子设计。

当前是适配开发分支，完整 `.ninfer` Engine 尚未在 AMD 上编译或推理。
Dot 提供的原生 gfx1201 BF16 MMA 已接入原有生产 BF16 GEMM 内核并验证；
完整 Linear Op 调度、其他矩阵格式、attention 和模型执行仍需继续移植。
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

## 生产算子本机资格验证

2026-10-20 当前整组共 48 项：46 项通过、KV 借用隔离失败、完整推测解码测试触及
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

| NVIDIA 机制 | 当前 AMD 实现与边界 |
|---|---|
| BF16 Tensor Core MMA | gfx1201 原生 WMMA，已验证片段与上述生产 GEMM kernel |
| ldmatrix x2/x4、转置 | 原生 LDS 装载及 wave32 shuffle，精确布局测试通过 |
| CUB 排序 | hipCUB，采样原测试通过；完整推测解码长套件尚未通过 |
| cp.async / pipeline copy | 先用逐线程完成的 HIP 全局→LDS 复制保持依赖；尚无异步重叠性能结论 |
| PDL 网格依赖 | HIP stream 顺序执行，保留数据依赖与 Graph 捕获语义，暂不实现网格重叠 |
| CUDA Graph | HIP Graph，Core、归一化和生产 BF16 GEMM 路径已验证 |
| NVTX | 原生 Windows ETW 范围事件；尚未验收 GPU 时间线关联 |
| 稀疏 VMM 借用 | 独立诊断出现数据隔离失败、停滞或运行时崩溃，不能作为已支持能力 |
| FP16/INT8/FP8/NVFP4 其他生产 MMA、TMA | 仍需移植与独立验收，不能按 NVIDIA SM 号选择 AMD 路由 |

上述测试不是吞吐或峰值显存基准；该轮有另一会话的模型测试重叠，耗时不作为性能比较。

## 共享 GPU 测试所有权

全部 CTest GPU 入口通过 `test_locked.py` 获得共享锁的 byte 0，再启动测试。当前协作主机
显式配置 `-DNINFER_GPU_LOCK_FILE=D:/programfile/work/deepseek/cache/gpu-benchmark.lock`，
与本地模型部署会话共用。其他机器用该选项指定各项目共同的锁路径。
默认等候 300 秒；锁忙且到期返回 77，表示未运行，不是通过。
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

本机兼容修正：原候选的 `-mwavefrontsize32` 改为编译器支持的 `-mno-wavefrontsize64`；
本机编译器没有 `__AMDGCN_WAVEFRONT_SIZE` 宏，改用设备 wavefront-size 检查，另保留
gfx1201 编译检查与运行时 wave32 设备检查。没有改动矩阵坐标映射或放宽数值判据。

## 后续接入边界

后续必须完成内存借用、其他算子/矩阵格式、attention、AMD 调度和完整 Engine 的接入。
AMD gfx1201 不能当作 NVIDIA sm_120 使用，NVIDIA 路由表与性能数字也不适用于本卡。
KVMem 检索或环形缓存仍应遵守原仓对超池长提示词的正确性边界，不能视作全量注意力等价。

云端原始 ZIP、补丁、实现记录及本机接收回执保存在工作区 `research/` 中；原件保持不变。
