# RX 9070 / Windows 原生 HIP 适配

开发基线：`1314521gjy/ninfer-fusion-kvmem@4e93bb8ab89b583653a9d16235582c688226ff05`。
主引擎位于 `src-tree/fusion-engine-src/`。保留其 KVMem 环、host-backed 复用和融合算子设计。

当前是适配开发分支，完整 `.ninfer` Engine 尚未在 AMD 上编译或推理。
这里先接收并验证 Dot 提供的原生 gfx1201 BF16 MMA 候选，未把它替换进生产算子。
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
HIPIFY 原许可保留在生成目录。生成阶段不翻译 PTX，未替换生产算子。
其他机器需用 `-DNINFER_PYTHON=<Python 3.11 路径>` 显式选择 Python 环境。

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
原测试仍将其视作不可借出；尚未改写该夹具。

AMD 上游也记录过尺寸相关的
[hipMemSetAccess 缺陷](https://github.com/ROCm/rocm-systems/issues/2516)，但尚未证实本机全部故障
与其同根因。后续优先确认可用的内存映射路径，必要时为借用机制提供显式的复制方案；
不能通过放宽隔离断言或忽略错误继续推理。

本机兼容修正：原候选的 `-mwavefrontsize32` 改为编译器支持的 `-mno-wavefrontsize64`；
本机编译器没有 `__AMDGCN_WAVEFRONT_SIZE` 宏，改用设备 wavefront-size 检查，另保留
gfx1201 编译检查与运行时 wave32 设备检查。没有改动矩阵坐标映射或放宽数值判据。

## 后续接入边界

必须先移植本基线的 Core / Graph / 内存机制，再替换生产算子的 PTX、ldmatrix、同步和调度。
AMD gfx1201 不能当作 NVIDIA sm_120 使用，NVIDIA 路由表与性能数字也不适用于本卡。
KVMem 检索或环形缓存仍应遵守原仓对超池长提示词的正确性边界，不能视作全量注意力等价。

云端原始 ZIP、补丁、实现记录及本机接收回执保存在工作区 `research/` 中；原件保持不变。
