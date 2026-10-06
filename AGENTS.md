# XJTLU-VLN-E2E AI Agent 协作行为准则与红线规范

> **[MUST READ]** 本文件为本仓库最高优先级的 Agent 行为规范。任何 AI Agent 在本仓库工作前必须完整阅读并严格遵守。

---

## 1. 仓库分工与物理架构关系 (Repository Topology)

| 仓库名称 | 物理定位 | 核心职责 | 协作方式 |
| :--- | :--- | :--- | :--- |
| **`XJTLU-VLN-E2E`**<br>(当前主仓库) | **算法大脑与系统集成**<br>(Decision & Policies) | • 维护导航策略 (`vln_policy`)、空间推理与 3DGS 算法。<br>• 维护实验评测器及已有的 ROS 2 消息实现；这些实现不预设研究方法、模型输入或动作接口。<br>• **所有实验脚本的主入口均在此仓库启动**。 | 主仓库，跨平台开发 |
| **`habitat-lab`**<br>(兄弟仓库) | **物理孪生与 3D 渲染**<br>(Physics & Simulator) | • 加载 3D 室内场景网格（HM3D / MP3D / `skokloster-castle.glb`）。<br>• 提供 0.45m 低高度前视相机 RGB-D 与 4x4 位姿渲染。<br>• 计算物理碰撞与真值测地线距离。 | 基础引擎，通过 `pip install -e habitat-lab` 注册为底层 Python 依赖 |

---

## 2. 双端分工与环境边界协议 (Dual-Machine Protocol)

本项目采用 **Mac 算法研发 + Linux 物理仿真** 的双端协作模式，Agent 必须严格遵守对应环境的动作边界：

### 🍎 当 Agent 运行在 Mac 本地时：
- **[ROLE]** 算法研发与架构大脑 (Design, Refactor, Fast Unit Testing)。
- **[PERMITTED]** 
  - 编写与修改纯 Python 策略模块、数学推理逻辑、测试用例；
  - 运行 `pytest` 快速单元测试（测试必须在 1 秒内执行完毕）；
  - 运行基于 `MockSceneAdapter` 的快速无头几何闭环仿真；
  - 提交并推送稳定的经过测试的代码到 GitHub 远端分支。
- **[PROHIBITED]** 
  - 严禁在 Mac 上尝试编译底层 CUDA 算子或启动真实 3D Habitat-Sim 渲染（Mac 缺少独立 GPU 与 Habitat-Sim C++ 环境）；
  - 严禁在 Mac 上尝试运行物理实车 ROS 2 底盘驱动节点。

### 🐧 当 Agent 运行在 Linux 工作机时：
- **[ROLE]** 3D 物理仿真与 GPU 基准评测端 (Simulation & Benchmark Execution)。
- **[PERMITTED]** 
  - 通过 `git pull` 同步 Mac/GitHub 推送的最新策略代码；
  - 挂载 `habitat-lab` 的 3D 网格场景，调用 NVIDIA 显卡与 OpenGL/CUDA 渲染；
  - 运行真实 3D 闭环仿真脚本并生成录像与评测报告；
  - 运行 ROS 2 Humble 节点。
- **[PROHIBITED]** 
  - **严禁在 Linux 端随意大幅手写/篡改核心业务策略代码**（所有策略改动优先在 Mac 端经单元测试验证后推送，避免产生双端 Git 冲突）；
  - 严禁关闭、重置或杀死现存的网络代理端口（如 10808）。

---

## 3. 小车不可变更的物理指纹契约 (Immutable Hardware Contracts)

任何 Agent 在编写策略、包装器或配置时，**严禁使用 Habitat 官方默认的 1.25~1.5m 人眼视角**，必须严格固化以下小车实车参数：

- **相机安装高度**：`sensor_height = 0.45` 米（前视居中，低视角，存在大面积地面透视与近景遮挡）；
- **相机视场角与分辨率**：`width = 640`, `height = 480`, `hfov = 90.0` 度；
- **底盘物理外形**：`agent_radius = 0.38625` 米，`agent_height = 0.5` 米；
- **动作控制空间**：差速驱动（`linear_velocity`, `angular_velocity`），步进频率 `10Hz`（`dt = 0.1s`）。

### 3.1 实车核心数据与证据边界

实车资料以 [`XJTLU-autonomous-vehicle`](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle) 的 `main@7bdf7afdb3bc8a8fc105076235cdb6f5445de09f` 为本次只读核查快照；今后涉及硬件结论或实车实验，须重新核对该仓库当前配置和车辆现场状态。上面的 0.45 m / 640×480 / 90° / 0.38625 m / 10 Hz 是**本仓库当前仿真基准**，不能整体当作已实测的车载相机、模型推理频率或底盘性能。

| 项目 | 上游仓库证据 | 使用边界 |
| :--- | :--- | :--- |
| 车载算力 | [硬件规格](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle/blob/7bdf7afdb3bc8a8fc105076235cdb6f5445de09f/docs-CN/hardware_spec.md)：Jetson Orin NX 16GB、Ubuntu 22.04、ROS 2 Humble。 | Jetson 使用统一内存；Linux 仿真机的独显/显存和推理速度不能代表车载预算。 |
| 车体与驱动 | 同一规格文档记载整车约 650×500×450 mm、六轮（四个动力轮、两个被动轮）、STM32F407 RM C Board、四个 RM3508；`0.38625 m` 是 Nav2 机器人半径配置。 | 约 25 kg 是仿真质量，并非实车称重；仿真圆形 footprint 不等同于精确车体碰撞模型。 |
| 实装感知 | 同一规格文档记载 Livox MID-360（2026-03-20 实测安装高度 0.447 m）、WIT/BMI088 相关 IMU；[README](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle/blob/7bdf7afdb3bc8a8fc105076235cdb6f5445de09f/README.md) 将当前 GNSS 列为基础 NMEA、无 RTK。用户于 2026-10-06 确认实验室已有深度相机，但**尚未安装到小车**，未确认具体型号。 | 0.447 m 是**雷达**高度，不能用于证明相机高度。D455f 在上游硬件规格中列为未集成的扩展模块；前视相机实际型号、安装高度、内参、帧率和 ROS 图像 topic 仍待安装后核实。 |
| 运动与限速 | 上游底盘经 `/cmd_vel → serial_twistctl → STM32 → RM3508` 执行 `v,ω`；硬件规格记载较早测试的 0.7 m/s、1 rad/s；[2026-04-01 日志](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle/blob/7bdf7afdb3bc8a8fc105076235cdb6f5445de09f/docs-CN/devlog/2026-04.md)某次室内运行记录最高 0.681 m/s；当前 [MPPI 配置](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle/blob/7bdf7afdb3bc8a8fc105076235cdb6f5445de09f/src/bringup/config/nav2_explore.yaml)为 `vx_max=1.0`、`wz_max=1.2`。 | 配置上限、某次测试速度与车辆可靠持续能力不同；本仓库 `vln_params.yaml` 的 0.8 m/s、1 rad/s 只是 VLN 软件限幅。仿真 10 Hz 是步进频率，不是已验证的模型推理频率。 |
| 资源与验证 | [2026-03-31 室内记录](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle/blob/7bdf7afdb3bc8a8fc105076235cdb6f5445de09f/docs-CN/devlog/2026-03.md) 记载传统 FAST-LIO2 + PGO + Nav2 MPPI 导航约 16 分 59 秒、1009 个 1 Hz `tegrastats` 样本：RAM 均值 3.224 GB、GR3D 均值 55.79% / 峰值 97%。 | 这是**无 VLN 模型的历史单次基线**，不能推算模型可用显存、实时频率或 VLN 成功率；须在当前车上同负载复测。 |

上游[硬件规格](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle/blob/7bdf7afdb3bc8a8fc105076235cdb6f5445de09f/docs-CN/hardware_spec.md)记载轮径 85 mm，而[固件源码](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle/blob/7bdf7afdb3bc8a8fc105076235cdb6f5445de09f/src/firmware/rm_c_board/Core/Src/Motor_Speed_pid.c)使用 `r=0.1 m`；未标定前不得据此推断里程计/速度精度。当前[串口节点](https://github.com/kevinlasnh/XJTLU-autonomous-vehicle/blob/7bdf7afdb3bc8a8fc105076235cdb6f5445de09f/src/sensor_drivers/serial_twistctl/src/serial_twistctl_node.cpp)的 5 秒无消息分支只记录等待日志；不可把本仓库的 watchdog 目标区间视为已验证的实车停车能力。

---

## 4. Git 与工程卫生纪律 (Git Hygiene)

1. **大文件防污染**：仿真生成的 `.mp4` 视频、点云文件、场景网格 `.glb`、ROS 2 `.bag` 数据包一律被 `.gitignore` 忽略，**严禁提交入库**。
2. **测试守门员**：任何代码修改在提交前，必须在本地运行并通过现有全部测试套件（`pytest`，当前基线 56+ 个测试用例）。
3. **清晰的提交与分支**：
   - 探索分支命名规范：`spatial-intelligence-reasoning`（空间推理）、`3dgs-vln`（三维高斯）；
   - 提交信息采用 Conventional Commits 规范（`feat:`, `fix:`, `docs:`, `test:`）。

---

## 5. 对话启动与双端同步流程 (Conversation and Sync Workflow)

1. **新对话先确认项目规则**：在本仓库目录启动的 Codex 对话必须以本文件为项目协同规则来源；开始工作前先检查当前目录、分支和 Git 工作区状态。
2. **Mac 是策略代码的修改端**：策略、接口、评测器和测试优先在 Mac 修改，先运行快速单元测试，再提交并推送到当前研究分支。
3. **Linux 是真实仿真与评测端**：Linux 只通过 `git pull --ff-only` 同步研究分支，然后运行 Habitat/HM3D、GPU 仿真和结果采集；不得在 Linux 端直接提交核心策略修改。
4. **两端使用同一个研究分支**：不得建立按机器区分的长期分支。实验前必须记录 `git rev-parse HEAD`，实验报告、日志和录像必须能对应到该 commit。
5. **同步前保留本地工作**：发现未提交修改或未跟踪实验文件时，不得擅自 `reset`、`clean` 或覆盖；先检查文件归属并保留用户数据。
6. **实验产物不入库**：录像、点云、场景网格、rosbag 和运行日志放在被 `.gitignore` 排除的目录中；代码提交只包含可复现的配置、脚本和必要的小型摘要。
