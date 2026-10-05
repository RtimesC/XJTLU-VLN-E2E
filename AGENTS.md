# XJTLU-VLN-E2E AI Agent 协作行为准则与红线规范

> **[MUST READ]** 本文件为本仓库最高优先级的 Agent 行为规范。任何 AI Agent 在本仓库工作前必须完整阅读并严格遵守。

---

## 1. 仓库分工与物理架构关系 (Repository Topology)

| 仓库名称 | 物理定位 | 核心职责 | 协作方式 |
| :--- | :--- | :--- | :--- |
| **`XJTLU-VLN-E2E`**<br>(当前主仓库) | **算法大脑与系统集成**<br>(Decision & Policies) | • 维护所有的导航策略 (`vln_policy`)、空间推理与 3DGS 算法。<br>• 维护 ROS 2 接口契约 (`vln_interfaces`) 与指标评测器 (`vln_evaluator`)。<br>• **所有实验脚本的主入口均在此仓库启动**。 | 主仓库，跨平台开发 |
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

---

## 4. Git 与工程卫生纪律 (Git Hygiene)

1. **大文件防污染**：仿真生成的 `.mp4` 视频、点云文件、场景网格 `.glb`、ROS 2 `.bag` 数据包一律被 `.gitignore` 忽略，**严禁提交入库**。
2. **测试守门员**：任何代码修改在提交前，必须在本地运行并通过现有全部测试套件（`pytest`，当前基线 56+ 个测试用例）。
3. **清晰的提交与分支**：
   - 探索分支命名规范：`spatial-intelligence-reasoning`（空间推理）、`3dgs-vln`（三维高斯）；
   - 提交信息采用 Conventional Commits 规范（`feat:`, `fix:`, `docs:`, `test:`）。
