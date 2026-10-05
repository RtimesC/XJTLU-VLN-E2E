# VLN 科研项目核心上下文与技术全景图

## 1. 当前阶段与核心进展

- **空间推理分支 (`spatial-intelligence-reasoning`)**：
  - 已实现自车中心 8 扇区拓扑工作记忆 ([`SpatialSectorMemory`](file:///Users/sousuke/Desktop/XJTLU-VLN-E2E/src/vln_policy/vln_policy/spatial_reasoning_policy.py))。
  - 已实现针对 0.45m 低视角的近场遮挡评估与开阔通道推断 ([`LimitedFovVisualGrounder`](file:///Users/sousuke/Desktop/XJTLU-VLN-E2E/src/vln_policy/vln_policy/spatial_reasoning_policy.py))。
  - 已跑通完整的 `ORIENT` -> `INFERRED_EXPLORE` -> `APPROACH` -> `VERIFY` -> `STOP` 状态机闭环，并接入车载 HUD 仪表盘与 MP4 录制。
  - 本地全部 56 个单元测试 100% 通过（`pytest`，执行耗时 ~0.3s）。
- **仿真基座 (`habitat-lab`)**：
  - 已在专属分支 `xjtlu-vln` 完成低底盘小车（0.45m 相机高度、0.38625m 半径、RGB-D 640x480、4x4 位姿矩阵）专属包装器 `XJTLUCarEnv` 开发与验证。
  - 已通过真实 3D 场景 `skokloster-castle.glb` 验证。

## 2. 仓库与双端架构

- **`XJTLU-VLN-E2E`**：主开发仓库（算法、策略、评测）。
- **`habitat-lab`**：外部仿真引擎（通过 `pip install -e habitat-lab` 安装在 `habitat_vln` Conda 环境中）。
- **双端分工**：
  - **Mac**：策略算法编写、数学推导、重构、快速单测（不运行真实 3D GPU 渲染）；
  - **Linux**：同步 Mac 代码，拉起 NVIDIA 显卡与 Habitat 3D OpenGL 引擎跑真实室内场景仿真与视频录制。

## 3. 常用运行与测试指令

### 在 Mac 本地验证测试套件：
```bash
pytest
# 运行极速 Mock 闭环仿真（必须显式允许 Mock）
python3 scripts/run_spatial_reasoning_sim_loop.py --allow-mock
```

### 在 Linux 机器运行真实 3D Habitat 闭环仿真：
```bash
cd ~/Desktop/XJTLU-VLN-E2E
git pull --ff-only origin spatial-intelligence-reasoning
/home/sousuke/miniforge3/envs/habitat_vln/bin/python scripts/run_spatial_reasoning_sim_loop.py \
  --scene /home/sousuke/Desktop/habitat-lab/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
```

真实场景运行在 Habitat-Sim 不可用或场景缺失时会直接失败，不会静默降级为 Mock。

## 4. 下一步研发候选

1. **`3dgs-vln` 分支开发**：基于 0.45m 高度 RGB-D 和 4x4 位姿，实现三维高斯点云反投影与增量记忆场。
2. **HM3D 数据集接入**：在 Linux 上加载真实居家户型，评测长距离语言导航任务。
