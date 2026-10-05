# XJTLU VLN-E2E

面向 XJTLU 自动驾驶小车的团队端到端视觉语言导航（Vision-Language Navigation, VLN）科研项目与代码仓库。

本团队研究由前视 RGB 图像、自然语言指令和有限历史信息驱动的策略，使其直接预测底盘线速度、角速度与停止决策。本项目不把 VLN 简化为 Nav2 目标点或路径生成器。

## 研究分支与仓库边界

本团队在本仓库中探索两条端到端 VLN 研究分支：

- `3dgs-vln`：3D Gaussian Splatting 与 VLN，研究语言条件空间记忆如何支持直接连续控制。
- `spatial-intelligence-reasoning`：受 Fei-Fei Li 启发的空间智能与空间推理，研究有限自我中心观测下的端到端 VLN 决策。

拓扑图/航点方向由团队在独立仓库 `ETPNav-MID360` 中维护；前视 RGB-LiDAR 方向由团队在独立仓库 `FrontRGB-LiDAR-VLN` 中维护。

## 当前状态

- 研究阶段：v0.2 控制链与接口实现，模型仍为 mock/DoorNav 工程基线
- 运行状态：纯 Python 闭环可运行；ROS 2/Habitat 与实车尚未在本仓库验证
- 实车状态：尚未验证，不得用于车辆控制

## 文档

- [研究范围](research_scope.md)
- [ROS 2 接口契约](docs/ros_interface_contract.md)
