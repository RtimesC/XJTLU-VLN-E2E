# XJTLU-VLN-E2E

面向移动机器人视觉语言导航与具身任务的研究仓库。当前任务是确定**值得解决且能够验证的研究问题**：先在公开任务中识别现有方法的具体失败，再决定模型、空间表示、动作接口和实验基准。现有空间推理和 3DGS 实现属于探索原型；研究题目尚未锁定。

## 仓库与环境

- 本仓库保存研究策略、实验入口、评测、Habitat 适配和已有 ROS 2 代码。
- `habitat-lab` 是独立的仿真框架 checkout；真实 3D 渲染和物理由 Habitat-Sim 提供。Mac 用于研发和快速验证，Linux 用于 GPU/HM3D 实验。
- Linux 上的 `EXPRESS-Bench` 是候选外部评测资源，`prismatic-vlms` 是其 Fine-EQA 基线相关依赖。它们不是本仓库的子目录，也没有被确定为最终研究主线。

Mac 端完成并测试一组改动后提交、推送当前研究分支；Linux 用户级定时器每分钟检查一次，只对干净且分支匹配的主仓库执行快进同步。自动同步的脚本和服务单元位于 [`scripts/sync_linux_checkout.sh`](scripts/sync_linux_checkout.sh) 与 [`ops/systemd/`](ops/systemd/)。

当前仿真使用 `0.45 m` 相机高度、`640×480`、`90°` HFOV、`0.38625 m` agent 半径及 `10 Hz` 步进；这些数值不等同于已核实的车载相机和模型运行性能。

## 阅读入口

- [AGENTS.md](AGENTS.md)：项目协作规则与双端边界。
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)：四个目录的职责、Mac/Linux 状态和现有证据边界。
- [研究方向探索记录](docs/research_exploration_journey_2026-10.md)：从论文方法到研究问题的讨论过程。
- [论文 1 机制实验记录](docs/paper1_initial_findings.md)：早期原型与真实 HM3D 测试的范围和局限。
