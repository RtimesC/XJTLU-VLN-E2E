# 项目上下文：研究问题、仓库职责与双端状态

更新日期：2026-10-07。本文件记录当前工作边界和已核对的目录状态；具体实验结论以对应 commit、配置和结果记录为准。

## 1. 当前研究阶段

项目仍在确定研究问题。近期讨论从 3D Gaussian Map、MindCube 等方法入口，转向考察移动智能体在部分可观测环境中如何寻找信息、核实观察并完成语言任务。这个方向尚不是确定的课题或方法创新；EXPRESS-Bench、OVON 等也尚未被选定为正式主基准。先用可复现的公开任务和现有方法找出具体、反复出现的失败，再决定要改善的结果、方法和动作接口。

早期 `spatial-intelligence-reasoning` 和 3DGS 代码是探索原型。Habitat 场景、RGB-D、地图构建和视频链路的运行，证明相关机制和依赖能够工作；使用仿真真值位姿、规则式动作预测的小规模测试，不能证明论文复现、学习型 VLN 成绩或实车能力。过程见 [研究探索记录](docs/research_exploration_journey_2026-10.md)、[早期实验记录](docs/paper1_initial_findings.md)。

## 2. 四个目录的职责

| 目录 | 来源与职责 | 本项目当前关系 |
| --- | --- | --- |
| `XJTLU-VLN-E2E` | 我们的研究代码：策略原型、实验入口、评测、Habitat 适配和已有 ROS 2 消息/节点。 | 唯一的项目主仓库；正式实验从这里启动并记录 commit。已有 ROS 代码不表示已经接入实车底盘。 |
| `habitat-lab` | Meta 的 Habitat-Lab 框架及本项目的场景适配代码；底层 3D 渲染和物理仿真由 Habitat-Sim 提供。 | 两端各有独立 checkout，Linux 使用它加载场景并运行 GPU 仿真；场景数据不属于主仓库。 |
| `EXPRESS-Bench` | 外部主动具身问答任务、数据、Fine-EQA 基线和评分代码。 | Linux 上的独立候选评测资源；尚未成为已跑通的正式主基准，也不应并入本仓库源码。 |
| `prismatic-vlms` | Prismatic VLM 外部模型库。 | 当前作为 Fine-EQA 相关依赖存在于 Linux；它不是本项目的策略模块或研究目标。 |

`XJTLU-VLN-E2E` 中的实验入口可以调用外部代码和数据，但要显式记录外部版本、配置及本地补丁。论文原始设置、我们的小车仿真设置和实车状态是三种不同条件，结果不得混写。

## 3. Mac 与 Linux 的实际分工

| 端 | 主要工作 | 当前核对状态 |
| --- | --- | --- |
| Mac | 修改研究代码和文档、快速单测、Mock 几何验证。 | 主仓库 `spatial-intelligence-reasoning@f972bf6`；`habitat-lab` 为 `xjtlu-vln@71037ff`。研究探索文档仍是未跟踪文件。 |
| Linux | 通过同一研究分支同步代码，运行 Habitat/HM3D、GPU 模型与评测，保存带版本号的结果。 | 自动同步启用前主仓库在 `d8c782b`；首次快进后到达 `d9d1c46`，定时器已自行触发并成功检查。`habitat-lab` 为 `xjtlu-vln@71037ff`。EXPRESS-Bench 为 `e8789da`，Prismatic 为 `874c5bb`，两个外部 checkout 均有本地修改或未跟踪文件。 |

上表是 2026-10-07 的部署记录，后续状态应通过 Git 和 `systemctl --user status vln-research-sync.timer` 实时核对。Mac 完成测试后提交并推送；Linux 用户级 `vln-research-sync.timer` 每分钟调用 [同步脚本](scripts/sync_linux_checkout.sh)，只在当前研究分支匹配、主仓库工作区干净时执行 `git pull --ff-only`。外部仓库不自动更新。Linux 实验前仍须检查 `git status`、HEAD 与定时器状态，将主仓库和外部依赖 commit 写进结果记录。自动同步不追溯改变既有实验的版本。

## 4. 已有能力与尚缺的证据

- 主仓库有 Habitat 适配器、Mock 场景、空间推理与 3DGS 探索原型、动作/安全接口和基础评测器。这些组件可供研究使用，不预设最终模型输入、输出或研究方法。
- Linux 已准备 EXPRESS-Bench 题目与轨迹、独立 `fine-eqa` 环境和 Prismatic 权重；据已核对的安装报告，7B 模型可在 CPU 加载，但在 RTX 4060 8GB 上无法完成官方配置的 GPU 加载，HM3D 语义场景仍缺，正式评分还依赖外部 API。因此目前没有完整 Fine-EQA 主动探索结果。
- `0.45 m / 640×480 / 90° / 0.38625 m / 10 Hz` 是本仓库的仿真基准。实验室深度相机尚未装车；这些数值不整体代表已实测的相机、模型推理频率或底盘能力。实车证据边界见 [AGENTS.md](AGENTS.md)。

## 5. 阅读顺序

1. [AGENTS.md](AGENTS.md)：仓库行为规则、双端同步和仿真参数。
2. 本文件：当前研究阶段及目录职责。
3. [研究探索记录](docs/research_exploration_journey_2026-10.md)：研究问题如何变化，以及哪些判断仍是候选。
4. 具体实验文档、脚本及带 commit 的结果：判断某项能力或结论是否已经验证。
