# 项目现状：`main` 研究基座

更新日期：2026-10-07。此处描述 `main` 的代码与证据边界；具体运行结果以执行时的 commit、配置和日志为准。

## 研究状态

研究问题尚未确定。近期讨论关注移动智能体在部分可观测环境中寻找信息、核实观察并完成语言任务，但这仍是探索方向，不能把“主动探索、更有依据地回答、固定算力预算”直接称为新的研究贡献。下一步应从公开任务和现有方法的可重复失败中确定要改善的结果，再选择方法与模型。

`spatial-intelligence-reasoning@8e69463` 保存早期 Habitat/HM3D、空间推理、3DGS/SAM2/CLIP 机制实验和研究探索记录；`3dgs-vln@38dd85c` 目前与旧 `main` 同点。这两个分支暂缓开发，未整体并入新 `main`。早期机制测试不等于论文复现、学习型 VLN 成绩或实车验证。

## `main` 现有代码

| 部分 | 当前内容 | 证据边界 |
| --- | --- | --- |
| `src/vln_policy` | DoorNav 与 Mock 策略。 | 规则/测试策略，不是经公开语言导航数据训练的模型。 |
| `src/vln_sim`、`scripts/run_doornav_sim_loop.py` | Mock 场景、Habitat 适配与基础闭环入口。 | 能验证接口和简单闭环；不等于 HM3D 批量基准。 |
| `src/vln_core` | episode、动作适配、安全过滤和基础评测。 | 可作为实验基础，尚未连接一个确定的研究任务与公开基线。 |
| `src/vln_interfaces`、`src/vln_bringup` | 既有 ROS 2 消息、节点和启动配置。 | 不预设研究方法，也不证明底盘实车接入或停车安全。 |

2026-10-07 Mac 上 `main` 的全部快速测试为 **51 passed**。Mac 不运行真实 3D Habitat 渲染；Linux 承担 GPU/HM3D 实验。两端另有 `habitat-lab` 的 `xjtlu-vln@71037ff` checkout，底层渲染与物理由 Habitat-Sim 提供。

## 外部候选资源

Linux 的 `EXPRESS-Bench@e8789da` 保存具身问答题目、轨迹、Fine-EQA 基线及评分代码；`prismatic-vlms@874c5bb` 是其模型依赖。二者有本地修改或未跟踪文件，不能清理或自动覆盖。题目和轨迹已准备，但官方 7B 配置在 RTX 4060 8GB 上无法完成 GPU 加载，HM3D 语义场景和正式评分条件仍有缺口；目前没有完整主动探索基线结果。它们是候选工具，不是已选题目。

`0.45 m / 640×480 / 90° / 0.38625 m / 10 Hz` 是本仓库的仿真基准。实际相机尚未装车，不能将整套参数作为实车测量结果；见 [AGENTS.md](AGENTS.md)。

## 双端协作

Mac 在 `main` 完成和测试改动后提交、推送。Linux 的 `vln-research-sync.timer` 每分钟调用 [同步脚本](scripts/sync_linux_checkout.sh)，只在 `main` 且工作区干净时快进；外部仓库不自动更新。正式实验前仍须记录 `git rev-parse HEAD`、依赖版本和结果路径，确认 Linux 实际运行的 commit。
