# XJTLU-VLN-E2E

面向移动机器人视觉语言导航与具身任务的研究仓库。当前优先确定具体、可检验且有价值的研究问题；公开任务、模型、动作接口和最终方法尚未选定。

## 当前主线

日常研究在 `main` 进行。`spatial-intelligence-reasoning` 和 `3dgs-vln` 保留既有探索历史，暂缓开发；它们的原型与实验结果不会自动成为 `main` 的能力或研究结论。

`main` 当前包含基础策略、Mock/Habitat 适配、动作与安全接口、评测器以及已有 ROS 2 消息和节点。它尚无经公开语言任务验证的学习型导航或具身问答方法。Linux 上的 EXPRESS-Bench 与 Prismatic 资源是独立的候选评测依赖，尚未成为正式研究主线。

Mac 完成改动并测试后提交、推送 `main`；Linux 用户级定时器每分钟检查一次，在主仓库干净且分支匹配时快进同步。自动同步仅作用于本仓库，不更新外部数据或依赖。

## 阅读入口

- [AGENTS.md](AGENTS.md)：协作规则、仿真参数和双端边界。
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)：`main` 的代码、实验和外部资源现状。
