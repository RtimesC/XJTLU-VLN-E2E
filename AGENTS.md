# XJTLU-autonomous-vehicle-vln 核心算法规范

## 1. 项目定位与研究目标

本项目（[`XJTLU-autonomous-vehicle-vln`](file:///Users/sousuke/Desktop/XJTLU-autonomous-vehicle-vln)）是面向 XJTLU 自动驾驶小车的**端到端视觉语言导航（VLN-E2E）核心主算法科研仓库**。

### 核心策略映射
```text
(前视 RGB 观测, 自然语言路线指令, 动作历史)
    -> VLN Policy
    -> (线速度 v_t, 角速度 omega_t, 停止概率 p_stop)
```

本研究中的“端到端”指从视觉与自然语言指令直接映射到底盘连续动作与停止决策，**不把 VLN 降级为 Nav2 目标点或路径生成器**。

---

## 2. 与其他仓库的协同与边界

- **🚗 底层实车工程（[`XJTLU-autonomous-vehicle-rtk`](file:///Users/sousuke/Desktop/XJTLU-autonomous-vehicle-rtk)）**：
  - 原车仓库负责硬件驱动（MID360、RTK、相机、IMU）、TF 变换、底盘串口通信、STM32 固件及车辆级安全保护。
  - 本仓库作为 ROS 2 overlay 运行，只通过明确版本化的 ROS 2 话题接口（如 `/vln/input/image`, `/vln/cmd_vel`）与原车连接，**不复制、不篡改原车底层驱动与固件代码**。
- **🌟 兄弟算法库（[`ETPNav-MID360`](file:///Users/sousuke/Desktop/ETPNav-MID360)）**：
  - 分别探索端到端连续控制（本仓库）与拓扑图/航点提议（ETPNav）两条学术路线，相互对照消融。
- **🖥️ 仿真平台基座（[`habitat-lab`](file:///Users/sousuke/Desktop/habitat-lab)）**：
  - 提供离线仿真训练与基准评测支持。

---

## 3. 实验隔离红线（Strict Isolation）

为了保证学术结论的严谨性与有效性，主实验必须遵守以下红线：

1. **禁止泄漏的信息**：
   - 严禁将 SLAM、FAST-LIO2 位姿、`map -> odom`、全局坐标、GPS/RTK 坐标输入主策略。
   - 严禁将 Nav2 全局路径、局部路径、costmap 或自动绕障决策输入主策略。
   - 严禁在自然语言指令中包含 GPS 经纬度、机器坐标点或程序化转向序列。
2. **安全层与策略严格审计隔离**：
   - 安全层（速度限幅、看门狗超时、急停避障）只作为外部拦截，严禁替策略规划绕障路线或修改行驶方向。
   - 评测必须同时记录原始策略输出、安全过滤后输出和最终底盘执行动作。
3. **推进阶段**：
   - 严格遵循：离线回放 -> Shadow Mode（只推理不控车）-> 仿真测试 -> 低速封闭有人值守实车测试。

---
