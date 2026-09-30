<a name="top"></a>

<div align="center">

<p>
  <img src="assets/ai-frontier-banner-v4.png" width="100%" alt="持续学习、攀登 AGI 之巅：山脚摆放着展开的书与书堆，漫画风格的学习者怀抱书本，沿蜿蜒山路向标有 AGI 的顶峰攀登" />
</p>

<h1>AI 前沿学习笔记</h1>

<p><sub>AI FRONTIER NOTES</sub></p>

<p><strong>从数学直觉，到训练实践与系统设计</strong></p>

<p>
  <a href="#tech-reports"><img src="https://img.shields.io/badge/Frontier_Tech_Reports-2563EB?style=flat-square" alt="跳转至前沿技术报告" /></a>
  <a href="#agentic-rl"><img src="https://img.shields.io/badge/Agentic_RL-0F766E?style=flat-square" alt="跳转至 Agentic RL" /></a>
  <a href="#recursive-self-improvement"><img src="https://img.shields.io/badge/Recursive_Self--Improvement-475569?style=flat-square" alt="跳转至递归自我改进" /></a>
</p>

</div>

---

这里记录我学习 AI 前沿技术时的推导、论文精读与思考，围绕 **前沿技术报告、Agentic RL、递归自我改进** 三个方向，把数学基础、前沿方法和工程实践串起来。

<a name="news"></a>

## News

- **2026.10.01**：发布 **[RSI survey 与最新进展](Recursive-Self-Improvement/RSI-Survey.md)**，结合综述与 11 篇近期论文，梳理自主课程、系统演化、自主后训练与改进机制继承，比较局部收益和长期积累的证据。[原始综述](https://arxiv.org/abs/2609.11873)
- **2026.09.30**：发布并更新 **[DeepSeek Elastic Compute（DSec）解读](TechReport/Deepseek/DeepSeekElasticCompute.md)**，分析环境分层、按需加载与资源管理，补充长轨迹恢复的边界、环境版本与采样偏差等 insight。[原始报告](https://arxiv.org/abs/2609.22978)
- **2026.09.28**：发布 **[MiMo V2.6 技术报告解读](TechReport/MiMo/MiMoV2.6.md)**，梳理混合 RL、奖励与行为约束、训练系统及 MOPD2 蒸馏。[原始报告](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)
- **2026.09.14**：发布 **[DeepSeek V4 解读](TechReport/Deepseek/DeepSeekV4.md)**，梳理注意力机制、MoE、Muon、低精度训练与蒸馏。
- **2026.09.14**：发布 **[DSpark 解读](TechReport/Deepseek/Dspark.md)**，分析投机解码、半自回归生成与置信度调度。
- **2026.09.14**：发布 **[强化学习基础](AgenticRL/BasicRL.md)** 与 **[RL 训练经验与实践](AgenticRL/RLTrainExp.md)**，覆盖策略梯度、PPO、GAE、GRPO，以及训练稳定性、奖励设计和训推一致性。

<a name="contents"></a>

## 内容导航

- [前沿技术报告](#tech-reports)
- [Agentic RL](#agentic-rl)
- [递归自我改进](#recursive-self-improvement)

<a name="tech-reports"></a>

### 01 · 前沿技术报告

拆解模型架构、训练策略、推理系统与 Agent 执行基础设施中的关键设计。

| 笔记 | 核心内容 |
| :--- | :--- |
| **[DeepSeek Elastic Compute（DSec）解读](TechReport/Deepseek/DeepSeekElasticCompute.md)** | 环境分层与按需加载、资源超配、长轨迹恢复、奖励可信度与采样偏差 |
| **[MiMo V2.6 解读](TechReport/MiMo/MiMoV2.6.md)** | 大规模混合 RL、质量评价、行为约束、训推一致性与 MOPD2 蒸馏 |
| **[DeepSeek V4 解读](TechReport/Deepseek/DeepSeekV4.md)** | 注意力机制、MoE、Muon、低精度训练与蒸馏 |
| **[DSpark 解读](TechReport/Deepseek/Dspark.md)** | 投机解码、半自回归生成与置信度调度 |

<a name="agentic-rl"></a>

### 02 · Agentic RL

从强化学习基础，到 LLM 与多轮智能体的训练实践。

| 笔记 | 核心内容 |
| :--- | :--- |
| **[强化学习基础](AgenticRL/BasicRL.md)** | 策略梯度、PPO、GAE、KL 与 GRPO |
| **[RL 训练经验与实践](AgenticRL/RLTrainExp.md)** | 训练稳定性、奖励设计、训推一致性与训练框架 |

<a name="recursive-self-improvement"></a>

### 03 · Recursive Self-Improvement

研究 AI 如何利用经验更新权重、工具、记忆和改进方法，并让后继系统继续学习。

<div align="center">

| 笔记 | 核心内容 |
| :--- | :--- |
| **[RSI survey 与最新进展](Recursive-Self-Improvement/RSI-Survey.md)** | 五层自主性框架、自主课程、agent 程序演化、自主后训练与改进机制继承；近期实验、验证风险与长期积累的边界 |

</div>
