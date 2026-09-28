<a name="top"></a>

<div align="center">

<p>
  <img src="assets/ai-frontier-banner-v2.png" width="100%" alt="AI 前沿学习笔记主题插画：展开的研究笔记、散落的论文与蜿蜒的蓝色路径，连接远山与探索之旅" />
</p>

<h1>AI 前沿学习笔记</h1>

<p><sub>AI FRONTIER NOTES</sub></p>

<p><strong>从数学直觉，到训练实践与系统设计</strong></p>

<p>
  <a href="#news"><img src="https://img.shields.io/badge/News-DC2626?style=flat-square" alt="跳转至最新更新" /></a>
  <a href="#tech-reports"><img src="https://img.shields.io/badge/Frontier_Tech_Reports-2563EB?style=flat-square" alt="跳转至前沿技术报告" /></a>
  <a href="#agentic-rl"><img src="https://img.shields.io/badge/Agentic_RL-0F766E?style=flat-square" alt="跳转至 Agentic RL" /></a>
  <a href="#recursive-self-improvement"><img src="https://img.shields.io/badge/Recursive_Self--Improvement-475569?style=flat-square" alt="跳转至递归自我改进" /></a>
</p>

</div>

---

这里记录我学习 AI 前沿技术时的推导、论文精读与思考，围绕 **前沿技术报告、Agentic RL、递归自我改进** 三个方向，把数学基础、前沿方法和工程实践串起来。

<a name="news"></a>

## News

- **2026.09.28**：发布 **[MiMo V2.6 技术报告解读](TechReport/MiMo/MiMoV2.6.md)**，梳理混合 RL、奖励与行为约束、训练系统及 MOPD2 蒸馏。[原始报告](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)

<a name="contents"></a>

## 内容导航

- [前沿技术报告](#tech-reports)
- [Agentic RL](#agentic-rl)
- [递归自我改进](#recursive-self-improvement)

<a name="tech-reports"></a>

### 01 · 前沿技术报告

拆解模型架构、训练策略与推理系统中的关键设计。

| 笔记 | 核心内容 |
| :--- | :--- |
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

递归自我改进方向，笔记待补充。

[返回顶部](#top) · [最新更新](#news) · [内容导航](#contents)
