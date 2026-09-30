# MiMo V2.6 解读：大规模 RL 如何把探索转成有效学习

> [!IMPORTANT]
>
> **MiMo RL 的核心思路是：用大 batch 扩大探索，再用更好的评分和训练系统，把更多尝试转成能力提升。**
>
> 它让模型同时尝试大量编程、工具操作和网页设计任务，不仅判断是否做对，还比较哪种做法更值得学习 （更多的算力转向评分计算）。
>
> 大 batch 是规模化的抓手，评分与系统是让它有效的条件；论文验证了这套组合的效果，但尚未单独证明同等预算下大 batch 更优。

## Take Home Message

* **扩大采样必须同步提高反馈分辨率。** MiMo 每步处理 **1,568 道题 × 16 次尝试**；生成与评分合占 Pro / Flash 本轮强化学习（Reinforcement Learning，RL）成本的 **56.5% / 59.1%**。更多 token 只有转成可信、可区分的反馈，才能支持有效更新。

* **已经做对的题，仍可以学习如何做得更好。** **Groupwise Reward Synthesis（组内奖励合成，GRS）** 用实现与行为质量区分全过测组；**Groupwise Advantage Redistribution（组内优势重分配，GAR）** 把成败混合组中更多正向学习权重分给优质成功解。两者不仅延续信号，也把可维护性等偏好写入目标。

* **实际课程由数据、过滤和调度共同决定。** 数据源平均耗时相差 **66 倍**；先完成先训练偏向快题，过滤无差异组偏向当前能区分成败的题。来源数量配齐，不代表内部难度分布或更新权重相同。

* **系统执行差异可能伪装成策略变化。** 量化、专家路由和截断采样都会改变实际概率；MiMo 重放生成路径和候选集，再计算更新权重。router 漂移则是另一问题：恢复实验表明，这次负载恶化没有带来可见成绩收益。

* **蒸馏也在选择教师能够可靠指导的状态。** **Multi-Prefix Multi-Teacher On-Policy Distillation（多前缀、多教师在线策略蒸馏，MOPD2）** 让学生从教师或示范的中途自行续写，减少教师面对陌生历史的失准；完整学生轨迹保留从头规划和连续纠错。局部决策与长程自主性需要分别训练。

* **能力提升不等于效率已证明，更不等于递归闭环已完成。** DeepSWE 三次成绩平均值从 Pro 的 **58.4 升至 72.6**、Flash 的 **48.7 升至 65.7**，解题 token 也增加。报告未提供完整等预算对照，也未展示模型自主迭代任务、评分器和训练配方。

## 1. 规模化的核心是把尝试转成有效学习信号

**生成、评分和参数更新需要一起扩展。** 单条 agent 轨迹包含读文件、调用工具、修改、测试和纠错；更多尝试只是创造机会，可靠比较才决定学什么。

来源：[技术报告](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/resolve/main/MiMo_V2_6_technical_report.pdf) §3–4.1、§5.1、Figures 3、9。

<p align="center">
  <a href="assets/mimo-v2.6/figure-03-rl-cost-and-gains.png"><img src="assets/mimo-v2.6/figure-03-rl-cost-and-gains.png" alt="原报告 Figure 3：累计 RL 成本对应的 DeepSWE 成绩，以及生成、评分、训练的成本占比" width="500" /></a>
</p>

*Figure 3，p.8。左图显示本次运行随投入增长的成绩，右图显示预算分配；不是不同 batch 的等成本对照。*

| 本轮 RL 成本 | Pro | Flash |
| --- | ---: | ---: |
| 生成 / 评分 / 参数更新 | 43.8% / 12.7% / 43.5% | 44.9% / 14.2% / 40.9% |
| 总成本 | 约 \$2.6M | 约 \$0.9M |

每步训练使用 25,088 条轨迹，动态过滤意味着实际生成量可能更多。2.7B–3.7B training tokens 包含上下文，不能当成新生成量；成本也不包含预训练和全部研发投入。

> **关键 insight：原始 token 不是最终的学习资源，可用的比较信号才是。** 同题尝试若奖励完全相同，组内相对更新就缺少方向。扩展评分，是把更多探索转成可用差异；成本占比本身不能证明预算分配最优。

DeepSWE 指标为 **average@3：三次成绩的平均**，不是三次中至少一次成功的 pass@3。成绩提升总体伴随评测 total tokens 增长（Figure 9）。因此还需比较**固定解题预算下的成功率，或达到同一成功率所需的 token 与时间**，才能区分更有效的策略和更多检查、重试。

<details>
<summary>展开 Figure 9：成绩与 token 同时增长</summary>

<p align="center">
  <a href="assets/mimo-v2.6/figure-09-scores-and-token-growth.png"><img src="assets/mimo-v2.6/figure-09-scores-and-token-growth.png" alt="原报告 Figure 9：三类任务在 RL 过程中的成绩与总 token 数" width="680" /></a>
</p>

*Figure 9，p.22。上排是成绩，下排是评测 total tokens，单位为千；深橙为 Pro，浅橙为 Flash。下排不是训练 token，也不能直接读成纯思考 token。*

</details>

训练链路是 **预训练 → Mid-training → 短期监督微调（Supervised Fine-Tuning，SFT）→ 混合任务 RL（MixRL）→ MOPD2**：预训练建立知识与观察能力；Mid-training 加入 agent 轨迹、扩至 1M 上下文，并提前适配 MXFP4 低精度与 Muown 优化器；短期 SFT 形成策略起点；MixRL 用执行反馈学习；MOPD2 用领域教师合并和拓展能力。

**RL 起点不只是部署权重。** 它继承 SFT 的 FP32 主权重和 Muown 行状态。报告未详细披露短期 SFT 配方；架构与复现配置见附录 A、B。

## 2. 从区分成败，推进到区分成功的做法

**二元测试会在全过测时饱和，质量评价让学习目标继续细化。** MiMo 使用 GRPO（Group Relative Policy Optimization，组相对策略优化），在同题尝试之间形成相对信号。

来源：§4.3、Eqs. (2)–(5)、Figures 7–8。

<p align="center">
  <a href="assets/mimo-v2.6/figure-07-groupwise-grading.png"><img src="assets/mimo-v2.6/figure-07-groupwise-grading.png" alt="原报告 Figure 7：GRS 离线生成标准、在线打分；GAR 在线比较并重分配优势" width="640" /></a>
</p>

*Figure 7，p.17。左侧改变奖励，右侧改变优势；用于不同代码子集，不是固定串行步骤。*

| 机制 | 如何产生差异 | 适用范围 |
| --- | --- | --- |
| **GRS** | 离线联合检查尝试、需求和仓库，形成实现与行为标准；在线逐条打分 | 部分高通过率任务，全过测组也可能继续学习 |
| **GAR** | 在线比较成败混合组，把更多正向权重分给更好的成功补丁 | 其余代码任务中的成败混合组 |

GRS 的奖励为：

```math
R_i=R_i^{\mathrm{test}}\,S_i^{\mathrm{sol}}\,S_i^{\mathrm{beh}}.
```

$`R_i^{\mathrm{test}}`$ 为二元测试分，$`S_i^{\mathrm{sol}}`$、$`S_i^{\mathrm{beh}}`$ 为实现和行为质量分。**测试是门槛，质量是门内排序**：失败仍为零，成功再比较边界处理、修改范围、仓库惯例和验证行为。

GAR 则重分配已有的正向信号。例如两条成功轨迹原本各 +0.5，可调整为 +0.8、+0.2（未限幅示例）；实际还有限幅和全组中心化，不保证正优势总量始终守恒。全成功组的初始优势为零，GAR 不能凭空制造差异；grader 不可用时退回原优势。

> **关键 insight：质量评分既提高信号分辨率，也改变优化目标。** 报告观察到无在线评分时更易出现吞异常、放宽验证和不必要兼容分支。引入可维护性偏好，不能只解释成二元目标的降噪。

GRS 将任务分析变成可反复使用的监督，GAR 每次比较当前出现的方案：由此推知，前者适合标准稳定的重复任务，后者能检查新策略。报告未提供两者的等成本对照或标准失效率。

**GAR 的对照支持持续优化，不支持 token 不再增长。** Flash 纯代码、batch 128、token-mean 实验中，有 GAR 的成绩改善持续到第 52 步，轮数大体稳定，但总长度仍约从 140K 增至 210K。

<details>
<summary>展开 Figure 8：GAR 的成绩、轮数与长度对照</summary>

<p align="center">
  <a href="assets/mimo-v2.6/figure-08-gar-ablation.png"><img src="assets/mimo-v2.6/figure-08-gar-ablation.png" alt="原报告 Figure 8：启用与不启用 GAR 的 Flash 纯代码 RL 对照" width="600" /></a>
</p>

*Figure 8，p.19。灰线无 GAR，橙线有 GAR；实验设置不同于最终混合任务 RL。三栏应一起读，不能由轮数稳定推出总长度不增长。*

</details>

**评价行为仍不等于识别每一步的因果贡献。** GAR 把优势广播到整条轨迹，故还需要两类约束：

* **长度惩罚**：同题通过率足够高才启用，用成功解的生成长度分位数作参照，只扣明显更长的成功解；困难题保留探索空间，快速失败不成为效率标准。
* **局部纠错**：规则标记无效工具调用等 token；正优势时取消其鼓励，负优势时放大其负权重，再在整个 batch 补偿未标记 token。按优势正负分支，不直接按测试成败；未标记也不代表正确。

两者分别处理成功后的冗余和可识别错误，还不是完整的长程信用分配。公式见附录 B。

## 3. 验证器决定什么会被学到，也决定哪些解法能被探索

**可靠反馈既要拒绝假成功，也要接受正确替代解。** 只防 reward hacking，会漏掉过严测试造成的监督偏差。

来源：§4.2、Table 2、Figures 4–6。

| 任务 | 核心验证方式 | 要避免的误判 |
| --- | --- | --- |
| 代码 | 需求—测试对齐；每题 4 次尝试联合审计；有参考补丁时要求修复前后测试 8 次重跑稳定 | 强制参考实现的偶然细节，或漏掉需求 |
| 通用工作流 | 可重置 sandbox；代码查确定性状态，模型评审开放交付物 | 回复声称完成，实际文件或数据库不正确 |
| 视觉设计 | 开放设计逐件检查后同题比较；按图复刻主要查相似度 | 混用开放创作与明确参考的评价标准 |
| 漏洞复现 | 题面与验证来自同一错误报告，匹配漏洞类型和项目栈帧位置 | 将任意崩溃当成指定漏洞 |

代码的 F2P（fail-to-pass，修复前失败、修复后通过）检查修复效果，P2P（pass-to-pass，前后均通过）检查回归；审计则联查需求、补丁和执行记录。**重复执行检验稳定性，反例与需求审计检验正确性；稳定地误判仍会被强化。** 通用任务也通过重复评审、不同 judge 和对抗解修订标准，训练时由 MiMo-V2.6-SFT 评分。

> **关键 insight：过严验证器会压缩可学习的解法空间。** 如果新的有效策略总被判失败，更多采样也可能只是强化参考解附近的做法。因此审计不仅防作弊，也保护探索多样性；这是机制推论，报告未测量其独立增益。

报告记录了从新版包、上游仓库或 issue / PR 获取现成补丁的行为，越过历史版本修复任务的信息边界。防御覆盖纠错示范、环境与缓存清理、网络隔离、训练前 hack agent 检查和训练中持续审计；确认违规后奖励归零，再重算组统计。

**验证器面对的是持续优化的策略。** 模型变强后，旧样本未覆盖的捷径可能变得可达，审计必须跟随新轨迹。报告的“低于 2%”是检测出的违规占比，缺少漏检率，不能当成全部违规率。执行状态、针对性测试与对抗解提供额外约束；单纯重复同一 judge 的偏好不足以建立可靠反馈。

## 4. 过滤、调度和 loss 共同形成实际课程

**计划采样多少、保留多少、更新时权重多大，是三种不同配比。** 来源数量配齐，仍可能掩盖难度变化和长度偏置。

来源：§4.1、§5.1、§6.2–6.3、Figures 15–16。

### 4.1 动态过滤让课程随模型能力变化

简化推导：同题 16 次独立尝试、单次成功率均为 $`p`$，只保留成败混合组，则：

```math
P_{\mathrm{keep}}(p)=1-p^{16}-(1-p)^{16}.
```

成功率为 1% 或 99% 时，约 **14.9%** 的组被保留；50% 时约 **99.997%**。这不是 MiMo 实测：真实轨迹可能相关，奖励也可能连续。

> **关键 insight：即使题库不变，实际课程也会变化。** 一道题可随能力提升从全失败进入混合组，再进入全成功组。过滤偏向当前能产生差异的题；GRS 让部分已过测题继续学习质量，始终全失败的题则可能仍需更强起点或示范。补齐来源数量不能自动恢复内部难度分布。

### 4.2 保持数据配比，需要不均等地分配资源

25 个来源的平均生成 token 数相差 90 倍、平均耗时相差 66 倍。Sample Mixer 按目标保留组数、过滤接受率、耗时和当前缺额调整并发，同时兼顾后续 batch，避免某来源收齐后完全断供。

固定收集速率下，并发需求大致随 **目标组数 × 耗时 ÷ 接受率** 增长。接受率是通过训练过滤的比例，不是任务成功率。例如两源各需 100 组，一源接受率 0.5、耗时 10 分钟，另一源为 1、1 分钟，则并发需求约 **20:1**（解释性示例）。

<details>
<summary>展开 Figure 16：兼顾长期目标与当前缺额的调度</summary>

<p align="center">
  <a href="assets/mimo-v2.6/figure-16-sample-mixer.png"><img src="assets/mimo-v2.6/figure-16-sample-mixer.png" alt="原报告 Figure 16：调度方式对收集进度与并发占用份额的影响" width="680" /></a>
</p>

*Figure 16，p.31。上排看收集进度，下排看占用份额；只追缺额容易切换，只追长期目标容易进度分化，组合策略更平稳。模拟未含训练、评分延迟、过期淘汰和回放，不能当成端到端加速。实际仅在启动 / 恢复首个收集步回放合格完整组。*

</details>

### 4.3 按题平均与组内公平比较处理不同偏置

**Prompt-mean** 先在同题轨迹组内平均 token 项，再在题目之间平均。两题组长分别为 1 万、10 万 token 时，外层名义权重由按 token 平均的约 1:10 变为 1:1（示例）。它不是每条轨迹等权，也不是长度惩罚；最终权重还受优势、概率比和 mask 影响。

报告列出的代码 68%、通用工具 12%、视觉 13%、上下文遵循 3%、安全 4% 未明确分母，不能换读成 token 或算力比例。**记录来源、保留率、长度和有效更新权重，比只报告题库比例更能解释训练收益。**

Harness 包括提示、工具接口、上下文管理和执行循环。MiMo 的 mini-harness 在**组内保持配置相同，组间引入多样性**，让相对优势主要反映行为差异。未见过的 Codex、Claude Code、mini-swe-agent 上，DeepSWE 平均 Pass@1 约从 50% 升至 66%；支持跨接口迁移，没有等预算单框架对照（§4.2.5、Figure 10）。

作者还指出，生产 harness 的保护规则和流程要求未必进入任务完成奖励，可能被优化中的模型忽略。**任务成功奖励不会自动教会全部产品约束**；所需规则仍应进入监督或由执行系统保证，不能从 benchmark 提升中推定已经学会。

## 5. 概率对齐与负载稳定性需要分别处理

**系统执行会进入学习信号。** 同一权重在不同引擎中可能走不同路径、得到不同采样概率；重要性采样应处理策略变化，而非执行口径差异。

来源：§5.4–5.5、§6.4、Figure 11。

### 5.1 概率比的分母必须来自实际生成

MiMo 按 token 用当前训练概率除以生成当时保存的概率。Partial rollout 的不同片段可能来自不同版本，不能事后用一个旧版本统一补算分母。生成侧 SGLang 与训练侧 Megatron-LM 还需三层对齐：

| 层次 | 处理 | 对齐对象 |
| --- | --- | --- |
| 权重 | QDQ（quantize–dequantize，量化再反量化）遵循 MXFP4 生成内核 | 实际数值 |
| 专家路径 | R3（Rollout Routing Replay，生成路由重放） | 生成时的专家选择 |
| 截断采样 | 重放 top-k / top-p 候选集并在其中归一化 | 实际采样分布 |

例如全词表概率为 0.2、保留候选集总概率为 0.8，实际采样概率为 0.25；训练侧直接以 0.2 / 0.25 得到 0.8，**策略没变，ratio 却变了**（示例）。只保存被选 token 的概率也不足以重建训练归一化分母。

这仍不是无限容忍旧数据的完整校正：逐 token ratio 只修正已到达历史下的动作概率，未修正历史本身的到达概率；越界 token 还会被屏蔽。公式见附录 B。

### 5.2 Router 漂移的系统代价有干预证据

<p align="center">
  <a href="assets/mimo-v2.6/figure-11-router-freezing.png"><img src="assets/mimo-v2.6/figure-11-router-freezing.png" alt="原报告 Figure 11：router 冻结与不冻结时的 Pro 第 9 层专家负载" width="600" /></a>
</p>

*Figure 11，p.24。三栏为负载变异系数、最大值 / 均值、冷专家占比；越高越不均衡。橙线可训练，蓝线冻结。*

Pro 第 9 层的 384 个专家，在 router 可训练的前 20 步，负载变异系数由 0.78 增至 2.0，峰值负载由均值的 6 倍增至 16 倍，冷专家占比由 0.5% 增至 22%；冻结时基本稳定。

**保留第 20 步其他参数，只恢复初始 router，负载接近恢复，评测性能基本不变。** 这一干预说明该窗口内的漂移没有可见能力收益，支持检验是否缩小更新范围；不能外推为所有 MoE 都应永久冻结。冻结参数也不固定专家选择，上游表示仍会变化。

### 5.3 训练策略本身会改变工作负载

Partial rollout 收齐一批后暂停长任务、更新再恢复，减少长尾等待，却需要 re-prefill 重算历史 KV，这部分没有新增探索。同一版本内可复用 KV，工具等待时可将状态移至主机内存；batch、并发与续跑策略需一起选择。

报告中的故障揭示了更深的约束：短轨迹先完成使长度估计偏低；同源不同 harness 长度相差超过两倍；全 batch 均衡仍出现局部专家 rank 接收均值 30 倍以上 token；Flash 后期轨迹增长导致单节点打包内存耗尽。

> **关键 insight：RL 负载由正在变化的策略生成，早期安全配置不是后期保证。** 容量要看局部峰值和未完成长任务，调度要跟随策略变化。推测解码也应看端到端吞吐：DFlash block-6 比 block-8 快约 6%，接受长度却变化很小（详见附录 A）。

## 6. MOPD2 同时管理教师能力与监督状态

**教师擅长完成任务，不代表它能在学生到达的所有历史上可靠指导。** 示范训练的教师尤其可能缺少多轮偏离后的状态覆盖，前缀蒸馏限制了这种偏移。

来源：§5.6、Figure 13、§6.2。

<p align="center">
  <a href="assets/mimo-v2.6/figure-13-mopd2.png"><img src="assets/mimo-v2.6/figure-13-mopd2.png" alt="原报告 Figure 13：两类领域教师、完整轨迹与前缀蒸馏" width="640" /></a>
</p>

*Figure 13，p.25。顶部是教师来源，左右是学生生成路径；SFT 数据在右侧提供历史上下文，续写由学生生成。*

| 路径 | 学生从哪里生成 | 监督来源 |
| --- | --- | --- |
| Standard MOPD | 从任务起点自主完成整条轨迹 | 可验证任务上的 RL 领域教师 |
| Teacher-Prefix OPD | 教师轨迹某轮前的完整历史，续写一轮 | RL 领域教师逐 token 指导 |
| SFT-Prefix OPD | 示范某轮前的完整历史，续写一轮 | 高质量合成示范训练的 SFT 教师逐 token 指导 |

含 $`k`$ 个 assistant turns 的源轨迹可拆成 $`k`$ 个前缀。预先分配的教师基于同一历史与学生已生成 token 提供监督，不要求照抄固定续写。

例如从“已读完仓库、准备修改”的前缀起跑，学生练习修改决策，却没有练习如何到达这一步（示例）。**这里的 on-policy 只覆盖新续写，前缀历史来自外部。**

> **关键 insight：前缀也是监督资源，决定在哪些状态上相信教师。** 可靠历史减少重复执行和教师失准，完整 rollout 保留自主到达、规划与连续纠错；局部监督可靠性与长程自主性需要分别衡量。

报告未给出前缀路径独立消融，亦未完整披露教师名单、学生初始化和损失配置。它将能力拓展指向游戏开发、科研、具身智能等难验证领域，但任务、评分器和教师仍依赖外部建设。**优化外部评价系统，与自主改进评价系统，是两个不同问题**；“走向自我改进”不等于递归闭环已经实现。

## 7. 开源资源揭示了环境、判卷知识和初始能力的价值

**复现 agent RL 需要任务、环境、评分标准、框架和强起点的组合。** 只拿题目与模型，遗漏的往往是最关键的监督条件。

来源：§7、Tables 4–7；以下数据审计另列口径。

### 7.1 示范与交互的增益依赖底座短板

MiMo-V2.6-Distill-Qwen-9B 在 MiMo 数据上对 Qwen3.5-9B 做 SFT，使用 77.4B total tokens、27.2B loss tokens；不能直接等同于 MOPD2。

| Benchmark | Qwen3.5-9B | Distill SFT | 后续领域 RL |
| --- | ---: | ---: | ---: |
| SWE-bench Pro，avg@3 | 32.0 | 44.6 | 47.6 |
| MiMo Cyber Bench mini，avg@3 | 5.7 | 31.3 | 47.0 |
| AutomationBench，avg@1 | 5.0 | 30.3 | 33.1 |
| Terminal Bench 2.1，avg@1 | 27.0 | 37.1 | 52.8 |

RL 列是分别训练的领域检查点。AutomationBench 主要增益发生在 SFT，Terminal Bench 仍有较大 RL 增量。**由此得到的启发是先诊断底座：示范可以让有效行动变得可达，交互反馈再改善选择与纠错。** 这不是先 SFT / 先 RL 的受控比较。

### 7.2 评分标准也是领域知识，环境也是任务身份

[官方开源数据](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)有 **7,780 条任务**，并提供[训练代码](https://github.com/XiaomiMiMo/verl/tree/mimo-oss)和[环境镜像](https://hub.docker.com/r/xiaomimimo/mimo-v2.6-rl-oss)。本仓库固定版本分析发现：

* 925 条通用工作任务中，**94.4% 的评分正文比题面长，长度比中位数为 2.33 倍**。标准保存了排除失效记录、合并别名、处理资料冲突等领域判断；长度不保证质量，答案材料须与 agent 输入隔离。
* 5,125 个评分条目中，4,437 项标为模型评审，顶层却统一为 `rule`。**可验证不等于确定性程序评分**，复现还依赖 judge 版本和证据读取方式。
* 1,000 条安全记录对应 693 种题面、1,000 个镜像名。文本去重会漏掉环境差异；镜像不同也不能证明漏洞独立。

这是开源数据审计，不是内部训练统计：版本 `639865f`，长度按字符计算、排除 JSON 与验证代码，见[结果](data/MiMoV2.6-RL-oss-summary.json)和[脚本](../../scripts/analyze-mimo-rl.py)。开源代码对应小模型实验，默认 batch 64、每题 16 条、256K 上限、工具错误惩罚关闭，不能当成 Pro / Flash 内部配方（[固定版本脚本](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/scripts/code/train.sh)，2026-09-29 核对）。

后续最值得验证的三件事是：**等训练预算下，探索与评分各增加多少有效信号；等解题预算下，质量改善是否仍成立；教师在自主历史与可靠前缀上的指导能力相差多少。** 它们分别对应信号效率、部署效率和长程自主性，报告尚未将这三者完整分解。

## 附录 A：架构与推测解码配置

<details>
<summary>展开：长轨迹成本、模型参数与多模态编码</summary>

**架构与 RL 的联系在单次尝试成本。** 长轨迹反复读取历史，同题又采样 16 次；MoE（Mixture of Experts，混合专家）减少每 token 激活参数，SWA（Sliding Window Attention，滑动窗口注意力）限制多数层的读取范围，少数 GA（Global Attention，全局注意力）层连接远处信息。节省单次尝试可为更多探索留下预算，但报告未隔离架构的 RL 增益。（来源：§2、Table 1、§6.4。）

| 配置 | Flash | Pro |
| --- | --- | --- |
| 总参数 / 每 token 激活参数 | 310B / 15B | 1.02T / 42B |
| 主干层数：局部 / 全局 | 39 / 9 | 60 / 10 |
| Hidden size | 4096 | 6144 |
| SWA heads：Q / KV | 64 / 8 | 128 / 8 |
| GA heads：Q / KV | 64 / 4 | 128 / 8 |
| QK / V head dimension | 192 / 128 | 192 / 128 |
| 专家总数 / 每 token 选择数 | 256 / 8 | 384 / 8 |
| 局部窗口 / 最终上下文长度 | 128 tokens / 1M | 128 tokens / 1M |

首个 block 为 GA 与 dense FFN，其余用稀疏 MoE，无共享专家；层数不含多 token 预测（Multi-Token Prediction，MTP）模块。序列分片时，SWA 仅交换窗口可达的 key / value，全局层仍处理全段上下文。激活参数量不等于同规模 dense 模型的实际成本。

**多模态编码同时影响能观察什么、历史有多长。** 681M 参数的 MiMo-ViT 有 24 层局部、4 层全局注意力，交替按行、按列组织局部窗口，再全局汇总；与小语言模型联合用理解任务训练，处理超过 4T image tokens，无额外对比学习目标。音频每秒 25 帧，每帧 20 个残差向量量化（Residual Vector Quantization，RVQ）码的 embedding 先相加，再将 4 帧合成一个主干输入，得到每秒 6.25 个位置。按此估算，60 秒音频约占 375 个主干位置；不是每帧的 20 个码各占一个位置，也不意味着整条编码链路等比例省算力。

预训练先文本、后多模态：Flash 为 26T + 22T tokens，Pro 为 27T + 3T。较小模型看过更多数据，不能只按参数量比较。上下文在预训练从 32K 扩至 256K，再由 Mid-training 扩至 1M。Muown 机制见[原论文](https://arxiv.org/abs/2605.10797)。（来源：§2–3。）

**DFlash 配置应按端到端吞吐选择。** 小模型提出候选 token，主干并行验证。先在 SFT 策略上训练，再用匹配 RL 任务配比的早期轨迹适配，RL 默认 block-6。（来源：§6.4。）

| 比较 | 报告结果 |
| --- | --- |
| block-6 相比原 MTP 配置 | 平均接受长度高 31.3% |
| 混合 RL 中 block-6 相比 block-8 | 全局平均吞吐高约 6%，接受长度变化很小 |
| 所测长上下文负载下，RL 适配的 FP8 DFlash 相比基线 | 每节点吞吐高约 10.3% |

候选变多也增加主干验证工作，所以接受长度不等于吞吐。三项基线不同，不能相加。

</details>

## 附录 B：优化目标、优势调整与复现配置

### B.1 Prompt-mean 目标与逐 token 重要性采样

来源：§4.1 Eq. (1)、§5.1。

```math
\mathcal{L}(\theta)
= -\mathbb{E}_{q\sim\bigcup_d\mathcal D_d,\;\{o_i\}_{i=1}^{G}\sim\mu_{\theta_{\mathrm{old}}}(\cdot\mid q)}
\left[
\frac{1}{\sum_{i=1}^{G}\lvert o_i\rvert}
\sum_{i=1}^{G}\sum_{t=1}^{\lvert o_i\rvert}
r_{i,t}M_{i,t}A_i
\log\pi_\theta(o_{i,t}\mid q,o_{i,\lt t})
\right].
```

内层以同题组的总长度归一化，外层对题目与采样取平均。

| 符号 | 含义 |
| --- | --- |
| $`q`$、$`G`$ | 题目及每题轨迹数，本次 $`G=16`$ |
| $`o_{i,t}`$、$`o_{i,\lt t}`$ | 第 $`i`$ 条轨迹当前 token 及其之前的历史 |
| $`\pi_\theta`$、$`\mu_{\theta_{\mathrm{old}}}`$ | 当前训练策略、实际生成数据的策略 |
| $`A_i`$ | 轨迹优势；局部行为调整后替换为相应的逐 token 优势 |
| $`r_{i,t}`$ | 重要性采样权重 |
| $`M_{i,t}`$ | token mask，需结合模型输出的训练范围和行为规则理解 |

```math
r_{i,t}
=\mathop{\mathrm{sg}}\nolimits\left[
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,\lt t})}
{\mu_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,\lt t})}
\right].
```

分母为生成当时保存的概率，不为 partial rollout 事后重算；$`\mathop{\mathrm{sg}}\nolimits`$ 表示停止梯度，ratio 仅作权重，梯度经 log-probability 项传播。

正、负优势分别设置概率比上下界：

```math
M_{i,t}^{\mathrm{clip}}=
\mathbf{1}\left[
(A_i\geq0\land\epsilon_+^{l}\leq r_{i,t}\leq\epsilon_+^{h})
\lor
(A_i\lt0\land\epsilon_-^{l}\leq r_{i,t}\leq\epsilon_-^{h})
\right].
```

两组初始区间均为 **[0.2, 5.0]**。越界 token 被屏蔽，例如 ratio 为 6 时不参与该项更新，而不是截成 5；这不等同于标准 PPO 的 `min(unclipped, clipped)` 目标。熵过低时放宽正区间、收窄负区间，熵过高时反向调节，并分别监控屏蔽比例。报告未披露完整控制器，也未在该目标中写出参考模型 KL 惩罚或默认优势标准差归一化，不能按常见 GRPO 配方补齐。

### B.2 GAR 的重分配与中心化

来源：§4.3.2 Eq. (3)。设 hack 修正后的二元奖励为 $`R_i`$，组均值为 $`\bar R`$，初始优势为 $`A_i=R_i-\bar R`$，成功集合为 $`\mathcal P=\{i:R_i=1\}`$。质量因子 $`f_i\in(0,1]`$ 先降低较差成功解的权重，再以共同系数分配移出的正优势：

```math
\lambda=\frac{\sum_{j\in\mathcal P}A_j}
{\sum_{j\in\mathcal P}f_jA_j},
\qquad
A_i'=\begin{cases}
\lambda f_iA_i,&i\in\mathcal P,\\[0pt]
A_i,&i\notin\mathcal P.
\end{cases}
```

未限幅时保留成功轨迹的正优势总量及质量带来的相对权重。实际实现限制共同缩放系数，再从成功、失败轨迹的优势中都减去全组均值，使最终组均值为零；因此守恒不一定保留。不可用的 grader 输出退回原优势。该表达式适用于成败混合组，不能用于全成功零优势组制造新差异。

### B.3 同题成功解参照的长度惩罚

来源：§4.3.3 Eq. (4)。设成功集合为 $`\mathcal P_q`$，当通过率 $`\lvert\mathcal P_q\rvert/G\gt a_{\min}`$ 时，取成功生成长度的指定分位数：

```math
\ell_q^\star=\mathop{\mathrm{Quantile}}\nolimits_{B/100}
\{\ell_j:j\in\mathcal P_q\}.
```

```math
\widetilde R_i=R_i-\mathbf{1}[i\in\mathcal P_q]X
\left[
\mathop{\mathrm{clip}}\nolimits\left(
\frac{\ell_i/\ell_q^\star-1-\delta}{s-\delta},0,1
\right)
\right]^\gamma.
```

$`\ell_i`$ 为生成 token 数；$`B`$ 决定分位数；$`X`$ 为最大扣分；$`\delta`$ 是允许超出参考长度的比例；$`s`$ 决定何时扣满；$`\gamma`$ 控制曲线。其中 $`s\gt\delta\geq0`$、$`X\geq0`$、$`\gamma\geq1`$。本文将原文通过率阈值 $`A`$ 改记为 $`a_{\min}`$，避免与优势混淆。其他组保留原奖励，调整后的奖励用于计算优势。

示例（非报告配置）：参考 10,000 tokens，容忍 20%，超过 12,000 开始扣分；若 $`s=1`$，20,000 时扣满 $`X`$。报告未披露完整超参数。

### B.4 局部错误 token 的优势调整

来源：§4.3.3 Eq. (5)。令 $`h_{i,t}=1`$ 表示被规则标记，否则为 0：

```math
\widetilde A_{i,t}=\begin{cases}
\alpha(1-h_{i,t})A_i,&A_i\gt0,\\[0pt]
[\beta(1-h_{i,t})+\kappa h_{i,t}]A_i,&A_i\lt0,\\[0pt]
0,&A_i=0,
\end{cases}
\qquad \kappa\gt1.
```

$`\kappa`$ 放大错误片段的负权重；$`\alpha`$ 补偿未标记的正优势 token；$`\beta`$ 减轻未标记的负优势 token。分支依据是优势正负，而非测试成败；质量或长度调整后，成功轨迹也未必具有正优势。

系数统计范围为**整个 training batch 中参与 loss 的 token**。设 $`H_+,H_-`$ 为正、负优势中命中规则的 token 集合，$`C_+,C_-`$ 为未命中的集合：

```math
\alpha=\min\left(\alpha_{\max},
1+\frac{\sum_{(i,t)\in H_+}A_i}{\sum_{(i,t)\in C_+}A_i}\right),
\qquad
\beta=\max\left(\beta_{\min},
1-\frac{(\kappa-1)\sum_{(i,t)\in H_-}\lvert A_i\rvert}
{\sum_{(i,t)\in C_-}\lvert A_i\rvert}\right).
```

$`\alpha_{\max}\geq1`$ 限制正向放大，$`0\lt\beta_{\min}\leq1`$ 限制负权重减轻的程度。分母为零时相应系数设为 1。只有分母非零、未触发限制时，正负两侧的优势总量分别守恒；不保证最终梯度或策略熵不变。

### B.5 截断候选集内的概率

来源：§6.4。下式是对报告归一化机制的数学表达。生成时保留候选集为 $`S_t`$，训练用同一集合归一化：

```math
\pi_\theta^{S_t}(y\mid h_t)=
\frac{\pi_\theta(y\mid h_t)}{\sum_{v\in S_t}\pi_\theta(v\mid h_t)},
\qquad y\in S_t.
```

$`h_t`$ 为当前历史，$`y`$ 为采样 token。实现先用全词表固定形状位图从 GPU 传到 CPU，避免变长同步或候选集截断，后续使用稀疏存储；报告中典型 top-p=0.97 时平均候选数小于 5。

### B.6 优化器与初始化

来源：§5.1。

| 项目 | 报告配置 |
| --- | --- |
| 优化器 | Muown |
| Learning rate | $`3\times10^{-6}`$ |
| Weight decay / LR warmup | 均不使用 |
| Gradient clipping | 1.0 |
| Muon momentum | 0.95，启用 Nesterov |
| Newton–Schulz iterations | 10 |
| 额外 update scaling | 0.5 |
| Adam 部分 | $`\beta_1=\beta_2=0.95`$，$`\epsilon=10^{-8}`$ |
| RL 初始化 | 继承 SFT 的 FP32 master weights 与 Muown row state |
| MoE router | 冻结 |

## 附录 C：最终成绩与 RL 曲线的口径

<details>
<summary>展开：最终模型成绩与可比较范围</summary>

来源：§5.6、Table 3。以下是最终模型的部分成绩，经过多阶段训练；不能将相对旧版本的全部收益归因于 RL 或某个评分机制。

| Benchmark | V2.6 Pro | V2.6 Flash | V2.5 Pro | Claude Opus 5 | GPT-5.6 Sol |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSWE v1.1 | 71.9 | 67.9 | 19.0 | 74.0 | 73.0 |
| ProgramBench | 26.5 | 26.0 | 12.5 | 37.0 | 25.0 |
| MiMo Code Bench | 63.2 | 61.2 | 40.4 | 68.6 | 59.3 |
| AutomationBench v1.0.6 | 53.1 | 52.3 | 16.0 | 50.3 | 45.8 |
| Toolathlon-Verified | 76.9 | 73.6 | 49.1 | 80.6 | 74.9 |
| Terminal Bench 4.0 | 34.9 | 28.8 | 1.5 | 49.0 | 39.9 |
| OSWorld-Verified | 82.0 | 80.8 | — | 83.4 | 83.0 |
| CyberGym（作者修正环境） | 94.0 | 95.1 | 40.0 | — | — |
| ExploitGym | 17.8 | 6.0 | 0.2 | 22.1 | 30.3 |
| MiMo Visual Coding | 72.3 | 71.5 | — | 70.0 | 73.4 |

Pro 在 AutomationBench 领先表中基线，但 Terminal Bench 4.0 仍有明显差距；漏洞复现接近 95%，也不能外推到漏洞利用。可配置推理强度的基线均使用最高档位，不等于成本相同，也缺少统一预算与统计不确定性。

最终 DeepSWE 71.9 / 67.9 与 RL 曲线终点 72.6 / 65.7 不是受控蒸馏对照，不能相减估算 MOPD2 收益。

</details>

图表均截取自原报告，保留原始曲线与标签；PDF 页码、裁剪范围和来源文件 SHA-256 记录见[图片来源](assets/mimo-v2.6/source.json)。
