# RL 训练经验与实践

> 本文档系统整理 LLM RL 与 Agentic RL 的训练稳定性、工程实践、训推一致、长 trajectory 调参等可复现 know-how。
> 重点是**调参细节**与**工程踩坑**，而非纯算法推导。调研截至 **2026/06**。

> **如果你只读一节,读下面这节"写在最前面"**。它是 Part I–V 全部工作的横向提炼,每条共识都有具体出处与数字。

---

## 目录

### 速读区(无论时间多紧都建议看)

- **[写在最前面:工程实战 15 条共识](#写在最前面工程实战-15-条共识)** — 跨 50+ 工作的横向提炼,带具体出处与数字
  - 共识 1 — Tool/observation output token 一律不计 loss、不计 IS 比率
  - 共识 2 — 几乎所有现代 reasoning RL 都移除 KL 项
  - 共识 3 — Clip-Higher 已成标配($`\epsilon_{\text{low}}=0.2, \epsilon_{\text{high}}=0.28`$)
  - 共识 4 — Outcome-only reward 是主流且足够强
  - 共识 5 — 长度归一化 / overlong filtering 是反共识陷阱
  - 共识 6 — Dynamic sampling / zero-advantage filtering 是 batch 利用率核心
  - 共识 7 — Actor lr 极保守 ≈ 1e-6;SFT lr ≈ 5e-6–7e-6
  - 共识 8 — 冷启动 SFT 视任务而定(三派分明)
  - 共识 9 — 多阶段长度课程是普遍范式
  - 共识 10 — 长 trajectory 必须 partial rollout / 异步训练
  - 共识 11 — 环境基础设施成为核心差异化
  - 共识 12 — Entropy collapse 是头号杀手,治法多样
  - 共识 13 — 判分两域分治:可验证用 RLVR、不可验证才上 GRM
  - 共识 14 — IS 粒度之争:token 级(+TIS) / 序列级(GSPO) / 裁权重不丢 token(CISPO)【2025H2 新浪潮】
  - 共识 15 — 多轮 agentic RL 有独立于单轮的崩溃模式(void turn / echo trap / entropy 失控 / template collapse)【最重要的新坑】
- **[训练仪表盘:Agentic RL 必看监控指标](#训练仪表盘agentic-rl-必看监控指标)** — 训的时候盯什么,以及看到 X 先怀疑什么
  - 类 1 — 策略分布健康度(头号 dashboard)
  - 类 2 — 训推一致性指标(系统层 bug 探测器)
  - 类 3 — Reward 与 Advantage 健康度
  - 类 4 — Trajectory 形态(长度 / 重复 / 截断)
  - 类 5 — Agentic 专属指标(multi-turn / tool / web)
  - 类 6 — 梯度与优化
  - 类 7 — Eval 指标(防 train reward 涨、eval 不涨)
  - 类 8 — 系统 / Infra
  - 诊断速查表 — 看到 X 先怀疑什么

### 详细章节

- **Part I. 通用 LLM RL 工程基础**(非 agentic 但必读)
  - §1 Stabilizing RL with LLMs / MiniRL (Qwen Team)
  - §2 DAPO (ByteDance Seed)
  - §3 Magistral (Mistral)
  - §4 MiMo / MiMo-7B-RL (Xiaomi)
  - §5 Skywork-OR1 / MAGIC (Skywork)
  - §6 DeepSeek-R1 (DeepSeek)
  - §7 Qwen3 (Alibaba)
  - §8 GLM-4.5 / GLM-4.6 (Z.ai)
  - §9 Seed-Thinking-v1.5 (ByteDance)
  - §10 Llama 3 / Llama 4 RL 部分
  - **【2025H2–2026 新增:新算法浪潮】**
  - §10A GSPO 序列级策略优化 (Qwen) — 免 Routing Replay 的 MoE RL
  - §10B CISPO + MiniMax-M1 — 裁权重不丢 token + FP32 LM head 精度坑
  - §10C GMPO 几何平均策略优化
  - §10D Dr.GRPO — R1-Zero 式训练的两个偏置(共识 5 的理论基础)
  - §10E Lite-PPO — "Tricks or Traps?" 极简两件套反超 DAPO
  - §10F ScaleRL — "The Art of Scaling RL Compute"(Meta,RL 的 scaling law)
  - §10G ProRL — 长时 RL 能否扩展 base 能力边界(对共识 12 的挑战)
  - §10H Open-Reasoner-Zero — 极简 vanilla PPO base RL
  - §10I Kimi K1.5 — 在线镜像下降 + L2 软信赖域(K2 算法之源)
  - §10J 小模型 RL 配方:DeepScaleR / Polaris / Light-R1
  - §10K Llama-Nemotron — FP8 生成 + 课程
  - §10L 熵的动力学:崩溃定律 + 高熵 minority token(共识 12 深化)
- **Part II. 多轮工具 / 搜索 Agentic RL**
  - §11 Search-R1 (UIUC)
  - §12 R1-Searcher / R1-Searcher++ (RUC)
  - §13 ReTool (ByteDance Seed)
  - §14 ToRL (GAIR-NLP)
  - §15 Tool-Star (RUC-NLPIR)
  - §16 ARTIST (Microsoft Research)
  - §17 rStar2-Agent (Microsoft Research)
  - **【新增:多轮崩溃机理 + 信用分配 — 训练经验金矿】**
  - §17A SimpleTIR — "Void Turn" 致梯度爆炸(最干净的崩溃闭环)
  - §17B RAGEN / StarPO — "Echo Trap" + reward std 提前预警
  - §17C EPO / RAGEN-2 — 熵失控 + "Template Collapse"(熵看不见的崩溃)
  - §17D GiGPO / Tree-GRPO — 长程 agent 的两级/树形信用分配
  - §17E ARPO — 工具返回后熵尖刺 → 熵驱动分支采样
  - §17F 高熵 token 机制 + Spurious Rewards(RLVR 评估陷阱)
- **Part III. 深度研究 / Web Agentic RL**
  - §18 Kimi-Researcher (Moonshot)
  - §19 Kimi K2 Technical Report (Moonshot)
  - §20 ASearcher (Tsinghua IIIS × Ant)
  - §21 WebDancer (Alibaba Tongyi)
  - §22 WebSailor / WebSailor-V2 (Alibaba Tongyi)
  - §23 WebShaper / WebWeaver (Alibaba Tongyi)
  - §24 Tongyi DeepResearch 汇总报告
- **Part IV. SWE Agentic RL**
  - §25 DeepSWE (Agentica × Together)
  - §26 SWE-RL / SSR (Meta FAIR)
  - §27 SWE-Gym (UC Berkeley)
  - §28 R2E-Gym (UC Berkeley)
  - §29 Nebius Long-Context Multi-Turn (Nebius)
  - §30 Qwen3-Coder (Alibaba)
  - §31 Devstral / OpenHands LM
  - §32 Cursor Composer 系列
  - §33 Anthropic Claude 4 / 4.5 (公开信息有限)
  - **【新增:环境构造 / 自博弈 reward / 前沿模型 — 重点关注代码】**
  - §33A SWE 环境自动构造流水线:SWE-smith / SWE-rebench / SWE-Flow / SWE-Dev / SWE-Fixer
  - §33B 开源 SWE 模型:Kimi-Dev / Skywork-SWE(数据 scaling law)/ Lingma SWE-GPT / Satori-SWE
  - §33C 自博弈 / 共进化 reward:CURE / Absolute Zero / AceCoder / rStar-Coder(代码 RL 的 reward hacking 重灾区)
  - §33D 前沿模型 agentic coding:DeepSeek-V3.2(GRPO 四件套)/ MiniMax-M2 / GLM-4.6 / Seed-Coder
- **Part V. Agentic RL 训练框架与基础设施**
  - §34 verl / HybridFlow (ByteDance Seed)
  - §35 AReaL (Ant × Tsinghua)
  - §36 Agent Lightning (Microsoft)
  - §37 OpenRLHF
  - §38 SkyRL / SkyRL-Agent (UC Berkeley Sky)
  - §39 rLLM (Berkeley Agentica)
  - §40 Tinker (Thinking Machines)
  - §41 AgentScaler / AgentRL / RollArt
  - **【新增:异步/staleness 记账 + 去中心化 RL】**
  - §41A ROLL / ROLL Flash (Alibaba) — per-sample staleness α
  - §41B slime (THUDM) 深入 — TIS/MIS + catastrophic-token veto
  - §41C NeMo-RL (NVIDIA) — 端到端 FP8 统一训推精度
  - §41D prime-rl + verifiers — gym-like 环境抽象 + 去中心化异步
  - §41E INTELLECT-2 / INTELLECT-3 — 全球去中心化异步(双侧 clip,async-8 实测 GSPO 崩)
  - §41F Trinity-RFT (Alibaba) / TRL (HF) / AReaL-lite
- **Part VI. 横向综合 take-home**
  - §42 跨工作对比速查表(reasoning / tool / web / SWE 四类横向超参对照)
  - §43 共识 → 出处反查表(15 条共识 × 支持工作 / 反例)
  - §44 训推一致性(隐形杀手)— 物理来源 + 对策 + TIS 争议 + 排查决策树
  - §45 算力规模参考
  - §46 抄作业指南(按 recipe / 环境 / 算法 / 部署 四类的推荐清单)
  - §47 奖励模型路线:offline GRM vs online GRM vs actor–GRM 合并(DeepSeek-V4)— 不可验证域的判分选型 + 推荐决策树
  - §48 实战教训精粹(practitioner lessons,带出处的非显然 know-how)
  - §49 PG 算法家族谱系 + 选型决策树(token/序列级 IS、clip、IS 修正怎么选)
  - §50 中文社区 + 框架操作层实战排障手册(practitioner lessons II)— verl/Megatron/OpenRLHF 框架坑 + FP16 修 mismatch + staleness 正负不对称 + SFT 起点致崩

---

### 阅读建议(按时间预算)

- **5 分钟**:速读区 15 条共识(只看每条粗体标题;时间更紧就先看新增的共识 14、15)
- **30 分钟**:速读区全部(13 共识 + 8 类监控指标 + 诊断速查表)+ 各 Part 开头的"本部分要点"
- **3 小时**:再加 §42 速查表 + §43 反查表 + §44 训推一致性 + 你最关心的 1–2 个 Part 的具体工作
- **抄 recipe**:直接看 §46 抄作业指南挑工作,然后跳到对应 §x 章节

---

# 写在最前面:工程实战 15 条共识

> 下面 15 条来自跨 50+ 工作的横向对比(详见 Part I–V)。每条都给出:**机制(为什么)、证据(谁这么做、关键数字)、反例/边界(什么时候不这么做)、如何抄**。
>
> 全文出现的 §x 都是后文章节锚点,可直接跳。
>
> **版本说明**:共识 1–13 是文档主体(reasoning RL 时代沉淀的稳定共识);**共识 14、15 是 2025 下半年–2026 的新浪潮**——14 讲 GSPO/CISPO 掀起的 "IS 粒度之争",15 讲多轮 agentic RL 独有的崩溃模式。这两条是本轮更新最该先读的部分。

## 共识 1 — Tool/observation output token 一律不计 loss、不计 IS 比率

**机制**:工具输出是 deterministic 环境信号,**不是 $`\pi_\theta`$ 采样出来的 token**。计入 loss 等于让模型背诵环境字面输出 → 梯度被环境固定 token 稀释、policy 学坏、entropy 漂移;IS 比率分母也无意义($`\pi_\theta`$ 对环境 token 的"概率"是个 garbage 数)。

**直觉**:把 trajectory 想成一段对话剧本——LLM 是演员,工具/环境是另一个演员。你只能教自家演员"怎么演",不能让他学着背别人的台词。一旦把对方台词也算进 loss,等于强迫演员去预测对手会说什么具体单词;这些 token 数量往往远大于自家台词(一个 search 调用回来几千 token,模型自己只生成几十 token 的 query),**真正想学的那点梯度信号被环境字面输出彻底淹没**。IS 比率同理: $`\pi_\theta(\text{环境 token})`$ 是个根本没意义的数,放进分母就是噪声放大器。

**证据(全员共识)**:
- **Search-R1** (§11):`<information>...</information>` 间 token 加 indicator mask;Table 4 ablation 显示去 mask 性能掉显著
- **R1-Searcher** (§12):`<|begin_of_documents|>...<|end_of_documents|>` 标签间 token 不参与 loss 和 IS
- **ReTool** (§13):`<interpreter>...</interpreter>` sandbox 输出全部从 loss mask
- **ToRL** (§14):`OBSERVATION` mask;论文明说"防记忆 deterministic 输出"
- **WebDancer** (§21):RL 阶段 tool response 进 context 参与 $`\pi_{\theta_{\text{old}}}`$ 计算,但**只对模型生成 token 做优化**
- **WebSailor / Nebius** (§22/§29):observation 在 loss 中 mask 是隐含约定
- **ARTIST** (§16):"gradient 只通过 model-generated token"
- **Agent Lightning** (§36):用 **transition 抽取**(只存 LLM action)等价实现,工具输出本就不在 output 里,无需 mask

**如何抄**:loss 上挂 $`\mathbb{1}[y_t\text{ 来自 LLM 生成}]`$;IS 比率同样按这个 indicator 算。

任何接 vLLM/SGLang 的 agent RL 必须做一次 token-id 对齐(见 §36 "No More Retokenization Drift" 踩坑)。

---

## 共识 2 — 几乎所有现代 reasoning RL 都移除 KL 项

**机制**:RL 训出来的策略本来就要大幅偏离 SFT/ref model(几千几万步训练),KL 罚相当于"不让进步";同时多算一遍 ref forward 是纯算力浪费。

**直觉**:KL 项的历史使命是 RLHF 时代——那时怕 PPO 把语言模型"洗"出非人话(reward hacking 一发不可收),用 ref model 当"安全绳"拉回来。但 reasoning RL 的目标恰恰是**让模型学会 ref model 不会的 long-CoT 推理**,留着这根绳就是自己绊自己。算力账更直接:每 step 多一遍 ref forward = 多 25%+ 的算力,换来的是"训练越久,惩罚越大"——本末倒置。所以**reward 可信(verifier 干净的 math/code)就能去 KL;reward 可疑(open-ended chat)才要 KL 兜底**。

**证据 — 移除 KL($`\beta = 0`$)**:
- **DAPO** (§2): $`\beta_{\text{KL}} = 0`$
- **Magistral** (§3):明言"策略本就大幅偏离参考模型,保留 ref 是无意义的计算"
- **MiMo** (§4):明确删 KL 后训练更稳
- **Skywork-OR1** (§5):不用 KL loss、不用 advantage mask
- **ReTool** (§13): $`\beta_{\text{KL}} = 0.0`$
- **ToRL** (§14):"all experiments omit the KL loss"
- **rStar2** (§17):KL 和 entropy loss 一起去
- **DeepSWE** (§25):GRPO++ 六件套含 "No KL Loss"

**证据 — 保留 KL 的少数派(都是早期 / 多模态稳定优先路线)**:
- **DeepSeek-R1** (§6): $\beta = 0.001$ (R1-Zero 元祖配置)
- **Search-R1** (§11):默认 PPO 保留 $\beta = 0.001$
- **R1-Searcher** (§12):Qwen-2.5 用 0,Llama-3.1-Instruct 用 $`1\text{e-}4`$ (理由:Llama instruct 早期不稳)
- **Llama 4**:加 KL 罚项(Meta 路线)

**边界**:KL 对"想保留预训练能力 / 防风格漂移 / instruct 模型起步前几百步"有用,代价是 reasoning 提升受限。**经验法则**:reasoning RL 大胆去 KL;agent SFT cold-start 之后前 100 步可保留 $`1\text{e-}4`$ 兜底,稳了再关。

---

## 共识 3 — Clip-Higher 已成标配

**机制**:对称 clip 下,低概率 token(探索候选)被 ratio 上调时容易撞 $1+\epsilon$ 被压住(从 $p=0.01$ 到 $p=0.02$ ratio 就 2 了),而高概率 token 微调不触限——**净效果抑制探索、加剧 entropy collapse**。抬高 $`\epsilon_{\text{high}}`$ 给低概率 token 上行空间。

**直觉**:对称 clip 看起来"公平"——上下都允许 ±20% 浮动——但它是**乘法约束**,作用到概率上完全不公平。算一笔账:

| token 类型 | 旧概率 | 想涨到 | ratio | 对称 clip(0.2)结果 |
|---|---|---|---|---|
| 探索 token | 0.01 | 0.02 | **2.0** | 撞 1.2 上限,梯度被压 |
| 探索 token | 0.05 | 0.07 | 1.4 | 撞 1.2,被压 |
| 高频 token | 0.50 | 0.55 | 1.1 | 通过 |
| 高频 token | 0.80 | 0.82 | 1.025 | 完全无感 |

低概率 token 想"翻身"必须经过 ratio > 1.2 这道坎——你越要鼓励的探索行为,clip 卡得越死。**结果就是 policy 只敢微调头部 token,长尾被默默掐死,entropy 一路下行**。Clip-Higher = 给低概率 token 留个"上升通道"。

**为什么只抬上界、下界 $`\epsilon_{\text{low}}`$ 保持 0.2**:上界 $`1+\epsilon_{\text{high}}`$ 在 $`A\gt 0`$ 时生效(管能多快抬概率),下界 $`1-\epsilon_{\text{low}}`$ 在 $`A\lt 0`$ 时生效(管能多快压概率)。抬高上界给低概率 token 上升空间、鼓励探索;而**抬高下界只会让负 advantage 的低概率 token 被压得更狠更快,直奔 0 再不回头**——同样加剧 entropy collapse。所以上界放松给"活路"、下界留紧当"刹车",两边都为保住探索,故不对称。

**证据 — 经典数值就是 $`\epsilon_{\text{low}}=0.2`$, $`\epsilon_{\text{high}}=0.28`$**:
- **DAPO** (§2):0.2 / 0.28(首倡)
- **rStar2** (§17):0.2 / 0.28
- **Nebius** (§29):Stage 1 [0.2, 0.3],Stage 2 [0.2, 0.26](随 context 增加收紧上界)
- **Magistral** (§3): $`\epsilon_{\text{high}} \in [0.26, 0.28]`$ **动态调整以维持 group entropy 稳定**——更精细的做法
- **MiniRL** (§1):0.2 / 0.27
- **MiMo** (§4):抬高 $`\epsilon_{\text{high}}`$ (具体值未披露)
- **Skywork-OR1** (§5):用 clip-higher 控熵

**反例**:
- **DeepSeek-R1** (§6): $\epsilon = 10$ **几乎不裁剪**(走"靠 KL+大 lr 自然约束"路线)
- **Search-R1** (§11):保留对称 $\epsilon = 0.2$ (PPO 经典)

**如何抄**:无脑用 **0.2 / 0.28**;长 context 后期 SWE/Web 可收紧到 0.2 / 0.26;Magistral 风格的动态调参属于锦上添花。

---

## 共识 4 — Outcome-only reward 是主流且足够强

**机制**:reward 越简单越难被 hack。任何中间 shaping(format 分、tool-success 分、step 分)都给模型"通过非答题手段拿分"的可能。在 verifier 干净的领域(math/code/SWE-test)稀疏不是问题——GRPO 用 group baseline 已经缓解了 credit assignment。

**直觉**:reward shaping 像考试给"过程分"——出发点是好的(鼓励学生写步骤),但学生很快摸出捷径:**写一堆看起来在思考的废话拿步骤分,最后不答题反而不会被扣"答错分"**。Search-R1 实测的 "answer avoidance"(F1 reward 下模型学会不答以避免错答)就是教科书例子。Outcome-only 反过来是"只看答案对不对"——简单粗暴,但**任何作弊路径都会反映在最终对错上**,无处遁形。"GRPO group baseline 已经缓解了 credit assignment"的意思是:组内 16 个 rollout 共享同一题,reward 全 1 或全 0 我没法学,但**只要组内有对有错,GRPO 自动告诉模型"这条轨迹比平均好/差"**,稀疏 outcome 也能传梯度。

**证据 — 纯 outcome**:
- **ReTool** (§13): $\pm 1$ outcome,**"没有 format reward、没有 code-error penalty——刻意简化以避免 hacking"**
- **ToRL** (§14): $\pm 1$ outcome;**Code Executability Reward (−0.5) 实验过无提升**
- **rStar2** (§17):答案 0/1,论文 §4.3.1 **明确反对** step-level reward 和 tool-error penalty,理由是 reward hacking
- **Search-R1** (§11):纯 EM,无 format reward、无中间 retrieval reward
- **Search-R1 Empirical Study (2505.15117)**:**intermediate retrieval reward 基本无用甚至有害**(reward hacking)
- **DeepSWE** (§25):稀疏 outcome(Pass2Pass + Fail2Pass 二值)
- **Kimi-Researcher** (§18):format + correctness + γ-decay,无中间 shaping
- **Tongyi DeepResearch** (§24):binary correctness reward

**例外 — 谨慎用 shaping 的**:
- **Tool-Star** (§15): $`r_M = 0.1`$ 显式 multi-tool 协作 bonus(强制 search+python 同时出现);**前提是单工具 outcome 学不出协作**
- **Magistral** (§3):四维 reward(format/correctness/length/language consistency)——但 correctness 仍是主体,其他三个是辅助约束
- **R1-Searcher** (§12):两阶段 reward,stage 1 教调用(retrieval bonus 0.5)、stage 2 切到纯 F1
- **Kimi-Researcher 的 γ-decay**: $`r_{\text{step}} = \gamma^{s-1} R`$ 隐式 length penalty,鼓励短轨

**经典坑**:Search-R1 实证 "F1 reward 会出现 **answer avoidance**"——模型学会不答以避免错答,需加 action-level penalty。

**边界(本条只管 verifiable 域)**:outcome-only 的前提是**有干净 verifier**。一旦任务 hard-to-verify(开放写作 / 研究综合 / 主观质量 / long-horizon agent),规则 reward 写不出来,就只能上**生成式奖励模型(GRM)**——此时 offline / online / actor–GRM 合并怎么选,见 **§47**。

---

## 共识 5 — 长度归一化 / overlong filtering 是反共识陷阱

> **别当绝对律——先分清两层:**
> - **reward(信号层)**:想控长度,就在这里加——软惩罚 / 超 100k 扣分 / 上限。
> - **loss 归一(聚合层)**:要不要除以 $`|o_i|`$。答案几乎总是**别除**,用 token-level(每 token 等权);除以长度只会稀释长样本梯度、引入偏置。
>
> 所以"reward 里已经罚了长度,还要不要 length-norm?"——**不要**。reward 已经把"太长"扣进 advantage 了,再除 $`|o_i|`$ 会把这份惩罚**稀释掉**,自己抵消自己。**长度控制永远放 reward,不靠 loss 归一。**
>
> **底层道理**:LLM 本质是 **per-token 学习**的——自回归模型每个 token 都是一次条件在上文的独立决策,RL 的梯度也天然落在每个 token 上,token-level(每 token 等权)正顺着这个粒度。而 len norm 给每个 token 的信号除以"它碰巧所在轨迹的长度",**把一个 token 的学习强度绑死在一个与它自身对错无关的序列级量($`|o_i|`$)上**——等于打破了 per-token 学习的本意,偏置正源于此。
>
> **两个边界**:① overlong filtering 只在被扔轨迹**没有可信 reward**时该 mask(DeepSWE 外生截断、结果未定);能判负就保留负 reward。② DPO 是特例——它要 normalize 的是**随长度涨的 reward 本身**(SimPO),不是这里的 loss 归一。

**机制 1(length norm 破一阶近似)**:MiniRL §1.2 推导。一句话——**序列目标的梯度天然是"各 token 梯度求和",除以长度把它变成"求平均",就把长样本压下去了**。

- 序列的对数概率 $`\log\pi_\theta(y)=\sum_t\log\pi_\theta(y_t)`$,所以梯度是把每个 token 的梯度**加起来**。长序列 token 多、本就该贡献更多梯度——每个 token 都在出力,这是对的。
- 除以 $|y|$ 后变成**平均**:不管 100 token 还是 4000 token,整条序列贡献都被压成 O(1)。长序列每个 token 的信号被 $1/|y|$ 稀释,**长样本被系统性低估** → bias。
- "破一阶近似"是同一件事的形式化说法:PPO 的序列 ratio 是连乘 $`\prod_t r_t`$,token-level 用 $`\sum_t(r_t-1)`$ 做它的一阶近似($`\prod_t r_t\approx 1+\sum_t(r_t-1)`$);除以长度后求和变平均,近似的就不再是序列 ratio。

**机制 2(overlong filtering 反向)**:扔掉超长 trajectory 等于撤销负反馈 → 模型继续生成 overlong pattern 无人纠正 → overlong 比例**反而上升**。

**直觉**:这两条都是"看起来善意,实则反向"的经典陷阱。

- **length norm(为什么"除以长度"是错的)**:除以 $|y|$ 看上去"公平"——长短样本权重相等——但有两条独立理由说明它是 bias 而非 unbias:
  - **理论侧(MiniRL §1.2)**:token-level loss $`\sum_t \log\pi_\theta(y_t)\cdot A`$ 是 sequence-level 真目标 $`\log\pi_\theta(y)\cdot A`$ 的**一阶近似**——前提是不除 $|y|$ 。一除,梯度方向不再对齐真序列目标,数学上就 biased 了。
  - **实证侧(DAPO)**:同样是低质量 trajectory,**长劣样本被欠惩罚**——一条 4000 token 的胡言乱语 trajectory,sample-level 归一化下 4000 个 bad token 平摊一份负梯度,每个 token 几乎不痛;一条 100 token 的劣质短样本,每个 bad token 承担 40× 惩罚。结果是**短劣 pattern 被快速纠正,长劣 pattern 学不掉**,语料里长劣 pattern 比例越训越高,行为上就表现为"模型越训越啰嗦"。
  - 注意:这里关键是**惩罚被稀释**,不是"奖励被偷拿"——长样本若 reward 正,每 token 奖励信号同样被稀释,只是这种正向稀释通常没坏处。
- **overlong filtering**:模型写了 32K 还没答完(显然有问题),你把它扔了不算 loss——等于告诉模型"写超长没有惩罚"。下一轮模型继续 32K,你继续扔——**负反馈通道彻底断了**。正确做法是给它**软惩罚**(超 4K 开始线性扣分),或者**保留 truncated 的负 reward**(rStar2 实测)。

一句话:**RL 的负反馈不能稀释、不能扔,凡是想"过滤掉麻烦样本让 loss 干净"的操作,99% 是在制造问题**。

**两个常见追问**:

- **Q:不 len norm,长轨迹 token 多、岂不是被它主导优化?** 分两种"主导":**per-token** 看——每个 token 权重完全相等,长轨迹不占便宜;**per-sample** 看——长轨迹总梯度确实更大,但这是**对的**(它真的有更多决策要学),且**不诱导更长输出**(方向由 advantage 符号管)。诱导更长的恰恰是 sample-level。唯一真问题是**方差**:少数超长 rollout 可能撑爆单步 batch → 用 Magistral 的"group 内总长度"归一或长度上限兜底。
- **Q:为什么 len norm 反而让模型越写越长?原理和前提?** **前提**:advantage 是整条轨迹一个数(outcome reward)、轨迹内每 token 共享;且模型学的是**跨轨迹共享的 per-token 习惯**。**原理**:同一个"啰嗦 token"在 4000-token 错答里被罚 $\hat A/4000$、在 100-token 错答里被罚 $\hat A/100$(强 40×),模型把这些信号平均后,**长答里的坏习惯被欠纠正、改不掉**;加上短错答被狠罚很快消失、长错答留存,错答分布整体往长偏移。**破直觉点**:len norm 在"单条轨迹"层面确实公平(每条总 log-prob 都移动 $\hat A$),但行为由 per-token 习惯决定,在那一层长轨迹每 token 被打了 $1/|y|$ 折扣——**公平按轨迹算、有偏按 token 算,而决定长度的是后者**。

**易混点 — 为什么这和 DPO"加长度正则"不矛盾**:DPO 文献明明 claim 要 length-normalize 防啰嗦(SimPO / R-DPO),这里却说别 normalize,看似打架,其实是同一原则在两种结构下的相反表现。关键在**长度从哪里进入目标函数**:

- **常见误解**:"不 norm → 长序列梯度 mass 大 → 模型偏向长输出"。漏了 **advantage 符号**:梯度 mass 大,但方向跟着 $A$ 走——长的好序列被更强**强化**、长的坏序列被更狠**惩罚**,不是一律偏长。
- **RL 里 verbosity 的真凶恰恰是 len norm 本身**(Dr. GRPO 实证:sample-level norm 会让**错误回答越训越长**)。机制:per-token advantage = $`\hat A/|o_i|`$,除以长度对**正负 advantage 作用相反**,四象限全被扭曲——

  | | 简洁(短) | 啰嗦(长) | 净推力 |
  |---|---|---|---|
  | **对** ($`\hat A\gt 0`$) | 每 token 奖励大 | 每 token 奖励小 | 对答被压**短**(砍掉有用推理) |
  | **错** ($`\hat A\lt 0`$) | 每 token 惩罚大 | 每 token 惩罚小 | 错答被拉**长**(藏进低惩罚区) |

  即 len norm **同时**"该长的压短、该短的拉长"。文献 spotlight"错答变长",是因为它病态可见、且训练期错答样本多主导整体趋势;但"对答被压短"同样是 bias。去掉 norm(token-level 每 token 等权)两头都修。
- **DPO 相反,因为长度成了 reward 里未被约束的自由变量**:DPO 的 implicit reward $`r(y)=\beta\sum_t\log\frac{\pi(y_t)}{\pi_{\text{ref}}(y_t)}`$ 是**未归一化的 token 求和**——注意它本身**不**机械随长度涨(每项 log-ratio 可正可负,初始 $`\pi=\pi_{\text{ref}}`$ 时为 0),但它**不惩罚长度**。叠加偏好数据"chosen 普遍比 rejected 长"的标注偏置,模型就把"写更长"当成拉大 margin 的廉价杠杆(同样风格下 token 越多、正 log-ratio 项越多、总和越大),越训越啰嗦。**SimPO 除以 $|y|$、R-DPO 加 $-\alpha|y|$ 罚项**,都是把长度从 reward 里 disentangle。
- **GRPO 里长度不在 reward 里**:advantage 来自外部 reward(对错 0/1、F1 等),**与长度无关**。再除 $|y|$ 不是"消掉长度项",而是**凭空引入** $1/|y|$ 稀释长坏样本的惩罚。

| | 长度进入的位置 | 正确操作 |
|---|---|---|
| **DPO** | 长度**在 reward 里**(log-prob 求和随长度涨) | **normalize 掉**,去除 length confound |
| **GRPO** | 长度**不在 reward 里**(reward 外部、与长度无关) | **别 normalize**,否则凭空引入 $1/y$ |

**两边目标完全一致——都在杀 length bias**;只因长度进入目标函数的位置不同,正确手法刚好相反。

**证据 — 反 length norm**:
- **MiniRL** (§1.7):"不要 length-normalize:它破坏一阶近似,看起来无害但实际有 bias"
- **DAPO** (§2):用 token-level $`1/\sum_i|o_i|`$ 而非 sample-level $`(1/G)(1/|o_i|)`$,正是为了让长序列得到应得权重
  - sample-level:先在每条样本内除以自身长度 $`1/|o_i|`$(组内拉平),再除以样本数 $1/G$——结果是**每条样本权重相等**,长短不论,正是机制 1 说的"求平均"。
  - token-level:所有样本的 token 汇到一起,统一除以**总 token 数** $`\sum_i|o_i|`$——结果是**每个 token 权重相等**,长样本因 token 多自然占更大权重,梯度回到"求和"语义。
- **Skywork-OR1** (§5):token-level loss **without length normalization**
- **Magistral** (§3):用"group 内总长度"归一化(折中——既不破一阶近似,又防组内长度偏差)

**证据 — 反 overlong filtering**:
- **rStar2** (§17 失败案例 1):"扔掉超长 trajectory 反让 overlong 比例上升——无负反馈 → 重复 pattern 不纠正;**保留 truncated trajectory 的负 reward 更有效**"
- **DAPO**:提供 overlong filtering 变体,但主路用 **Overlong Reward Shaping**(4k cache 线性软惩罚 + 16k 上限)
- **Nebius** (§29):**明确反对** DeepSWE 的 Compact Filtering,改用 **soft length penalty**(超 $`L_{\text{thr}}`$ 线性惩罚到 $`T_{\max}`$)

**反例(DeepSWE)**:
- **DeepSWE** (§25):Compact Filtering 对 max-context/20min-timeout/max-steps trajectory mask loss——但配套是 64k context + 100 turns,**只对真正打不完的样本 mask**,不是普遍超长
  - **为什么这里 mask 而非罚负**:判据不是"长不长",而是**"这条轨迹有没有可信 reward"**。reasoning RL 的 overlong 是**模型内生啰嗦、且能判负**(答案错就是错)→ 必须保留负 reward 治病;DeepSWE 的截断是**外生预算耗尽、结果未定**(patch 没写完,Pass/Fail 无从判定)→ 强行判负 = 注入噪声标签,还会教模型"别碰需要长步数的解法、早收手"。有有效 reward → 保留并惩罚;没有(被预算砍断)→ mask。
  - **仍有争议**:Nebius(§29)明确反对,改用 soft length penalty——担心硬 mask 让模型学到"拖到截断就免罚"。DeepSWE 这套只在**宽预算 + 结尾才有 reward + 截断罕见**时安全。

**经验法则**:**永远保留 truncated trajectory 的负 reward**;真要约束长度,用软惩罚(DAPO 风格)而非硬 mask。

---

## 共识 6 — Dynamic sampling / zero-advantage filtering 是 batch 利用率核心

**机制**:GRPO group 内全对/全错时 $`\hat A_i = 0 \forall i`$,该 group 不产生任何梯度。训练后期"easy + impossible"占比走高,batch 有效样本急剧衰减,卡步数不卡 GPU。过滤 + 重采等于免费算力放大。

**直觉**:把 GRPO 当成"组内排名考试"——同一题让模型答 16 次,排名靠前的相对靠后的就是 advantage。如果 16 个 rollout 全对(题太简单)或全错(题太难),所有人**并列**,advantage 全部为 0,这一组不产生任何学习信号。

训练后期会出现一个**反直觉的衰减曲线**:模型越来越强 → 简单题全对的比例越来越高 → batch 里"零梯度组"占比 30%–50% → 你以为在训 16 题,实际只在训 8–10 题。**GPU 满载,但有效梯度在贬值**。

Dynamic sampling 两种思路:
- **DAPO 派**(过滤+重采):**over-sample 1.5–2× 然后挑有梯度的填满 batch**——简单直接,代价是 rollout 多花 50% 算力
- **WebSailor DUPO**(复制):**把有 variance 的样本复制几份**填掉零信号 slot,不重新 rollout——agentic 任务 rollout 极贵时的最优解

类比:期末复习只刷"会一半"的题最有效,完全会的和完全不会的都是浪费时间。

**证据 — 过滤路线**:
- **DAPO** (§2):**过滤 batch 中 accuracy=0 或 1 的 prompt,持续采样至 batch 填满"有梯度"样本**
- **Magistral** (§3):"过滤 non-diverse groups:全对/全错的 group 直接剔除"
- **Skywork-OR1** (§5):rejection sampling on zero-advantage groups
- **ASearcher** (§20):dynamic filtering 剔除 reward 全相同 / advantage = 0 的 query

**证据 — 复制路线(更激进,WebSailor 自研)**:
- **WebSailor DUPO** (§22):**复制 batch 内 reward variance ≠ 0 的样本** 把零信号 slot 填掉,两阶段动态采样;**收敛速度比 PPO 快一倍以上、比 DAPO 快 2–3×**——agentic env rollout 慢、reward 稀疏的最优解之一

**证据 — 课程式动态丢弃**:
- **Skywork-OR1** (§5):每阶段切换时丢弃上阶段 acc=1 prompt
- **rStar2 Stage 3** (§17):用上阶段策略筛去 8/8 全对的题,只留 hard subset

**如何抄**:over-sample $`1.5\text{–}2\times`$ 然后过滤到目标 batch(DAPO 标配);rollout 极贵时用 DUPO 复制路线。

> **🧠 思考点(作者本人,待验证)——训推异步 + 流式 variance 筛选**:把动态采样推到极致——**rollout 端作为常驻生产者源源不断产轨迹进 buffer,训练端只挑"非全对/非全错"(有 variance)的组消费**。这把 DAPO 那种**同步 over-sample 再丢弃**的阻塞代价,换成**非阻塞的流式筛选**:生产者一直满产、训练端永远有有效梯度,长尾长轨迹也不再卡 batch。代表系统:**AReaL**(全异步、生成/训练解耦 + staleness 控制)、Magistral 的 online 异步生成。
>
> - **为什么可能更好**:GPU 两端都不空转;"零梯度组"的浪费被异步藏掉、不再阻塞训练;天然吃掉生成长尾。
> - **两个不能忽略的代价**:① **off-policy staleness**——流里的轨迹由**旧策略**生成,必须配重要性采样修正 + staleness 上界,放任会崩(这是异步 RL 的核心难点);② **生成 FLOPs 并没省**——全对/全错轨迹照样被生成、只是被丢,**rollout 本身极贵时(agentic)这仍是浪费**,此时 WebSailor DUPO 的"复制"比"丢弃重采"更划算。另外全扔已解决的题可能遗忘,通常留一小撮兜底。
> - **一句话**:它是 dynamic sampling 的自然超集——**生成便宜、长尾严重时是更优解**;rollout 极贵时反而要回到复制路线。

---

## 共识 7 — Actor lr 极保守 $`\approx 1\text{e-}6`$;SFT lr $`\approx 5\text{e-}6\text{–}7\text{e-}6`$

**机制**:RL 不是从 0 优化,而是对已经 well-trained 的策略做小步修正。lr 大容易把好行为洗掉,也会放大 train-infer ratio 漂移(让 clip 频繁触发)。SFT 是分布回拉,可以 5–10× lr。

**直觉**:SFT 和 RL 的 lr 差一个量级,根源是**它们做的事情根本不同**:

- **SFT** = 把模型分布**拉向**一个已知好分布(数据集),目标明确、监督密集(每个 token 都有 ground truth),允许大刀阔斧。**毛坯房装修**。
- **RL** = 在已经 well-trained 的策略上**做小幅度微调**,目标是探索 + 强化好的行为,监督稀疏(整条 trajectory 一个 reward),lr 大一点就把已有的好行为洗掉。**精装房调家具**。

第二个隐藏理由更工程化:**lr 越大,policy 一步走得越远, $`\pi_\theta`$ 和 rollout 时的 $`\mu_{\theta_{\text{old}}}`$ 差异越大** → IS ratio 偏离 1 越远 → clip 频繁触发 → 大部分梯度被丢弃。所以即使你"勇敢地"把 lr 调到 3e-6,实际更新量未必比 1e-6 大,反而稳定性差很多。这是为什么 1e-6 几乎是 dense 模型 RL 的"宇宙常数"。

**证据 — Actor lr = 1e-6 派(主流)**:
- DAPO (§2)、ReTool (§13)、Search-R1 (§11)、rStar2 (§17)、Nebius (§29)、MiMo (§4) 全部 **1e-6**
- 都加 **20 步 linear warmup**

**证据 — 3e-6 派**:
- DeepSeek-R1 (§6):3e-6(早期工作)
- Skywork-OR1 同量级
- 7B 小模型(R1-Searcher §12)用 2e-6

**证据 — SFT cold-start lr**:
- R1-Searcher++ (§12):**2e-5**(720+85 题,6 epochs)——小数据可大 lr
- rStar2 non-reasoning SFT (§17):**5e-6**(故意保守,不想 SFT 出 reasoning)
- Tool-Star (§15):**7e-6** + cosine warmup 0.1,3 epochs
- Nebius RFT (§29):**5e-6**,50 steps
- SWE-Gym (§27):**1e-4**(491 trajectory + 5 epoch,极端小数据例外)

**经验法则**:dense 模型 RL 一律 1e-6 起步,MoE/大模型不要超 3e-6;SFT 5e-6 ~ 2e-5 视数据规模选。

---

## 共识 8 — 冷启动 SFT 视任务而定(三派分明)

**机制**:SFT 教什么决定 RL 起点。教 reasoning → RL 提升空间小但稳;只教 format/tool protocol → RL 自由度大、上限高但前期反复;完全不 SFT → 适合 base 模型直接 RL 已可 work 的领域(math/code)。

**直觉**:SFT 是"画起跑线",画在哪里决定 RL 的探索空间。关键洞察(rStar2 提出来,很多工作隐含遵守):**SFT 教 reasoning 会把 RL 探索锁死**——因为模型已经"会"一种 reasoning pattern,RL 期间不愿意探索其他路径(分布峰太尖,梯度推不动)。

三派的本质区别是**让 RL 自由探索什么**:

| 派别 | SFT 教什么 | RL 自由探索什么 | 适合场景 |
|---|---|---|---|
| A. No SFT | 啥也不教 | 一切(包括 format) | base 模型 + 任务定义清晰(math/code) |
| B. Format-only SFT | 只教 instruction-following + JSON 格式 | reasoning 路径完全留给 RL | reasoning 主导(rStar2 哲学) |
| C. Long-CoT SFT | format + reasoning 模板 + tool 协议 | 主要是优化已有 pattern | 长 trajectory agent(web/SWE) |

派 C 不是因为它"更高级",而是因为长 trajectory agent 完全不 SFT 的话 RL 起步太慢——**模型连工具怎么调都不会,前 500 步全在学 JSON 格式**,reward 全 0 没法学。一句话:**任务越长越复杂,SFT 越要把"程序性知识"打牢,把"创造性 reasoning"留给 RL**。

**派别 A — No SFT(纯 RL on base)**:
- **ToRL** (§14):Qwen2.5-Math-**Base** 直接 RL,AIME24 43.3%
- **Search-R1** (§11):"No SFT cold start"
- **ARTIST** (§16):"No SFT cold start"
- **DeepSWE** (§25):"没用 SFT 冷启动,直接在 Qwen3-32B 上 RL(32B 规模上首次)"
- **R1-Searcher v1** (§12):"不需要 distillation、不需要 SFT"

**派别 B — Yes SFT 但只教 format/protocol,不教 reasoning**(rStar2 关键洞察):
- **rStar2** (§17):**"Non-Reasoning SFT cold start"**——故意不放 reasoning 数据,只装 instruction-following + JSON function calling
  - SFT 后 MATH-500 仅 57.4%、**AIME24 仅 3.3%** 也无所谓
  - 哲学:**几乎不用 SFT 提 reasoning,全部留给 RL**(因为 SFT 的 reasoning 容易把 RL 探索锁死)

**派别 C — Yes SFT 含 long-CoT(传统 distillation 路线)**:
- **ReTool** (§13):cold-start SFT 2 epochs,人工心算改写为可执行代码
- **WebDancer** (§21):SFT 7,678 short + 6,550 long-CoT trajectory
- **Tool-Star** (§15):Tool-Star-SFT-54K + Hint-based Sampling
- **Nebius RFT** (§29):6,548 成功 trajectory + 1 epoch SFT,11% → 20% → RL 到 39%
- **R1-Searcher++** (§12):720 + 85 题,6 epochs

**经验法则**:**math/code 倾向派 A 或 B**;**web/search/long-trajectory agent 倾向派 C**(否则 RL 起步太慢、tool 调用格式炸)。

> **🤔 延伸问题:SFT 越厚越好吗?同等总 sample 预算下,"薄 SFT + 厚 RL" 能赢过 "厚 SFT + 薄 RL" 吗?**
>
> **答:base 够强、RL 能拿到非零可区分 reward 时,薄 SFT + 厚 RL 上限更高;但小模型 / 起步太难的任务,厚 SFT(蒸馏)反而赢。这是个仍有争议的开放问题——下给两侧硬数据。**
>
> **支持"薄 SFT + 厚 RL"**:
> - **rStar2-Agent** (§17,Qwen3-14B-base):本节最强证据。只做 non-reasoning SFT(SFT 后 **AIME24 仅 3.3%**、rollout ~1k token),reasoning 全交给 RL → **510 步 RL 后 AIME24 80.6% / AIME25 69.8%(pass@1),反超 DeepSeek-R1 671B**。即 **3.3% → 80.6% 几乎全靠 RL**。机制:厚 reasoning SFT 会锐化分布峰、锁死 RL 探索,薄起点反而上限高。
> - **DeepSeek-R1-Zero** (§6,671B):**完全无 SFT 纯 RL**,AIME24 **15.6% → 71.0%**;正式 R1 也只加**几千条**冷启动 SFT 兜可读性,主升仍靠 RL。
> - **"SFT Memorizes, RL Generalizes"**(Chu et al., ICML'25,[2501.17161](https://arxiv.org/abs/2501.17161)):同设定下 **RL 的 OOD 泛化显著强于 SFT**(SFT 偏记忆训练规则);V-IRL mini 上多轮 RL **+33.8%(44.0% → 77.8%)**。**关键:该文同时强调 SFT 不可省——它负责稳住输出格式、为 RL 铺路**,这恰是"薄 format SFT + 厚 RL"的依据。
> - **Nebius** (§29):SFT 11% → 20%,RL 再 20% → 39%——RL 段绝对增益更大。
>
> **反过来"厚 SFT 才对"**:
> - **小模型**:DeepSeek-R1 论文 Table 16——同一 Qwen2.5-32B-Base,**蒸馏(纯 SFT)AIME24 72.6% vs 直接大规模 RL(>10K 步)仅 47.0%**(差 25 点)。小模型探索弱,**吃强 teacher 分布 > 自己 RL 探索**。⚠️ 但这是"纯蒸馏 vs 纯 RL"、且用 DeepSeek 自家 RL 配方;**DAPO 用更好配方把同一 base 推到 50%、只花一半步数**——差距部分来自配方,非"RL 必不如蒸馏"。
> - **长程 agent(web/SWE,派 C)**:不喂够 tool 协议 / long-CoT,RL 起步 reward 全 0、无从 bootstrap(§13/21/15)。
>
> **小结**:不是越厚越好——**SFT 只需厚到"能让 RL 拿到非零、可区分的 reward"为止(尤其要稳住格式),多出的 reasoning pattern 会锐化分布、锁死 RL 探索上限**。预算够 + base 够强 → 薄 SFT 厚 RL;模型小 / 起步难 → 才需厚 SFT。⚠️ 仍有反例(后续工作发现某些 SFT checkpoint 可反超 RL,见 [2509.12235](https://arxiv.org/abs/2509.12235)),非定论。

---

## 共识 9 — 多阶段长度课程是普遍范式

**机制**:直接 train 长 context 浪费算力(短题用不到 32K)且训练不稳(KV cache 爆 → batch 缩小 → 梯度噪声大)。短 ctx 学快速 reasoning pattern,再放长补长上下文能力。**上 ctx 时减 batch 控显存**。

**直觉**:把长度课程想成**先学算术再学微积分**——短 ctx 阶段模型必须"言之有物",**逼出最精炼的 reasoning pattern**;放长后再学"什么时候该展开、什么时候该回顾"。如果一开始就给 32K,模型有"水时间"的余地,会习得啰嗦冗长的 pattern,后面再想拉回来很难。

工程账算得更直白:**ctx 翻倍 → KV cache 翻倍 → batch 减半 → 梯度噪声 √2 倍**。短 ctx 阶段用 8K 跑 256 batch,梯度信号干净;直接上 32K 只能 64 batch,前期全是噪声,reward 抖来抖去看不清趋势。**切换时机**:上阶段 acc 和 clip ratio 都稳了再升 ctx——避免"短没学透就放长"。

注意 GLM-4.5 的反例:它单阶段 64K 反而比渐进 schedule 好——说明**课程不是万能,base 模型本身长上下文能力足够时,课程可能是 overengineering**。判断标准:测一下你的 base 在 32K 上的 needle-in-haystack,> 95% 可以考虑单阶段。

**证据 — 数值**:
| 工作 | 课程 | 同步降 batch |
|---|---|---|
| Skywork-OR1 (§5) | **8K → 16K → 32K** | 收敛后切换 + 丢 acc=1 |
| Magistral (§3) | **16k → 24k → 32k** | n_batch **8k → 4k → 2k** |
| MiMo (§4) | 32K → 48K(0530 升级) | — |
| rStar2 (§17) | **8K → 12K → 12K** | Stage 3 只换数据不换 ctx |
| Nebius (§29) | **65k → 131k** | rollouts 不变,batch 128→256 |
| Tongyi DeepResearch CPT (§24) | **32K → 128K** | — |
| Cursor Composer (§32) | 32k 主训 + 后期长 ctx 扩展 | — |

**反例(单阶段也能 work)**:
- **DeepSeek-R1** (§6):32K 单阶段——能 work 但训练步数多
- **GLM-4.5 reasoning RL** (§8):**单阶段 64K** 论文报告**优于渐进式 schedule**——值得关注的反例

**经验法则**:**长 ctx 任务(SWE/Web)用课程几乎必要**;**短任务(纯 math)单阶段也行**。课程切换点选在 "上阶段 acc 收敛 + clip ratio 稳定" 时。

---

## 共识 10 — 长 trajectory 必须 partial rollout / 异步训练

**机制**:同步 RL = 等 batch 内最慢的 rollout。长 trajectory 中 p99 可能比 p50 长 10×,整 batch GPU 空转等长尾。partial rollout 把超时 trajectory 存 buffer 下轮继续;异步训练让 rollout / training 完全解耦。

**直觉**:同步 RL 像**老式接力跑——全班 16 个人,每轮等最慢的那个跑完才能开下一轮**。reasoning 任务 trajectory 长度差不多还行,**agentic 任务 p99/p50 比例能到 10×**(有人一步就答,有人 50 轮 tool call 还在 debug),意味着 90% 的时间整组 GPU 在等 1 个长尾 rollout——利用率只有 30–50%。

两条解决路径:

- **Partial rollout**(Kimi、verl):**超时的 trajectory 不丢,存进 buffer,下一个 iteration 用更新后的权重继续跑剩下的 turns**。代价是 trajectory 内部跨多个权重版本,off-policy 程度变高,但 IS correction 能修。
- **全异步**(AReaL、ASearcher):**rollout worker 和 training worker 完全解耦**,各跑各的,只在权重 update 时同步。staleness 上限 η 控制"rollout 用的权重最多比当前训练版本旧多少",η=0 退化为同步,η>0 把 GPU 拉满。

阈值参考:**> 5K tokens 或 > 5 turns 必须考虑 partial rollout;> 20 turns 必须异步**。SWE/Web agent 几乎都在异步区间,这就是为什么这些工作必须自研框架(verl/AReaL/Cursor Anyrun),纯靠开源同步 RL 跑不出来。

**证据 — Partial rollout**:
- **Kimi-Researcher** (§18):**Turn-level Partial Rollout** —— 超时任务存 replay buffer,下个 iteration 用新权重继续剩余 turns;**rollout 加速 $\geq 1.5\times$**
- **verl** (§34):AsyncPartialToolAgentLoop;rStar2 用 verl + Load-Balanced Rollout Scheduler 实现 **65K 并发 tool call、平均 0.3s 端到端延迟**
- **Magistral** (§3):NCCL broadcast 推权重不丢 in-flight 序列,KV cache 轻度过期靠 off-policy IS 校正
- **Cursor Composer 2** (§32):**mid-rollout weight sync** —— inference worker 在 rollout 中途 update 权重,后期 token 更 on-policy(突破"一次 rollout 一组权重"的传统约束)

**证据 — 全异步框架**:
- **AReaL / ASearcher** (§35/§20):**staleness 上限 η** 超参,η=0 退化为同步,η>0 GPU 利用率近 100%;**boba² v0.3 达 2.77× 同步系统加速**
- **WebSailor V2** (§22):**simulator + real-world 双环境**,仿真器调参 + 真实环境收敛
- **Seed-Thinking-v1.5** (§9):Streaming reasoning system → **3× 更快 RL 周期**

**经验法则**:trajectory 平均 > 5K tokens 或 > 5 turns 必须考虑 partial rollout;> 20 turns 必须异步。

---

## 共识 11 — 环境基础设施成为核心差异化

**机制**:RL 实际优化的是"数据分布 = 环境采样分布"。环境数量少 → 模型 overfit 几个仓库的 pattern;质量差(假 issue / 不可复现) → 学到错信号;并发低 → GPU 闲死。**算法可以抄,环境抄不来**。

**直觉**:reasoning RL 时代大家比的是**算法**(DAPO 这些 trick),agentic RL 时代大家比的是**环境**——这是核心范式迁移。

为什么?因为 reasoning 任务的"环境"就是一个 verifier 函数(对答案 → 0/1),搭起来一天的事;agentic 任务的环境是**真实可执行的代码沙箱**,要装依赖、要跑测试、要处理 timeout、要隔离副作用,**每个仓库都是独立工程**。规模差异极其惊人:

- 学术界做 SWE: 几百到几千个 docker(DeepSWE 512、SWE-Gym 2438)
- 大厂训 coding agent: **Qwen3-Coder 20K 并发**、**Cursor 数十万 sandbox(Firecracker VM,500+ pods/sec)**

环境工程比算法更难追赶,有三个层次:
1. **数量**:够不够多样,决定泛化上限
2. **保真度**(Cursor 信条 "Environment Fidelity"):**训练 harness 必须 = 生产 harness**,在简化版 SWE-bench 训练会过拟合到"题库"
3. **自动生成**:R2E-Gym SWE-GEN(commit→test 反向出题)、Cursor Function Deletion(在有测试的仓库"删功能"反向出题)——**人工标 task 是死胡同,必须有自动 pipeline**

一句话:**未来 agentic RL 的胜负在 infra 不在 algorithm**——这是为什么 Cursor / Qwen / Anthropic 都在重金投环境平台,而开源还停留在几百个 docker 的规模。

**证据 — 数量级**:
| 工作 | 环境数 | 备注 |
|---|---|---|
| DeepSWE (§25) | **512** docker | Kubernetes 编排 |
| SWE-Gym (§27) | **2,438** Python task | 11 仓库 |
| Nebius (§29) | **7,249** task | SWE-REBENCH |
| R2E-Gym (§28) | **8,700+** task | **SWE-GEN 自动 commit→test** |
| rStar2 infra (§17) | **65K 并发 tool call** | 平均 0.3s 端到端 |
| Qwen3-Coder (§30) | **20,000 并发 envs** | 目前公开最大 |
| Cursor Composer (§32) | **数十万 sandbox** | **Firecracker VM + filesystem snapshot** ("Anyrun"),500+ pods/sec 调度 |

**工程要点**:
- **Environment Fidelity**(Cursor 信条 §32):训练 harness = 生产 harness,**反对在简化环境(裸 SWE-bench)训练**——真实用户 query 是 under-specified
- **自动环境构造**是 enabler:**R2E-Gym SWE-GEN**(commit→test 反向生成)、**AgentScaler Environment Scaling**(§41)、**Cursor Function Deletion**(AI 在有单测的 codebase 里"手术式删功能"反向出题)
- **隔离与稳定**:ToRL (§14) 用 **Sandbox Fusion** 而非 qwen-agent;error message **仅保留最后一行** 防 context 爆炸

---

## 共识 12 — Entropy collapse 是头号杀手,治法多样

**机制(回顾)**:train-infer 数值差 + 没 IS correction + clip 不对称 → 分布只缩不扩(详见 §1.1 三股力分析)。一旦熵接近 0,policy 只输出一两种 pattern,reward 卡死。

**直觉**:把 policy 分布想成"探索气球"——训练开始气球饱满(高熵,啥都可能输出),正常优化应该让气球**缓慢收缩到一个稳定的中等体积**(focus on 高 reward pattern,但保留探索)。**collapse 就是气球突然漏气**,从中等体积瞬间瘪成一个点——policy 只剩一两种输出模式,reward 卡死,再训也涨不上去,而且**几乎救不回来**(rStar2 失败案例 4:温度、ctx、 $`\epsilon_{\text{high}}`$ 、optimizer reset 全试了都救不回)。

为什么这么难救?因为分布一旦塌掉,**长尾 token 的概率被推到接近 0**(比如 1e-8),Adam 优化器再想把它拉回 0.01,需要的梯度方向已经被低概率自身严重稀释——**数值上等于不存在**了。所以 entropy collapse 是**单向门**,只能预防,不能事后补救。

**三件套告警必须实时盯**(任何一个先动都是预警,三个同步爆基本确诊):
1. **entropy** 50–200 步断崖下跌
2. **clip ratio** 飙到 20%+(说明 policy 步子太大,系统在硬性约束)
3. **$`D_{\text{KL}}[\mu_{\text{old}} \| \pi_{\text{old}}]`$** 同步爆涨(说明 train-infer 数值漂移正在放大)

反直觉的两个点必须记牢:
- **温度 τ=0.6 比 τ=1.0 更容易 collapse**(Skywork 实证):低温 rollout 多样性低 → 探索少 → 熵更快塌。温度调的是"喂给 RL 的数据有多 diverse",不是"策略有多确定",别被"低温=更稳"骗了。
- **RL 上限 ≈ base model 上限**(rStar2):RL 不教新能力,只把 base **本来就偶尔能采对**的答案概率拉高(pass@k → pass@1);base 采不到的,RL 没素材可强化,自然冲不过去(参见"RL 是否真提升 base 能力"[2504.13837](https://arxiv.org/abs/2504.13837))。所以别硬刚冲过上限——上限处 reward 已饱和、没新的正确行为可学,再优化只剩"把现有赢家 token 概率怼向 1"一条路 = over-sharpen → 长尾压到 0 → collapse,前功尽弃;要用**最少 compute 抵达上限**的姿态训练。

**监控信号(三件套同时出现 = 基本确诊)**:
1. **entropy** 50–200 步内断崖式下跌
2. **clip ratio** 飙到 20%+
3. **$`D_{\text{KL}}[\mu_{\text{old}} \| \pi_{\text{old}}]`$** 同步爆涨

**治法谱(激进 → 保守)**:

| 治法 | 代表 | 说明 |
|---|---|---|
| **Clip-Higher** | DAPO/Magistral/rStar2 | 已是标配(共识 3),给低概率 token 上行空间 |
| **Adaptive entropy schedule** | **Skywork-OR1 MAGIC** (§5) | 目标熵 **tgt_ent = 0.2**,按当前与目标差动态调 $`\alpha_k`$ |
| **去掉 entropy loss** | **DeepSWE** (§25) | "entropy loss 反而会让 entropy 指数级爆炸";前提是 base 的 token-entropy ∈ [0.3, 1] |
| **同时去 KL + entropy loss** | **rStar2** (§17) | 配合 Clip-Higher + GRPO-RoC,Stage 1 clip ratio >10% 也不管 |
| **选择性丢负样本** | **Kimi-Researcher** (§18) | "负样本拉低 token 概率会引发 collapse,主动 discard 部分 negative samples" |
| **Strict on-policy + 1 grad step / rollout** | **Skywork-OR1 7B/32B** (§5) | 慢但最有效;Math-7B 用 2 steps + adaptive entropy 补偿 |
| **温度 / 阶段重置** | rStar2 Stage 3 (§17) | reset optimizer + update reference model 为最新 policy |

**反直觉证据**:
- **Skywork-OR1** (§5):**温度 $\tau = 0.6$ 反而让 entropy 提前 collapse, $\tau = 1.0$ 稳**——降温降熵是错的直觉
- **rStar2** (§17 Stage 1):故意让 clip ratio >10% 也不放,强迫学短 reasoning——**clip rate 高不必然坏**,要看在训啥
- **rStar2** (§17 失败案例 4):"step 510 之后继续训会 collapse;温度提到 1.2 / 加长 max len / 提 $`\epsilon_{\text{high}}`$ / T=20 / reset optimizer **都救不回来**"——**RL 上限 ≈ base model 上限**,关键是用最少 compute 触达上限,不要追求"再涨一点"

**【深化】熵动力学的三个硬结论(2025–2026,详见 §10L)**:

1. **熵-性能定律**(Cui et al. [2505.22617](https://arxiv.org/abs/2505.22617)):验证性能与策略熵满足 $`R=-a\cdot e^{H}+b`$,熵耗尽($`H{\to}0`$)时性能天花板 $=-a+b$。意味着 **熵一旦塌,reward 上限就被锁死**。更狠的数字:**前 200 步就消耗了 73% 的熵、拿走 76% 的性能增益**——绝大部分训练步只在做边际优化。这给共识 12 的"用最少 compute 触达上限"提供了定量背书。

2. **崩溃由极少数 token 驱动**:熵下降 $`\propto \mathrm{Cov}(\log\pi, \text{advantage})`$ 且该协方差全程为正 → 熵单调降。但驱动它的是**极端离群 token**:top-0.02% token 的平均协方差 = 5.654,全体均值仅 0.003。所以治法是**只掐这一小撮**:**Clip-Cov / KL-Cov**(对高协方差 token 裁剪或加 KL 罚),32B 上比 GRPO **+6.4%**,且熵能维持 "10× higher"。

3. **张力:压制高协方差 token vs 保留高熵 forking token**(Beyond 80/20, [2506.01939](https://arxiv.org/abs/2506.01939)):另一面是 **只有 ~20% 高熵 "forking token" 在驱动 RL 学习**,只对这 20% 做梯度更新能**匹配甚至超过**全量(Qwen3-32B AIME +11)。**两条看似矛盾**:一个说"掐掉极端高协方差 token 防崩",一个说"聚焦高熵 token 促学"。其实不矛盾——前者是 top-0.02% 的**病态离群点**(协方差异常,该抑制),后者是 top-20% 的**正常分叉点**(熵高但协方差正常,该保留)。记住:**梯度应聚焦少数关键 token,但要分清"病态离群"和"健康分叉"。**

**【深化】bf16 rollout 不崩、量化 rollout 才崩**:多个团队独立观察到 **bf16 rollout 时不出现 entropy collapse,而 FP8/量化 rollout 会**——这把 entropy collapse 与 §44 的 train-infer 数值失配直接挂钩:**很多"熵崩"其实是"训推不一致"的下游症状**,先查 §44 再调熵超参。另一干净分解(2509.26114):**clip-high 降熵、clip-low 升熵**——光靠选 clip 参数就能控熵,不必动 KL。

---

## 共识 13 — 判分两域分治:可验证用 RLVR、不可验证才上 GRM

**机制**:reward 的可信度决定方法。**可验证任务**(math / code / SWE-test,有干净 verifier)直接 RLVR / outcome-only 最稳最省(共识 4),还能删 KL(共识 2);**不可验证任务**(开放写作 / 研究综合 / 主观质量 / long-horizon agent 行为)规则写不出来,只能上**生成式奖励模型(GRM)**。而 GRM 只是真实偏好的 proxy,policy 一旦漂出它的训练分布就被 reward hacking(proxy 分单调涨、真实质量先升后崩),所以 GRM 这条线的工程重心全在"**怎么不让 judge 被自己优化的 policy 钻空子**"。

**直觉**:RL 防作弊靠的是"出题人 ≠ 答题人"。verifier 干净时,出题人是铁面无私的规则(对就是对),怎么 optimize 都钻不动;一旦换成 GRM,出题人变成一个会犯错、有盲区的模型,而答题人正全力以赴地找它的漏洞——**这就是为什么 verifiable 域几乎不用为 reward 操心,不可验证域却要把一半精力花在 reward 上**。

**证据 — 两域分治**:
- **可验证 → RLVR 无 RM**:共识 4 全员(ReTool §13 / ToRL §14 / rStar2 §17 / DeepSWE §25 / Search-R1 §11 …)
- **不可验证 → GRM,三条路线**:
  - **offline 冻结**(经典 RLHF RM):易 over-optimization,Gao et al. 2023 [2210.10760](https://arxiv.org/abs/2210.10760) 给出"真实 reward 随 policy 偏离 init 单调恶化、RM 越大越严重"的标度律
  - **online 刷新**(主流稳健解):Kimi K2 **闭环 critic refinement + critic 资格门控**(§19)、SPCT / DeepSeek-GRM **online RL + principle-as-reward + meta-RM 投票**([2504.02495](https://arxiv.org/abs/2504.02495))、Seed-Thinking **双轨 reward**(§9)、DeepSeek-V3 **自身 + voting 做开放问题 self-feedback**(§6)
  - **合并 actor–GRM**(前沿):DeepSeek-V4 让 actor 自己当 judge、RL 同时优化生成与评判

**反例 / 边界**:
- **KL 在这里和共识 2 反向**:共识 2 说 reasoning RL 删 KL,但那条留了边界"reward 可疑才要 KL 兜底"——**GRM 域正是 reward 可疑的全部场景,所以要把 KL 加回来**,当限制 distribution shift、抑制 over-optimization 的头号手段。
- **合并 actor–GRM 是高风险前沿,不是默认**:它结构上消灭了 actor–RM 分布失配、还省一份模型,但**拆掉了"独立判分"这层安全垫** → 自我合谋式 hacking + 目标干扰,必须用 principle / 资格门控 / verifiable anchor / 判分支路 detach 人为补回独立性。**base 越强越安全,base 越弱越该保留独立 judge**。

**如何抄**:**能 verify 就别碰 GRM**(RLVR + 删 KL);**必须 GRM 就从"online 刷新 + 资格门控 + KL 兜底"起步**(Kimi K2 §19 + SPCT 的成熟组合);**合并是方向性正确但高风险的前沿**。完整三路对照、over-optimization 机制与推荐决策树见 **§47**。

> **🧠 思考点(作者本人,待验证)**:三条路线本质是一条连续谱(冻结 → 周期 refresh → 共训分离 → 共享骨干分离头 → 全合并)。**合并 actor–GRM 不是"缝一起省事",而是拿"独立判分"这个安全资产去换"零分布失配 + 能力互促"**——而"省一份模型"省的只是显存、不是判分算力,别被它诱进高风险区。换不换得值,看四个护栏能否补回那层独立性:
> - **① gap 错配**:合并在 generator–verifier gap 大的任务最安全、gap 小(纯主观)最危险——而 gap 小恰是你不得不上 GRM 的原因 → 先用在"半可验证"任务。
> - **② 检测器失效**:合并把 hacking 从"搜固定对手盲点"变成"协同收敛到自洽但错的不动点",**还废掉了 gold-RM divergence 这个检测信号,必须外挂一个完全独立的 judge 做周期体检**。
> - **③ 务实端**:实战甜点大概率是"共享 trunk + 独立 critique head + 判分支路 stop-grad",而非 V4 式全合并。
> - **④ 偏悲观**:self-judge 的 false-positive 会被正 reward 复利放大 → judge 应**刻意调成"拿不准就低判"**(接 offline-RL pessimism)。
>
> 四点展开 + 出处见 **§47**。

---

## 共识 14 — IS 粒度之争:token 级(+TIS) / 序列级(GSPO) / 裁权重不丢 token(CISPO)【2025H2 新浪潮】

**机制**:GRPO 的 importance ratio 是 **per-token** 的 $`r_{i,t}=\pi_\theta(y_{i,t})/\mu_{\theta_{\text{old}}}(y_{i,t})`$。但 reward 是**整条序列**给的(outcome 0/1)。GSPO 一句话点破:**"优化目标的单位应当与 reward 的单位匹配"** —— reward 是序列级,off-policy 校正也该是序列级。token 级比率的病在于:每个 next-token 分布**只有一个样本**,IS 在 $`N{=}1`$ 上根本起不到分布校正作用,只是**注入高方差噪声**,噪声沿长序列累积、再被 clip 放大,最终崩溃"往往不可逆"。

**直觉**:把一条 16k token 的 rollout 想成 16k 次"掷骰子"。GRPO 给每次掷骰单独算一个修正系数 $`r_{i,t}`$ —— 但你只掷了一次,这个"系数"纯属噪声。16k 个噪声系数连乘/累加,方差爆炸。三条新路线是三种"降噪"哲学:

| 路线 | 代表 | 做法 | 一句话 |
|---|---|---|---|
| **token 级 + 截断 IS** | TIS / MIS / IcePop | 保留 per-token ratio,但对极端值**单边截断** $\min(r,C)$ 或置零 | "留着但别让它炸" |
| **序列级** | **GSPO** | ratio 改成整条序列似然比的**几何平均**(长度归一),clip 也在序列级 | "换掉单位,从根上降方差" |
| **裁权重不丢 token** | **CISPO** | 裁的是 IS **权重**而非 token,**所有 token 都保留梯度** | "别把探索 token 删了" |

**为什么这是 2025 下半年的大事**:它和[共识 3](#共识-3--clip-higher-已成标配)(clip-higher)、[共识 12](#共识-12--entropy-collapse-是头号杀手治法多样)(entropy)、[§44](#§44-训推一致性隐形杀手)(train-infer)全是同一个根问题的不同切面——**怎么处理 $`\pi_\theta`$ 与 rollout 分布 $`\mu_{\theta_{\text{old}}}`$ 的偏离**。GRPO 时代靠 clip 硬截;新浪潮直接动 IS 比率的定义。

**证据 — 序列级 GSPO**(§10A):
- **GSPO**(Qwen, [2507.18071](https://arxiv.org/abs/2507.18071)):序列比率 $`s_i=(\pi_\theta(y_i)/\mu(y_i))^{1/|y_i|}`$;**用于 Qwen3**。最关键的工程红利:**MoE RL 不再需要 Routing Replay**(对照 MiniRL R3 §1.3)——序列似然对"单个 token 路由漂移"不敏感,而 GRPO 在 Qwen3-30B-A3B 上每步更新后约 **10% expert 路由会变**,逼得必须 replay。GSPO 直接绕过。
- **反直觉**:GSPO 被 clip 掉的 token 比例比 GRPO **高两个数量级**,用更少 token 估梯度反而更稳——作者据此断言 GRPO 的 token 级估计"本身就是噪声大且低效的"。

**证据 — 裁权重 CISPO**(§10B):
- **CISPO**(MiniMax-M1, [2506.13585](https://arxiv.org/abs/2506.13585)):裁 IS 权重、**不丢任何 token**。动机是那些 **"However / Wait / Aha" 这类低概率高 ratio 的 "fork token"**(推理分叉点)在 PPO/GRPO 下"第一次 on-policy 更新就被 clip 掉了",而它们恰恰对维持熵、支撑长程探索最关键。在"每 batch 16 轮 off-policy 更新"的重复用样设置下,**DAPO 的 clip-higher 反而效果差**,CISPO 才稳。
- **ScaleRL**(Meta, [2510.13786](https://arxiv.org/abs/2510.13786))在其 scaling-law 实验里把 **CISPO 选作 loss(优于 GSPO 和 DAPO)**,并指出 loss 类型会**改变性能渐近线(天花板)**而非只改算力效率。

**证据 — 几何平均 GMPO**(§10C):
- **GMPO**([2507.20673](https://arxiv.org/abs/2507.20673)):token reward 的**几何平均**(对离群 ratio 鲁棒,AM-GM 不等式保证目标方差更小);token 级 clip 但范围放宽到 $`(e^{-0.4},e^{0.4})`$;R1-Distill-7B 上比 GRPO **平均 +4.1%**。

**反例 / 边界(必读,别把新算法当银弹)**:
- **Lite-PPO**(§10E, [2508.08221](https://arxiv.org/abs/2508.08221)):**clip-higher 的收益依赖模型类型与规模** —— base 模型 clip 率本就 ~0.003,抬上界几乎无效甚至有害;只有 aligned 模型 + 合适规模才显著。4B 模型上 $`\epsilon_{\text{high}}{=}0.32`$ 最优,8B 上 0.28 最优,**scaling 关系小模型成立、大模型不成立**。结论:**极简两件套(group 均值 + batch 标准差归一 + token-level loss)就能反超组件繁多的 DAPO**。
- **GSPO 在高 staleness 下会崩**:INTELLECT-3(§41E)用 **async-8** 做压测,**GSPO 下 reward 直接崩溃**——序列级 IS 在极端 off-policy 时反而不如双侧裁剪(INTELLECT-2 的 $`\epsilon{=}0.2,\delta{=}4`$)。
- **GSPO 机制归因有争议**:follow-up [2509.24203](https://arxiv.org/abs/2509.24203) 认为真正起作用的是"序列级 clip 当正则",而非序列级 IS 本身——标"有争议"而非定论。
- **TIS 本身也有争议**:见[共识 12](#共识-12--entropy-collapse-是头号杀手治法多样)与 §44——slime 在 Search-R1 3B 上开 TIS 反而早期 collapse,改用 MIS 才稳。

**如何抄**:
- **dense + 中小规模 + 近 on-policy**:token 级 GRPO/DAPO + TIS 兜底就够,别急着上序列级。
- **MoE**:优先 **GSPO**(免 Routing Replay),或 MiniRL R3 二选一。
- **大量 off-policy 复用样本(每 batch 多轮更新)**:**CISPO**(别 clip 掉 fork token)。
- **高 staleness / 异步**:**双侧 clip**($`\epsilon{=}0.2,\delta{=}4`$,共识见 §41E),不要纯 GSPO。
- 完整谱系与决策树见 **§49**。

---

## 共识 15 — 多轮 agentic RL 有独立于单轮的崩溃模式(void turn / echo trap / entropy 失控 / template collapse)【最重要的新坑】

> **这条是本轮更新最该看的**。单轮 reasoning RL 的崩溃主要就一种(entropy collapse,共识 12)。**多轮 agentic RL 引入了一整族单轮见不到的崩溃**,根因都是"工具/环境反馈把分布拽出训练域 + 跨轮复合"。下面四种各有独立的**触发机制、监控信号、修复手段**。

**机制(四种崩溃)**:

**① Void Turn → 梯度爆炸(SimpleTIR §17A)**:某一轮 LLM 响应**既没有完整代码块、也没有最终答案**(残缺代码 / 重复 / 过早 eos)= void turn。工具反馈本就 OOD,模型对其后 token 赋异常低概率 → **IS ratio 上方无界、梯度范数灾难性爆炸**;低概率 token 又喂回下一轮,**逐轮复合恶化**。

**② Echo Trap → 模板化坍缩(RAGEN/StarPO §17B)**:模型**过拟合到局部高奖励的推理套路**,早期输出多样、训练后坍缩成固定措辞模板,RL 强化的是表层 pattern 而非泛化推理。

**③ Entropy 失控(不只是塌,还会炸)(EPO §17C)**:多轮里所有轮**共享同一套策略参数**,逐轮调熵无法解耦"早期探索 vs 后期利用",熵会**剧烈震荡**(暴跌也暴涨)——称 "exploration-exploitation cascade failure"。**单轮的"熵只会塌"直觉在这里失效**。

**④ Template Collapse → 熵看不见的崩溃(RAGEN-2 §17C)**:**即使熵保持高位**,推理仍可能漂向固定模板——单个输入内看着多样,**跨输入几乎相同**(互信息 $I(X;Z)\to 0$)。这是"熵和所有现有指标都看不见的故障模式"。

**直觉**:单轮 RL 像一次性考试,答完就给分;多轮 agentic 像连续多场带反馈的对弈,**每一步的环境反馈都可能把模型推到它没见过的局面**。这些 OOD 局面上模型最慌(token 概率乱、熵乱),而 RL 又把这些慌乱时刻的梯度放大——于是出现单轮永远不会有的"逐轮雪崩"。

**监控信号(看哪个指标)**:

| 信号 | 对应崩溃 | 出处 | 阈值/现象 |
|---|---|---|---|
| **梯度范数尖刺** | void turn 致 IS 爆炸 | SimpleTIR | 尖刺出现 ≈ 不可逆崩溃临界点 |
| **reward std 断崖(早于 reward mean)** | Echo Trap | RAGEN | FrozenLake:std 在 step 40 跌,reward 直到 step 90 才崩——**提前 50 步预警** |
| **熵剧烈震荡(非单调下降)** | entropy 失控 | EPO | 暴涨暴跌交替 |
| **熵高但互信息 MI(X;Z) 跌** | template collapse | RAGEN-2 | MI z-score 与性能 Spearman **+0.39**,而熵是 **−0.11~−0.14(方向反了!)** |
| **工具返回后前 10–50 token 熵飙升** | 工具反馈不确定性 | ARPO | search 反馈比 python 引入更多不确定性 |

**修复手段**:
- **过滤含 void turn 的整条轨迹**(SimpleTIR):注意**单纯过滤"低概率 token"或"高 ratio token"都救不了**,必须以 void turn 为标志过滤整条。Qwen2.5-7B 上 text-base 3.2 → SimpleTIR **50.5**(AIME24)。
- **StarPO-S 三件套**(RAGEN):① 按轨迹 reward std 只**保留 top-25% 高方差** prompt;② 用 **PPO(带 critic)而非 GRPO**(critic-free 更不稳);③ clip-higher + 去 KL。保留 75% rollout 把稳定期从 100→140 步,保留 50% 直接不崩。
- **EPO**:轨迹级熵正则 + 把熵锚定到历史均值走廊 **$[0.5\bar H,1.5\bar H]$**(±50%)。ScienceWorld 上 +152%。
- **RAGEN-2 的 SNR 过滤**:用 prompt 内 reward 方差作信噪比,保留 top-ρ(ρ≈0.9),**还省 26–41% 每步时间**。
- **长程信用分配**:GiGPO(§17D)两级优势(episode + anchor-state step 级),ALFWorld +12%/WebShop +9%;Tree-GRPO 前缀共享,1/4 预算超 chain。
- **熵驱动分支**(ARPO §17E):在工具返回后高熵点多分支采样,一半 tool 预算达到/超过 GRPO。

**反例 / 边界(stabilizer 会反噬)**:
- **RAGEN-2 给 StarPO-S 打补丁**:当 **reward 方差趋近 0**(任务太易/太难),任务梯度消失但**正则梯度恒定**,对所有推理链施加"均匀收缩" → **反而加速 template collapse**。即"奖励区分度弱时盲目加正则有害",诊断指标低时(如某些 GRPO 设置)过滤反而掉点。
- **EPO 并非处处正收益**:PPO+EPO 在 ALFWorld 的某指标反降 10.9%(GRPO 因 group-relative 本就稳,增益更小)。
- **RAGEN-2 / EPO 是 2026 极新预印本**,结论可能演化。

**如何抄**:多轮 agentic RL 的最低配监控 = **梯度范数 + reward std + (条件允许加)MI 诊断**;开训先加**轨迹级过滤**(void turn / 低方差);长程任务上 step 级信用分配(GiGPO);**别只盯 entropy**——它在 template collapse 下会骗你。完整故障→信号→修复表见 **§17C 末** 与诊断速查表。

---

> 以上 15 条之外的训推一致性细节见 **§44**,算力规模参考见 **§45**,实战教训精粹见 **§48**,算法选型决策树见 **§49**。完整章节走 Part I–V。

---

# 训练仪表盘:Agentic RL 必看监控指标

> 15 条共识告诉你**该怎么训**;这一节告诉你**训的时候盯什么**。
>
> 一句话总览:**没崩之前看 entropy / IS ratio / clip ratio,崩了之后看 reward 各分量 / response 长度 / trajectory 形态找 root cause**。MiniRL §1.5 / Skywork-OR1 / rStar2 都把 "entropy + KL + clip" 列为顶级监控量(详见各自章节)。
>
> 下面按 **重要性 × 通用性** 排序,每类指标都给出:**算什么、健康范围、异常含义、谁明确监控**。

## 类 1 — 策略分布健康度(头号 dashboard)

| 指标 | 计算 | 健康范围 | 异常含义 |
|---|---|---|---|
| **Token-level entropy** $`H[\pi_\theta(\cdot \mid x, y_{\lt t})]`$ | 每个 generation step 的 softmax 熵,batch 平均 | base 模型通常 0.3–1.0;训练中平稳或缓降 | **断崖下跌(50–200 步内掉到 < 0.1)= entropy collapse 确诊**(共识 12) |
| **Per-position entropy curve** | 按 token position 分桶画熵 | reasoning 段 > tool-arg 段(structured token 本就低熵) | 所有 position 同步 collapse = 系统性 bug;尾部 position 先 collapse = trajectory 过长积累 |
| **Generation diversity** | 同 prompt 16 rollouts 间 BLEU / distinct-n / pass@k 分散度 | 同 prompt 内 pass 数差异 ≥ 30% | 所有 rollouts 完全一样 = 已经 mode collapse(group advantage 全 0) |
| **Token-entropy 阈值告警** | $`H \lt  0.1`$ 持续 20 步 | — | DeepSWE (§25) 经验:base token-entropy ∈ [0.3, 1] 时**不需要 entropy loss**;脱出此区间立刻 entropy schedule 介入 |

**明确监控的工作**:MiniRL (§1.5)、Skywork-OR1 (§5,**目标熵 tgt_ent=0.2** 动态控)、Magistral (§3,clip ε 动态调以维持 group entropy)、rStar2 (§17,Stage 划分按 clip ratio 稳定时机)。

**反直觉**:Skywork-OR1 实证 **温度 τ=0.6 反而让 entropy 提前 collapse,τ=1.0 才稳** —— 训练阶段降温不等于降熵。

---

## 类 2 — 训推一致性指标(系统层 bug 探测器)

这一组是 train-infer mismatch 的"血液检测"。任何接 vLLM/SGLang 的 agentic RL pipeline 都必须开。

| 指标 | 计算 | 健康范围 | 异常含义 |
|---|---|---|---|
| **IS ratio 分布** $`r_t = \pi_\theta(y_t) / \mu_{\theta_{\text{old}}}(y_t)`$ | 整 batch token 的 ratio 直方图;mean / std / p99 | mean ≈ 1.0,分布**集中在 [0.8, 1.27]**(§44 实战 checklist) | mean 系统性 > 1 = train-infer 数值差(FP8 vs BF16) 或 retokenization drift;p99 > 5 = 必上 TIS 截断(MiniRL §1.4,阈值=5) |
| **Clip ratio (high / low 分开)** | $`\Pr[r_t \gt  1+\epsilon_{\text{high}}]`$ 和 $`\Pr[r_t \lt  1-\epsilon_{\text{low}}]`$ 分别统计 | < 5%(reasoning) / < 10%(agentic) | **> 15% 红线**(§44 checklist 第 3 条);单侧 > 20% 必查 logprob mismatch |
| **$`D_{\text{KL}}[\mu_{\theta_{\text{old}}} \,\Vert\, \pi_{\theta_{\text{old}}}]`$** | 同 prompt 同权重,推理 logprob vs 训练 logprob 的 KL | 趋稳或缓变 | **同步爆涨 = 训推数值漂移**(§1.1 "三股力"中的 train-infer discrepancy 物理来源) |
| **$`D_{\text{KL}}[\pi_\theta \,\Vert\, \pi_{\theta_{\text{old}}}]`$** | 同权重新旧策略 KL | 平稳 | 爆涨 = policy staleness 过大(mini-batch 拆太多步);收紧 ε 或减 mini-batch 数 |
| **MoE expert routing diff** | 训推同 token 的 top-K expert 重合率 | > 95% | < 80% = 必上 R3 (Rollout Routing Replay,见 MiniRL §1.3) |

**经典告警组合(§44 实战 checklist 总结的"三件套同时报警 = 训推不一致确诊")**:
1. entropy 骤降
2. clip ratio > 15%
3. KL(μ_old ‖ π_old) 同步爆涨

→ **立刻停训查 logprob mismatch**(优先级:retokenization drift > FP8/BF16 数值差 > MoE 路由)。

---

## 类 3 — Reward 与 Advantage 健康度

| 指标 | 算什么 | 看什么 |
|---|---|---|
| **Training reward (mean / std / median)** | batch 内 reward 统计 | mean 应缓涨;std 突降到 0 = batch 已全对/全错,**dynamic sampling 必须开**(共识 6) |
| **Zero-advantage 比例** | $`\Pr[\text{group } \hat A_i \equiv 0]`$ | > 30% = batch 利用率塌方,触发 DAPO / DUPO 重采(共识 6) |
| **Group advantage 方差** | 每 group 内 advantage 的 std | 单调降至 0 = 即将无梯度(GRPO 致命) |
| **Reward 各分量分别画线**(混合 reward 必须做) | format / correctness / shaping / length 分量分别记 | **某辅助分量异常上涨 + correctness 不动 = reward hacking 信号**;典型案例: Cursor "学会主动询问澄清以避免被罚"(§32 涌现/坑) |
| **Pass-rate 课程位置** | 当前 batch 题目的 pass-rate 直方图 | 应中间峰(0.3–0.7 占主体);全跑到 1.0 = 课程升级时机(MiMo §4 / Skywork §5) |

**Reward hacking 探测信号(综合各工作教训)**:
- **Search-R1 (§11) "answer avoidance"**:F1 reward 下,**回答率突降 + reward 不降**——模型学会"不答以避免错答"
- **rStar2 (§17) tool-error 比例**:positive trajectory 里 tool error 比例稳定 10–15% 不降 = step-level shaping 在掩盖问题(GRPO-RoC 应运而生)
- **Magistral (§3)**:把 reward 拆 format / correctness / length / language consistency 四线,**单分量异常单挑出来**

---

## 类 4 — Trajectory 形态(长度 / 重复 / 截断)

| 指标 | 算法 | 阈值 / 出处 |
|---|---|---|
| **Response length 分布** | mean / p50 / p95 / max | mean 持续 ↑ + correctness 不动 = 啰嗦化 hacking;追 Kimi-Researcher (§18) γ-decay 风格抑制 |
| **Overlong / truncation rate** | $`\Pr[\text{len} \geq L_{\max}]`$ | DAPO 软惩罚窗 $`L_{\text{cache}}=4096`$ 起触发(§2);**rStar2 (§17 失败案例 1) 警告:overlong 比例上升后,过滤反而加剧** |
| **N-gram 重复率** | k-gram (k=10) 出现次数 | WebDancer (§21) **10-gram 阈值 = 4**;超阈值直接判 invalid。**rStar2 (§17 失败案例 2)**:N-gram 检测会误杀合法 verify pattern("换个输入验证"),需白名单 |
| **Compact filtering trigger 比例** | max_context / 20min timeout / max_steps 三类 | DeepSWE (§25):三类触发率超过 ~5% 启动 mask;Nebius (§29) **反对** 此做法,改用 soft length penalty |

**经验法则**(综合 §1.7 + §17 + §29):**永远画 response length 直方图,而不只是 mean**;mean 平稳但 p95 飙 = 长尾失控;p95 平稳但 mean 漂 = 整体啰嗦化。

---

## 类 5 — Agentic 专属指标(multi-turn / tool / web)

这一类是 reasoning RL 没有的、agentic 特有的"诊断信号"。

| 指标 | 含义 | 典型出处 / 红线 |
|---|---|---|
| **Turn count 分布** | trajectory 平均 / max turns | rStar2 (§17):Stage 1/2 max=10、Stage 3=15;Kimi-Researcher 平均 23 步、可达 70+;ASearcher 7B/14B=32、QwQ-32B=128 |
| **Invalid trajectory rate** | format 不合规 / 调用不存在 tool / JSON 解析失败 比例 | WebDancer (§21) 失败案例:**long-CoT → instruction model 迁移 invalid rate 13.6–21.4%**;> 10% 必须改 SFT 或加 format reward |
| **Tool call 成功率** | $`\frac{\text{成功 tool call}}{\text{总 tool call}}`$ | GLM-4.6 (§8) 关键迭代目标;**拒识未知工具、最小化臆造参数** 是显式监控目标 |
| **Tool call 类型分布** | search / python / browse 各占比 | Tool-Star (§15):若 multi-tool bonus $`r_M`$ 加了,但分布仍单工具 = 协作没学起来 |
| **Tool error 类型直方图** | timeout / syntax / API failure / 语义错误 | rStar2 (§17) GRPO-RoC 按 $`p_{\text{err}} = \text{错}/\text{总}`$ 反比例采正样本,前提是这个分布算得出 |
| **"Over-action"率** | 答案确认后继续行动 比例 | WebDancer (§21) 失败案例明确点名;**长 trajectory RL 经典 reward hacking** |
| **Context utilization** | 实际用到的 history token / context window | Kimi-Researcher (§18) "naive 约 10 iter 就 OOM";开 context-management 后单条 rollout > 50 iter;DeepSeek-V3.2 128K 下 ~20% agent 任务超限,Discard-all 管理把 BrowseComp 51.4→67.6(§33D) |
| **Self-summarization 频率** | Cursor-style 主动总结调用次数 / trajectory | Cursor Composer 2.5 (§32):**hard task 上主动多次 summarize 是健康信号** |
| **Void turn 比例**(TIR/多轮) | 一轮内既无完整代码块又无答案的响应占比 | SimpleTIR (§17A):>0 即埋雷;过滤含 void turn 的**整条轨迹**(只过滤 token 没用) |
| **工具返回后 token 熵** | 工具反馈后前 10–50 token 的熵 | ARPO (§17E):此处熵必飙;**搜索反馈比 python 反馈引入更多不确定性**,是分支采样的触发点 |
| **重复动作序列长度**(SWE agent) | 连续重复 action 的最大长度 | SWE-smith (§33A):长度-10 重复序列 → **89% 失败概率**;>25% 的 32B 轨迹有此问题(Claude 3.7 <4%) |

---

## 类 6 — 梯度与优化(怀疑被噪声主导时看)

| 指标 | 健康范围 | 异常含义 |
|---|---|---|
| **Gradient norm** | 平稳;clip 阈值通常 1.0(Nebius §29);**INTELLECT-2 激进到 0.1**(§41E) | 持续 hit clip = lr 过大或 reward scale 失控;趋 0 = entropy 已死;**多轮任务突然尖刺 = void turn 致 IS 爆炸(SimpleTIR §17A),尖刺即不可逆崩溃临界点** |
| **Policy update magnitude** $`\lVert \theta_{t+1} - \theta_t \rVert`$ | 与 grad norm × lr 一致 | 与 grad norm 不一致 = optimizer state 异常(rStar2 (§17) Stage 3 显式 reset optimizer) |
| **Actor loss / value loss / entropy loss 分别记** | actor 缓降;value(PPO)与 actor 同量级 | entropy loss 系数 × entropy 绝对值若过大 = 反向推熵爆炸(DeepSWE 删 entropy loss 的理由 §25) |
| **AdamW 配置(易漏)** | 默认 β=(0.9,0.999), eps=1e-8 | RL 梯度幅度可低至 1e-18,**MiniMax-M1 实测默认 eps 不收敛**,改 β2=0.95 / eps=1e-15(§10B);ORZ/INTELLECT 用 β2=0.95(§10H/§41E) |
| **跨输入互信息 MI(X;Z)**(多轮/推理多样性) | 与性能正相关 | **熵高但 MI→0 = template collapse**(RAGEN-2 §17C);MI 预测性能比熵可靠 2×,熵方向甚至是反的 |

---

## 类 7 — Eval 指标(防 train reward 涨、eval 不涨)

| 指标 | 出处与建议 |
|---|---|
| **Avg@K 而不是 Pass@1** | **Skywork-OR1 (§5) 强推**:Pass@1 噪声大,Avg@K 更能反映"系统性"提升;K 通常取 16–32 |
| **Pass@K spread (K=1, 8, 32 对比)** | 揭示是 capability 涨还是 sampling 涨;P@32 涨 P@1 不涨 = 探索好但收敛差 |
| **OOD benchmark** | rStar2 (§17) 在 AIME / Olympiad 之外补 LiveCodeBench;SWE-RL (§26) 涌现 5 个 OOD 能力(function coding / library / math / language understanding)正是靠 OOD eval 发现的 |
| **Eval reward vs eval pass@1 gap** | reward 涨但 benchmark 不涨 = **reward hacking 黄金信号**;Cursor (§32) 主要靠这个发现"主动询问澄清"hacking |
| **Trajectory 长度 vs 难度的相关性** | γ-decay (Kimi §18) 生效信号:难题 trajectory 更长;失效信号:所有题等长 = decay 没起作用 |

---

## 类 8 — 系统 / Infra(决定是否要换架构)

| 指标 | 出处 / 决策 |
|---|---|
| **Rollout 长尾 p99 / p50** | > 5× → partial rollout 必上(共识 10);Kimi-Researcher (§18) turn-level partial rollout 加速 $\geq 1.5\times$ 的根源 |
| **GPU 利用率(rollout 阶段 vs train 阶段)** | rollout idle > 30% → async;AReaL boba² v0.3 (§35) 全异步达 **2.77× 同步系统加速** |
| **Staleness 分布**(async 训练) | AReaL 引入的 **η 上限超参**(§35),η=0 退化同步,η 越大 GPU 利用率越高但 off-policiness 越大,通常监控 mean staleness 调 η |
| **Tool call 端到端延迟 p50 / p99** | rStar2 (§17) 实现 **65K 并发 tool call、平均 0.3s 端到端**;p99 > 30s 必排查 sandbox 资源 |
| **Sandbox 失败率 / 重启率** | ToRL (§14) 用 Sandbox Fusion 替代 qwen-agent 的核心理由;失败率 > 1% 必须排查环境 |

---

## 诊断速查表:看到 X 先怀疑什么

| 异常现象 | 优先怀疑(从高到低) |
|---|---|
| Entropy 断崖 + clip ratio 飙 + KL(μ‖π) 爆 | **train-infer mismatch**(retokenization drift > FP8 vs BF16 > MoE routing)→ §44 |
| Reward 持续涨 + benchmark 不动 | **reward hacking** → 拆分 reward 各分量画图 → 共识 4 + Search-R1 answer avoidance(§11) |
| Group advantage 多数 = 0 + batch 利用率塌 | 课程切换时机到了 → DAPO/DUPO 重采(共识 6) |
| Response length 持续涨 + correctness 不动 | 啰嗦化 hacking → Kimi γ-decay(§18) 或 Magistral 长度惩罚(§3) |
| Overlong rate 持续涨 | **不要 overlong filtering**(rStar2 失败案例 1 §17)→ 软惩罚或保留负 reward |
| Tool call invalid rate > 10% | SFT 没教好 format,或 RL 阶段 format reward 缺失 → 共识 8 派 B/C |
| Eval Avg@K 涨,Pass@1 不涨 | sampling 多样性好但收敛差 → 检查温度 / clip-low |
| Eval Pass@1 涨,Avg@K 不涨 | mode collapse 早期信号 → 检查 entropy 是否已经掉 |
| MoE 训了几百步突然 collapse | expert routing diff 没监控,R3 没上 → MiniRL §1.3;或换 **GSPO** 免 routing replay(§10A) |
| 长 trajectory 训练 GPU 利用率 < 50% | 同步 rollout 长尾,partial rollout / async 没上(共识 10) |
| **梯度范数突然尖刺(多轮/TIR 任务)** | **void turn** 致 IS 爆炸 → 过滤含 void turn 的整条轨迹(SimpleTIR §17A);尖刺=不可逆临界点,别硬抗 |
| **reward std 断崖(reward mean 还没动)** | **Echo Trap** 早期预警(提前数十步)→ 上 StarPO-S top-25% 方差过滤 + PPO critic(§17B) |
| **熵剧烈震荡(暴涨暴跌交替,非单调降)** | 多轮 **entropy 失控** → 轨迹级熵正则 + 熵走廊 [0.5,1.5]·H̄(EPO §17C) |
| **熵保持高位但性能停滞 / 跨输入输出雷同** | **template collapse**(熵看不见)→ 上 MI 诊断 + reward 方差(SNR)过滤(RAGEN-2 §17C) |
| **reward 方差近 0 时加正则反而更糟** | stabilizer 反噬 → 此时**别盲目加正则**,先提高任务区分度(RAGEN-2 §17C) |
| **异步/高 staleness 下 GSPO reward 崩** | 序列级 IS 扛不住高 off-policy → 改**双侧 clip** ε=0.2/δ=4(INTELLECT §41E) |
| **量化 rollout(FP8)熵崩,bf16 不崩** | train-infer 数值差被量化放大 → FP32 LM head / TIS / 换 bf16 rollout(§44 + §10B) |
| **代码 RL:测试全过但 patch 仍错 / reward 涨 benchmark 不动** | 测试用例质量差致 **reward hacking**(trivial 测试、public test 截断)→ §33A/§33C |
| **IS ratio 系统性偏 1 但 token-id 已对齐** | 纯精度问题 → 试 **FP16 全局**([2510.26788] §50.2)或 **FP32 lm_head**(§44③),别先调熵 |
| **异步/高 staleness 崩 + 崩前负 advantage 样本占比高** | **正负 staleness 不对称** → 对**负样本**更紧 clip / 负优势 off-policy 样本 zero loss(§50.3),别一律加 KL |
| **RL 一开训就崩 / SFT 训得越久崩得越早** | 根因在 **SFT 起点**:熵太低、epoch 过拟合 → 减 epoch + dynamic-γ(§50.4),别只调 RL 超参 |
| **权重转换后 embedding 形状报错 / logprob 错位** | **Megatron 词表 padding**(`make_vocab_size_divisible_by=128`)→ 转换设 `--vocab-size`(§50.1) |
| **verl:advantage 全组同号 / 分组数对不上 / 归一化怪** | **uid 分组 / repeat-n 配置错**,先查 `rollout.n` 与 uid 传播(§50.1),不是算法错 |

---

> 监控这件事的本质是 **"训练崩之前你能不能看见"**。rStar2 (§17 失败案例 4) 的核心教训:**step 510 之后 collapse 救不回来**——所以早期监控比事后调参重要。最低配:**entropy + clip ratio + IS ratio 分布**,三条线撑起 90% 的 debug 能力。


# Part I. 通用 LLM RL 工程基础

> **本部分要回答**:reasoning RL(单轮 math/code)训练稳定性的核心问题是什么,业界用什么算法把它压住。
>
> **核心观点**:
> - **理论锚点(§1 MiniRL)**:token-level surrogate 只在 $`\pi_\theta \approx \mu_{\theta_{\text{old}}}`$ 时是 sequence-level 真目标的一阶近似;任何破坏这点的 trick(length norm、重 clip)都引入 bias
> - **最完整公开 recipe(§2 DAPO)**:Clip-Higher + Dynamic Sampling + Token-Level Loss + Overlong Reward Shaping 四件套,后续 reasoning RL 几乎都在它基础上改
> - **不同路线(§3–§8)**:Magistral 动态 ε / MiMo 数据 re-sampling / Skywork adaptive entropy / DeepSeek-R1 ε=10 + 保留 KL — 都在解同一个 entropy collapse 问题,但取舍不同
> - **agentic RL 必先掌握 Part I**:所有 Part II–V 的工作都建立在这些算法基础上;先理解 reasoning RL 再做 agentic

## §1 Stabilizing Reinforcement Learning with LLMs: Formulation and Practices (MiniRL)

- **arXiv**: [2512.01374](https://arxiv.org/abs/2512.01374) (Qwen Team, Alibaba, Dec 2025)
- **HTML 全文**: <https://arxiv.org/html/2512.01374v2>
- **作者**: Chujie Zheng, Junrong Lin, Kai Dang, Bowen Yu, Yuqiong Liu, Hao Lin, An Yang, Jingren Zhou, Mingze Li, Huiqiang Jiang, Chencan Wu, Feng Hu, Junyang Lin
- **一句话总结**: 把 token-level 的 policy gradient 目标视为"真序列级 reward"的**一阶近似**，从而把 IS correction / clipping / Routing Replay 这些工程 trick 统一在同一个理论框架下，并给出 30B MoE 模型上的大规模稳定训练 recipe。

### 1.1 问题动机

LLM RL 中的真目标本是 sequence-level reward 的期望：

```math
J^{\text{seq}}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[R(x,y)]
```

但由于：
1. **推理引擎 ≠ 训练引擎**：rollout 用 inference engine（vLLM/SGLang，FP8 等），训练用 BF16 等不同 kernel；
   > **怎么理解**：现代 RL pipeline 是"两套软件栈跑同一份权重"。
   > - **Rollout 侧**用 vLLM/SGLang 做高吞吐生成：PagedAttention / FlashAttention / continuous batching / FP8 (甚至 INT8) 权重量化 / 自定义 CUDA kernel。目标是 throughput。
   > - **Training 侧**用 Megatron / FSDP / DeepSpeed 做反向传播：BF16 mixed precision、标准 attention kernel、固定 micro-batch。目标是数值稳定 + 梯度正确。
   >
   > **同样的权重 θ_old、同样的 prompt，两边算出来的 logits/logprobs 并不严格相等**，原因有四：
   > (a) **数值精度**：FP8/INT8 量化 vs BF16 全精度有 round-off 误差；
   > (b) **kernel 实现差异**：PagedAttention 的 reduction order 与 training attention 不同，浮点加法不结合，结果有 ulp 级别漂移；
   > (c) **batch-invariance 缺失**：vLLM continuous batching 下同一 token 在不同 batch 位置可能得到不同 logits（非确定性 reduction）；
   > (d) **MoE expert 路由**：见第 3 点。
   >
   > 所以即使 $`\pi_{\theta_{\text{old}}} = \mu_{\theta_{\text{old}}}`$ （权重相同），实际 logprob 比率 $`\pi_{\theta_{\text{old}}}(y_t)/\mu_{\theta_{\text{old}}}(y_t) \neq 1`$ ，会有 0.9–1.1 甚至更大的系统性漂移。PPO 假设比率在 1 附近的 trust region 被破坏，长期累积成 bias，最终 entropy 崩塌、训练发散。这就是 Eq.5 里"训练-推理 discrepancy"那一项的物理来源。
   >
   > **为什么必然是 entropy 崩塌**（而不是别的失败形式）？
   >
   > **一句话直觉**：推理端采的样本比训练分布更尖，没 IS 修正就等于"只给已经高概率的 token 加分",分布越用越窄,熵就崩了。
   >
   > **打个比方**：你在做民调，但采样器坏了——只去找态度明确的人，从不问犹豫的人。你拿这批样本训模型，模型以为"大家都很明确"，就把所有犹豫的可能性抹掉。下一轮采样器更偏，如此循环。
   >
   > 拆开看三股力，全都朝"更尖"一个方向走：
   > - **采样有偏**：推理引擎（FP8/PagedAttention 非确定性 reduction）把高概率舍入得更高、低概率压成 0，μ_θ_old 实际是从一个比 π_θ_old 更窄的分布采的。
   > - **梯度盲推**：没有 IS 修正，policy gradient 把 reward 直接乘到 $`\nabla \log \pi_\theta(y_t)`$ ，而被采到的 $`y_t`$ 大概率本来就高 → 梯度再推一把更高；那些被 μ 舍入到 0 的 token 根本进不了 batch，永远拿不到正向梯度——**词表在 RL 视角下被悄悄裁剪了**。
   > - **Clip 不对称**：ratio 系统性 > 1 时，正 advantage 撞 $`1+\epsilon_{\text{high}}`$ 被压住（想涨的涨不动），负 advantage 照跌（想跌的继续跌）。**净效果"只缩不扩"**。
   >
   > 三股一旦启动就互相喂养：分布变尖 → 下一轮 rollout 更窄 → batch 内 token 多样性更低 → group advantage 方差变 0（GRPO 直接没梯度）→ 残留梯度仍指向"再尖一点" → entropy 几步内崩到接近 0，模型只输出一两种 pattern，reward 卡住。监控上的典型特征：**entropy 在 50–200 步内断崖式下跌，clip ratio 飙到 20%+，KL(μ_old ‖ π_old) 同步爆涨**——三个信号同时出现基本就是 train-infer 不一致没修。
   >
   > **对策**：token-level IS correction + TIS（截断极端比率）+ FP8 训推统一（部分团队）/ batch-invariant kernel（Thinking Machines 2025 deterministic inference 路线）。
2. **batch 拆 mini-batch**：大 batch 拆成 N 个 mini-batch 多步更新时，rollout 时的策略 μ_θ_old 与当前 π_θ 已经发散；
3. **MoE 路由不一致**：MoE 模型中 inference 和 training 各自的 expert 路由可能不同。

直接优化 sequence-level 目标在数值上不可行（序列似然量级巨大），所以业界实际优化的是 **token-level surrogate**，但缺乏理论说明它为什么 work、什么时候 work。

### 1.2 核心理论：Token-Level 是 Sequence-Level 的一阶近似

#### Sequence-level 经过 IS 后的形式

```math
J^{\text{seq}}(\theta) = \mathbb{E}_{y \sim \mu_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(y|x)}{\mu_{\theta_{\text{old}}}(y|x)} R(x,y)\right]
```

#### Token-level surrogate（带 stop-gradient）

```math
J^{\text{token}}(\theta) = \mathbb{E}\left[\sum_t \text{sg}\!\left[\frac{\pi_\theta(y_t \mid x,y_{\lt t})}{\mu_{\theta_{\text{old}}}(y_t \mid x,y_{\lt t})}\right] \cdot R(x,y) \cdot \log \pi_\theta(y_t \mid x,y_{\lt t})\right]
```

关键结论：当 $`\pi_\theta \approx \mu_{\theta_{\text{old}}}`$ 时

```math
\nabla_\theta J^{\text{seq}}(\theta) \approx \nabla_\theta J^{\text{token}}(\theta)
```

也就是说 token-level 目标只在策略相近时才与真目标一阶等价。**任何让二者偏离的 trick（如 length normalization）都会破坏一阶近似的 validity**。

#### IS 权重的两源分解（Eq. 5）

```math
\frac{\pi_\theta(y_t|\cdot)}{\mu_{\theta_{\text{old}}}(y_t|\cdot)} = \underbrace{\frac{\pi_{\theta_{\text{old}}}(y_t|\cdot)}{\mu_{\theta_{\text{old}}}(y_t|\cdot)}}_{\text{训练-推理 discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t|\cdot)}{\pi_{\theta_{\text{old}}}(y_t|\cdot)}}_{\text{policy staleness}}
```

- **训练-推理 discrepancy**：训练 vs 推理引擎的数值差异（kernel、batch-invariance、FP8 vs BF16、MoE 路由不同）；
- **Policy staleness**：rollout 策略与当前训练策略的差异（来自 mini-batch 切分 / 异步 RL）。

### 1.3 三大稳定化技巧的统一解释

#### Importance Sampling Correction

token-level IS 权重 $`r_t = \pi_\theta(y_t|\cdot)/\mu_{\theta_{\text{old}}}(y_t|\cdot)`$ 修正两类 gap。
- **去掉 IS correction → rapid training collapse + entropy 急剧下降**。
- IS 权重外包 stop-gradient，梯度不流经权重本身。

#### Clipping (PPO-style 不对称)

```math
M_t = \begin{cases} 0 & \hat A\gt 0 \text{ 且 } r_t \gt  1+\epsilon_{\text{high}} \\ 0 & \hat A\lt 0 \text{ 且 } r_t \lt  1-\epsilon_{\text{low}} \\ 1 & \text{otherwise} \end{cases}
```

- 超出比率边界的 token 不更新，**抑制 policy staleness**，使 $`\pi_\theta`$ 不偏离 $`\mu_{\theta_{\text{old}}}`$ 太远，保住一阶近似。
- 论文取 $`\epsilon_{\text{high}}=0.27`$ 、 $`\epsilon_{\text{low}}=0.2`$ （即 r 超过 1.27 或低于 0.8 时 mask 掉）。

#### Routing Replay (R2 / R3) — 针对 MoE

MoE 模型中分母还要带 expert 路由：

```math
\frac{\pi_\theta(y_t \mid x,y_{\lt t}, e^\pi_t)}{\mu_{\theta_{\text{old}}}(y_t \mid x,y_{\lt t}, e^{\mu_{\text{old}}}_t)}
```

路由不一致 → 同时放大 train-infer discrepancy 和 staleness。

- **Vanilla Routing Replay (R2)**：训练时强制使用训练引擎旧策略的 expert 路由 $`e^\pi_{\text{old},t}`$ → 只缓解 policy staleness。
- **Rollout Routing Replay (R3)**：训练时直接 replay 推理引擎 rollout 时的 expert 路由 $`e^{\mu_{\text{old}},t}`$ → 同时缓解两个 gap。
- **代价**：对目标策略引入 bias（target policy 不再用其"自然"路由）。off-policiness 越大，这个 bias 越值得付。

### 1.4 MiniRL 算法

完整目标：

```math
J_{\text{MiniRL}}(\theta) = \mathbb{E}\!\left[\sum_t M_t \cdot \text{sg}\!\left[\frac{\pi_\theta(y_t \mid x,y_{\lt t})}{\mu_{\theta_{\text{old}}}(y_t \mid x,y_{\lt t})}\right] \cdot \hat A(x,y) \cdot \log \pi_\theta(y_t \mid x,y_{\lt t})\right]
```

- $`\hat A(x,y) = R(x,y) - \mathbb{E}_{y'}[R(x,y')]`$ ，**group-normalized** advantage（同 prompt 内做基线）；
- **没有 length normalization**（即不除以 $|y|$ ）；
- IS 权重外加 **Truncated Importance Sampling (TIS)**，截断阈值 = 5（Yao et al. 2025）；
- $`M_t`$ 为不对称 clipping mask。

### 1.5 实验设置

| 项目 | 配置 |
|---|---|
| 模型 | Qwen3-30B-A3B (MoE, cold-start 变体) |
| 训练精度 | BF16 训练 + FP8 推理 |
| 任务 | 数学推理，binary reward (对/错) |
| 数据 | 4,096 道精选数学题 |
| 评测 | HMMT25 + AIME25 + AIME24（共 90 题） |
| mini-batch size | 1,024 responses |
| global batch | 1,024–8,192（控制 off-policiness） |
| max gen length | 32,768 tokens |
| 总算力 | "hundreds of thousands of GPU hours" |
| 监控指标 | training reward、token-level entropy $`H[\pi_\theta]`$ 、 $`D_{\text{KL}}[\mu_{\theta_{\text{old}}} \,\Vert\, \pi_{\theta_{\text{old}}}]`$ |

### 1.6 关键实验结论

**On-policy (gbs = mbs = 1024)**
- **MiniRL（即 plain policy gradient + IS correction）取得最佳性能与稳定性**。
- 去掉 IS correction → 训练崩塌、entropy 暴跌。
- 加上 length normalization → 性能次优（破坏一阶近似）。
- on-policy 下加 R3 → 无收益甚至降点。

**Off-policy（按 N = gbs/mbs 划分）**

| 设置 | 现象 |
|---|---|
| N=2 (gbs=2048) | R2 优于 R3（staleness 小，R3 的 bias 不值得） |
| N=4 (gbs=4096) | R3 反超 R2 |
| N=8 (gbs=8192) | R2 无法稳住；只有 R3 可维持稳定训练 |

**Cold-start 鲁棒性**：从 Qwen3-Max / DeepSeek-R1 / gpt-oss-120b 三种完全不同的冷启动初始化出发，只要训练稳定，**长时间 RL 后性能趋于一致**。
> 关键 takeaway：**稳定的训练 > 初始化的好坏**。

### 1.7 工程层面 take-home

1. **永远带 token-level IS correction**，并在 IS 权重外加 stop-gradient；建议同时加 TIS（threshold≈5）防极端比率。
2. **不要 length-normalize**：它破坏一阶近似，是一个看起来无害但实际有 bias 的项。
3. **不对称 clipping** ($`\epsilon_{\text{high}}\gt \epsilon_{\text{low}}`$ ，例如 0.27/0.2) 配合 IS 权重可有效抑制 staleness。
4. **MoE 场景必须处理 routing**：几乎 on-policy（N≤2）→ vanilla R2；显著 off-policy（N≥4）→ rollout R3。
5. **监控三大量**：training reward、entropy、 $`D_{\text{KL}}[\mu_{\text{old}} \| \pi_{\text{old}}]`$ 。entropy 骤降 + KL 增大几乎一定是要崩。
6. **scale RL 的关键不是冷启动也不是 on-policy/off-policy 哲学**，而是能否让 token-level surrogate 始终维持在它对 sequence-level 真目标的一阶近似域内。

---

## §2 DAPO (ByteDance Seed × Tsinghua AIR, 2025-03)

- **arXiv**: [2503.14476](https://arxiv.org/abs/2503.14476) · [项目主页](https://dapo-sia.github.io/) · [GitHub](https://github.com/BytedTsinghua-SIA/DAPO) · [verl recipe](https://verl.readthedocs.io/en/latest/algo/dapo.html)
- **定位**: 首个完整公开 RL 训练系统细节的工作；Qwen2.5-32B base 上 AIME24 50 分，仅用 R1-Zero 一半训练步数。

### 四大核心 trick（精确配置）

1. **Clip-Higher（解耦上下 clip）**： $`\epsilon_{\text{low}}=0.2`$ ， $`\epsilon_{\text{high}}=0.28`$ 。提升低概率 token 的上行空间，缓解 entropy collapse。
2. **Dynamic Sampling**：过滤 batch 中 accuracy=0 或 accuracy=1 的 prompt（无梯度），持续采样直至 batch 填满"有梯度"样本。
3. **Token-Level Policy Gradient Loss**：用 `1/Σ|o_i|` 跨样本对 token 求和，而非 GRPO 的 sample-level `(1/G)·(1/|o_i|)` 嵌套平均；让长序列对梯度有应得权重。
4. **Overlong Reward Shaping**： $`L_{\max}=16384`$ ， $`L_{\text{cache}}=4096`$ 软惩罚窗（线性从 0 衰减到 -1），硬截断 $`L_{\text{hard}}=20480`$ ；也提供 "overlong filtering" 变体（直接 mask loss）。

### 关键超参

- **LR = 1e-6**（constant, AdamW），20 步 linear warm-up
- Prompt batch = 512，**16 rollouts/prompt**，mini-batch = 512（每轮 16 次梯度更新）
- **β_KL = 0**（完全移除 KL 项）
- Rollout: T=1.0, top_p=0.7；评估用 avg@32
- 可抄 recipe：verl 框架的 `recipe/dapo` 目录直接对应

---

## §3 Magistral (Mistral AI, 2025-06)

- **arXiv**: [2506.10910](https://arxiv.org/abs/2506.10910)
- **定位**: Mistral 首个推理模型；纯 RL 从 Mistral Medium 3 直接训练（Magistral Medium），SFT+RL 路线给 Magistral Small (24B 开源)。

### GRPO 改造（与 DeepSeek/DAPO 对照）

- **完全移除 KL 罚项**：策略本就大幅偏离参考模型，保留 ref model 是"无意义的计算"。
- **Clip-Higher**: $`\epsilon_{\text{high}} \in [0.26, 0.28]`$ 动态调整以维持 group entropy 稳定。
- **去掉 group std 归一化**: $`\hat A_i = r_i − \mu`$ ，避免 easy/hard 题被 σ 误放大。
- **Minibatch 内 advantage 归一化** (Andrychowicz et al. 2020 风格)： $`\hat A_{\text{norm}} = (\hat A_i − \hat A_{\text{mean}})/\hat A_{\text{std}}`$ 。
- **Loss 归一化用 group 内总长度**（防止组内长度偏差）。
- **过滤 non-diverse groups**：全对/全错的 group 直接剔除（与 DAPO dynamic sampling 同源）。
- **不用 entropy bonus**：尝试过但跨域不稳定；用 $`\epsilon_{\text{high}}`$ 控熵更稳。

### Reward 四维

format / correctness (math 严格 verifier, code) / length (soft penalty) / **language consistency** (跨语言一致奖励)。

### Multi-stage 训练 schedule

- 生成长度 $`L_{\max} − L_{\text{cache}}`$ 从 **16k → 24k → 32k** 递增
- n_batch 同步从 **8k → 4k → 2k** 递减（控 KV cache 显存）

### Async 系统

- Generators 持续以最大吞吐生成不等待 trainer
- 权重通过 **NCCL broadcast** 推给 generators，不丢弃 in-flight 序列；KV cache 可轻度过期，靠 off-policy IS 校正补救
- **三个 batch 尺度**：n_batch（更新生成器权重前的累计样本）/ n_minibatch（一次优化步样本）/ n_async（并发生成数）

### 意外发现

纯文本 RL 训练能迁移到 multimodal benchmark（MMMU），reward 与输出长度呈对数关系。

---

## §4 MiMo / MiMo-7B-RL (Xiaomi LLM-Core, 2025-05)

- **arXiv**: [2505.07608](https://arxiv.org/abs/2505.07608) · [GitHub](https://github.com/XiaomiMiMo/MiMo)
- **定位**: 7B 小模型反超 32B；MiMo-7B-RL-Zero 在 math/code 上比 32B base RL 后还强。

### 核心工程 trick

- **Test-Difficulty-Driven Code Reward**: 按 pass-rate 分组 test cases，提供"严格奖励"（必须通过本组与较低组）与"软奖励"（组内分数均分到 test）双模式，解决稀疏奖励。
- **Easy Data Re-sampling**: 完美 pass 的题进入"easy pool"，以 **α=10%** 概率回采，稳定训练。
- **Removal of KL Loss**: 删 KL 项后训练更稳，且不影响稳定性。
- **Clip-Higher**: 抬高 $`\epsilon_{\text{high}}`$ （具体值未披露），保留 $`\epsilon_{\text{low}}`$ 。
- **Seamless Rollout Engine**: continuous rollout + async reward computation + early termination，训练 **2.29×**、验证 **1.96×** 加速。

### 关键超参

- LR = **1e-6**, batch = **512**, actor mini-batch = **32**, 16 grad updates / iteration
- Max seq len = **32768**（0530 版扩到 48k）
- T=1.0, top_p=1.0
- 数据：100K math + 30K code = **130K verifiable**

### RL 数据筛选 pipeline

1. 过滤 advanced reasoning model 都解不出来的（太难或错答案）
2. 剩下的用 SFT 版本 MiMo-7B rollout 16 次，pass rate > 90% 的扔掉（太简单）
3. 结果：移除约 50% 简单题，math 留 100K

### 0530 升级

SFT 数据 500K→6M，RL 窗口 32K→48K，AIME24 超 DeepSeek-R1。

---

## §5 Skywork-OR1 / MAGIC (Skywork AI, 2025-05)

- **arXiv**: [2505.22312](https://arxiv.org/abs/2505.22312) · [GitHub](https://github.com/SkyworkAI/Skywork-OR1)
- **定位**: 全面公开 **entropy collapse 的根因分析与对策**；32B 在 AIME24/25 超 DeepSeek-R1 与 Qwen3-32B。

### MAGIC pipeline 核心 trick

- **Multi-stage context length curriculum**: **8K → 16K → 32K**，每阶段收敛后切换；同时丢弃上阶段已 acc=1 的 prompt（online filtering）。
- **Token-level loss without length normalization**: 去掉 $`1/|y_{ij}|`$ （与 DAPO 一致方向）。
- **Adaptive entropy control**: 目标熵 **tgt_ent = 0.2**，根据当前熵与目标差动态调整 $`\alpha_k`$ 系数。
- **Strict on-policy training**: 7B/32B 用 **1 grad step / rollout**（慢但抑制 entropy collapse 最有效）；Math-7B 用 2 steps + adaptive entropy 补偿。
- **Rejection sampling on zero-advantage groups**：组内全部 advantage 为 0 时整组剔除。
- **不用 KL loss、不用 advantage mask**（包括对 truncated 响应）。

### 关键超参

- GRPO group size **M = 16**, batch 64–256, mini-batch 32–128
- Clip $\epsilon = 0.2$ （用 clip-higher 控熵，具体值未列）
- Temperature **τ = 1.0**（τ=0.6 会导致 entropy 提前 collapse）
- 评估提倡 **Avg@K** 替代 Pass@1
- 32 张 H800，2700 steps 即达 7B 终版水平

---

## §6 DeepSeek-R1 (DeepSeek, 2025-01)

- **arXiv**: [2501.12948](https://arxiv.org/abs/2501.12948) · [Nature](https://www.nature.com/articles/s41586-025-09422-z)
- **定位**: 引爆 reasoning RL 浪潮的 "GRPO + 规则奖励" 开山之作。

### First RL Stage (R1-Zero) 关键披露超参

- LR = **3e-6**, **β_KL = 0.001**
- **GRPO clip $\epsilon = 10$**（极大，几乎不裁剪）
- **16 rollouts/prompt**, max output **32768 tokens**
- 32 unique questions × 16 = **batch 512**
- Rollout T = **1.0**

### 多阶段 pipeline

cold-start SFT → 推理 RL → rejection sampling 重生成 SFT 数据 + V3 通用数据 → 第二轮 RL（覆盖 helpfulness/harmlessness）。

R1-Zero 不经 SFT 直接 RL，AIME24 pass@1 15.6%→71.0%。

**未公开**: 第二阶段 RL 的奖励权重、reward model 细节、对齐 hyperparams。

---

## §7 Qwen3 (Alibaba Qwen Team, 2025-05)

- **arXiv**: [2505.09388](https://arxiv.org/abs/2505.09388)
- **定位**: thinking / non-thinking 双模态融合 + 四阶段 post-training，明确转向 agentic RL 方向。

### 四阶段 pipeline

1. **Long-CoT cold start**
2. **Reasoning RL (GRPO)**: 仅 **3995 query-verifier pairs**，170 steps，Qwen3-235B-A22B AIME24 70.1→85.1
3. **Thinking Mode Fusion**: `<think>`/`</think>` + `/think`/`/no_think` flags + **thinking budget** 机制
4. **General RL**: 5 类能力 (Instruction Following / Format / Preference / **Agent Ability** / Specialized)；rule-based + model-based 混合 reward

### 训练经验披露

- **大 batch size + 高 rollouts/query + off-policy training** 三者组合改善样本效率
- 关键是"控熵"（让 entropy 平稳或缓增）
- 训练全程**无需手动调超参**

### Agentic RL（Stage 4 核心）

允许 RL rollout 中**完整 multi-turn 环境交互**（agent 调用工具）。小模型走 Strong-to-Weak Distillation（off-policy + on-policy）路线，不走完整四阶段。

**未公开**: 具体 LR、batch、ε 数值；社区复现 (ms-swift) 用 LR=1e-6, num_generations=16, T=1.0, max_completion=4096。

---

## §8 GLM-4.5 / GLM-4.6 (Z.ai / Zhipu, 2025-08)

- **arXiv**: [2508.06471](https://arxiv.org/abs/2508.06471) · [GitHub](https://github.com/zai-org/GLM-4.5) · [slime](https://github.com/THUDM/slime)
- **定位**: 把 Agentic / Reasoning / Coding (ARC) 三类能力统一训练；自研 RL infra "slime" 开源。

### RL 训练 trick

- **Reasoning RL**: 单阶段 RL @ **64K context window**（优于渐进式 schedule）；难度课程；**dynamic sampling temperature** + **adaptive clipping**。
- **Agentic RL**: 信息搜索 QA + SWE 任务；人在环 + 网页内容混淆生成合成数据；coding 用真实 SWE 反馈。
- **专家模型分头训练 → 统一自蒸馏** 的 post-training 两段式。
- 重新设计 function-call 模板以最小化字符转义，提升 tool-use 鲁棒性。

### slime infra（agentic 关键）

- **Hybrid 同步/异步**: reasoning 任务用 colocated 同步模式；agentic 任务用 disaggregated 异步模式
- **FP8 rollout** 加速（GLM-4.5 FP8 部署只需 8×H100）
- Megatron-LM 训练 + **SGLang** 高吞吐 rollout
- Docker 沙箱 + decoupled training loop 支撑 long-horizon agent
- 统一 HTTP endpoint + centralized data pool 接入异构 agent framework

GLM-4.6: tool-calling 准确度进一步提升，拒识未知工具、最小化臆造参数。

---

## §9 Seed-Thinking-v1.5 / Doubao 1.5 Thinking Pro (ByteDance Seed, 2025-04)

- **Seed blog**: [bytedance-seed/seed-thinking-v1.5](https://seed.bytedance.com/en/blog/bytedance-s-latest-thinking-model-seed-thinking-v1-5-technical-details-disclosed) · [GitHub](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5)
- **定位**: 200B 总参 / 20B 激活 MoE，AIME24 86.7；BeyondAIME 数据集首次提出。

### RL 训练披露

- **三轨数据引擎**: verifiable / generic / hybrid，配合 online data adaptation 动态调分布
- 算法侧: **value pre-training** + **decoupled GAE**（控制长 CoT 推理"断裂"）
- **双轨 reward system**: 可验证任务用智能逻辑验证；不可验证任务用 pairwise comparison
- SFT: 400K 实例（300K verifiable + 100K non-verifiable）

### Infra

- 优化的 **HybridFlow** 编程模型（即 verl 学术对应）
- **Streaming reasoning system**: 异步管理跨模型版本的"部分完成"生成 → **3× 更快 RL 周期**
- Tensor / Expert / Sequence 三层并行

---

## §10 Llama 3 / Llama 4 RL 部分

- **Llama 3** ([arXiv 2407.21783](https://arxiv.org/abs/2407.21783))
  - **DPO 而非 PPO**（PPO 计算贵且 IFEval 表现更差）；LR = **1e-5**, **β = 0.1**
  - **6 轮迭代**: SFT → rejection sampling → DPO
  - 稳定 trick: 对 header/termination special tokens 在 loss 中 mask；chosen 序列加 **NLL loss 系数 0.2**

- **Llama 4** (2025-04，无完整 RL paper)
  - 三步 pipeline: 轻量 SFT (难 prompt 上的小集合) → **online RL** (大规模) → 持续 refine
  - 混用 PPO-RLHF + DPO + rejection sampling，加 **KL 罚项**
  - 官方定位为 **non-reasoning model**（不长 CoT）；RLVR 可能局部使用但非主体

---

> **【2025H2–2026 新增:新算法浪潮】** §10A–§10L 是 DAPO 之后涌现的新一批 reasoning RL 算法/配方。它们的共同主题是**重新审视 PPO/GRPO 的两个核心组件——importance ratio 的粒度(token vs 序列)和 loss 归一化**,以及把 RL 当成有 scaling law 的系统工程来做。配合[共识 14](#共识-14--is-粒度之争token-级tis--序列级gspo--裁权重不丢-tokencispo2025h2-新浪潮)与 **§49 算法谱系**一起读。

## §10A GSPO — Group Sequence Policy Optimization (Qwen, 2025-07)

- **arXiv**: [2507.18071](https://arxiv.org/abs/2507.18071) · [Qwen blog](https://qwenlm.github.io/blog/gspo/) · 作者 Chujie Zheng, Bowen Yu, Junyang Lin 等(Qwen 团队)
- **定位**: 用于 **Qwen3**;把 importance ratio 从 token 级换到**序列级**,顺手免掉 MoE RL 的 Routing Replay。是 GRPO 之后最重要的算法改动之一。

### 核心:序列级 IS 比率(对 GRPO 的根本批判)

GRPO 的 per-token ratio 问题:每个 next-token 分布**只有 1 个样本**,IS 在 $`N{=}1`$ 上起不到分布校正,只注入高方差噪声 → 沿序列累积 → 被 clip 放大 → **崩溃往往不可逆**。GSPO 的原则:**"优化目标的单位应当与 reward 的单位匹配"**——reward 是序列级,IS 也该是序列级。

```math
s_i(\theta)=\left(\frac{\pi_\theta(y_i|x)}{\mu_{\theta_{\text{old}}}(y_i|x)}\right)^{1/|y_i|}=\exp\left(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t})}{\mu_{\theta_{\text{old}}}(y_{i,t})}\right)
```

即对 per-token log-ratio 取**几何平均(长度归一)**,clip 也在序列级:$`\min(s_i\hat A_i,\ \mathrm{clip}(s_i,1-\epsilon,1+\epsilon)\hat A_i)`$。**GSPO-token 变体**:为多轮/per-token 优势定制,用 stop-gradient 技巧让数值上等价于 GSPO 但允许逐 token advantage。

### MoE 红利:免 Routing Replay(对照 MiniRL R3 §1.3)

- **量化证据**:Qwen3-30B-A3B 上,每次梯度更新后对同一 rollout,**约 10% 激活专家与旧策略不同**,越深越严重 → token 级比率剧烈波动失效。
- GRPO 因此**必须**上 Routing Replay(缓存并重放 $`\mu_{\text{old}}`$ 的路由,即 MiniRL R3 那一类),代价是内存/通信 + 限制 MoE 实际容量。
- **GSPO 只依赖序列似然,对单 token 路由漂移不敏感 → 彻底不需要 Routing Replay**。原文:"GRPO necessitates Routing Replay … GSPO has obviated the need"。**这是 MoE agentic RL 的一条捷径。**

### 超参 / 反直觉

- clip ε:GSPO 约 **3e-4 / 4e-4**(数值上比 GRPO 的 0.2/0.27 小一个数量级,因比率定义不同);从 Qwen3-30B-A3B-Base 起,每 batch 4 个 mini-batch。
- **反直觉**:GSPO 被 clip 的 token 比 GRPO **高两个数量级**,用更少 token 估梯度却更高效——作者据此判定 GRPO 的 token 级估计本身就噪声大。
- **额外红利(与 §44 强相关)**:序列级似然对训练引擎(Megatron)vs 推理引擎(SGLang)的精度差**容忍度高得多**,有望直接用推理引擎的 logprob 免重算——利好 partial rollout / 异步。

### 边界(必写)

- **机制归因有争议**:follow-up [2509.24203](https://arxiv.org/abs/2509.24203) 认为真正起作用的是"序列级 clip 当正则",而非序列级 IS。
- **高 staleness 下会崩**:INTELLECT-3(§41E)async-8 实测 **GSPO reward 崩溃**——序列级 IS 在极端 off-policy 时反不如双侧 clip。
- benchmark 具体数值:论文主要给曲线(Fig.1–3),**精确分数仅图示**。

---

## §10B CISPO + MiniMax-M1 — 裁权重不丢 token + FP32 LM head 精度坑 (MiniMax, 2025-06)

- **arXiv**: [2506.13585](https://arxiv.org/abs/2506.13585) · 模型 MiniMax-M1(456B 总参 / 45.9B 激活,hybrid MoE + Lightning Attention,原生 **1M** context)
- **定位**: CISPO 是"裁 IS 权重、不丢 token"路线的代表;MiniMax-M1 报告里那段 **train-infer 精度失配**分析是本文档 §44 的金矿。

### CISPO:裁权重而非裁 token

```math
\mathcal J_{\text{CISPO}}=\mathbb E\!\left[\frac{1}{\sum_i|o_i|}\sum_i\sum_t \mathrm{sg}(\hat r_{i,t})\,\hat A_{i,t}\,\log\pi_\theta(o_{i,t})\right],\quad \hat r_{i,t}=\mathrm{clip}(r_{i,t},1-\epsilon^{IS}_{\text{low}},1+\epsilon^{IS}_{\text{high}})
```

- PPO/GRPO 的 clip 会把被裁 token 的**梯度直接置零(丢 token)**;CISPO 改为裁 IS 权重,**所有 token 都保留梯度**。
- **关键动机:fork token**——"However / Wait / Aha / Recheck"这类低概率、高 ratio 的推理分叉点,在 PPO/GRPO 下"第一次 on-policy 更新就被裁掉",而它们对稳熵、支撑长程探索最关键。在"每 batch 16 轮 off-policy 更新"设置下,**DAPO 的 clip-higher 效果差,CISPO 才稳**。
- 只调 $`\epsilon^{IS}_{\text{high}}`$、**不设下界**;去掉权重裁剪即退化为标准 policy gradient。
- **2x 加速声称**(Qwen2.5-32B-base 受控实验):仅用 50% 步数匹配 DAPO,同步数同时优于 GRPO/DAPO——属单设置受控结果。

### ⭐ Train-Infer 精度失配:坑在 LM head 不在 attention(§44 必引)

- **现象**:RL 中 rollout token 在训练模式 vs 推理模式下概率显著不同,**直接导致 reward 涨不上去**。
- **架构相关**:只在他们的 hybrid Lightning-Attention 架构出现,**小 dense + softmax-attention 模型不出现**。
- **逐层定位**:误差主要来自 **LM head 的 high-magnitude activations**。
- **修复**:**把 LM 输出 head 精度提到 FP32**,使训练/推理概率相关性 **0.9x → 0.99x**,恢复 reward 增长。
- **独立佐证**:Meta ScaleRL(§10F)同样发现 FP32 LM head 把渐近线从 0.52→0.61——**这是文档 FP8/BF16 主题最硬的两个独立数字**。

### 其他硬核工程坑

- **AdamW 默认配置不收敛**:RL 梯度幅度跨 1e-18~1e-5、多数 <1e-14,VeRL 默认 (β2=0.999, eps=1e-8) **不收敛**,改 **β1=0.9, β2=0.95, eps=1e-15**。
- **重复早停**:连续 3000 token 每个概率 >0.99 即 halt 生成。
- **80K 长度扩展的 pattern collapse**:扩到 80K 时尾部乱码,根因**负样本长度增长远快于正样本**(负梯度失衡);修复 = 早停 + sample/token-level norm 结合 + 降 grad clip 和 $`\epsilon^{IS}_{\text{high}}`$;长度分阶段 40K→48K→…→80K。
- **GenRM 长度偏置**:离线缓解常失败,最终靠 **RL 中在线监控长度偏置 + 一旦检测到 length-seeking 就立刻重校准 GenRM**。
- **算力**:512×H800,3 周,租赁约 **\$0.53M**(厂商估算);M1-80k SWE-bench Verified 56.0。

---

## §10C GMPO — Geometric-Mean Policy Optimization (2025-07)

- **arXiv**: [2507.20673](https://arxiv.org/abs/2507.20673) · [GitHub](https://github.com/callsys/GMPO)
- **定位**: token reward 的**几何平均**版 GRPO,对离群 IS 比率更鲁棒。

**机制**:GRPO 最大化 token reward 的算术平均,单个极端 ratio 就能把 token 梯度顶飞;GMPO 改用**几何平均**(= log-ratio 的算术平均,在 log 空间实现),由 AM-GM 不等式保证目标值范围更窄、方差更低。**token 级 clip**(反对 DeepSeek-R1 的序列级 clip,后者"一触发就把整条序列梯度置零"),但范围放宽到 **$`(e^{-0.4}, e^{0.4})`$**(远宽于 GRPO 0.8/1.2、DAPO 0.8/1.28)。

**结果**:R1-Distill-Qwen-7B 上比 GRPO **平均 +4.1%(63.4 vs 59.3)**,跨 5 个数学 benchmark;MoE(Qwen3-32B)MATH500 +2.1%;多模态 +1.4%。消融:去长度归一 −0.7%;CountDown 上 MoE-GRPO 约 250 步崩、GMPO 稳(与 GSPO 的 MoE 不稳观察互证)。**LR 未披露**。

---

## §10D Dr.GRPO — R1-Zero 式训练的两个偏置(共识 5 的理论基础)(2025-03)

- **arXiv**: [2503.20783](https://arxiv.org/abs/2503.20783) ("Understanding R1-Zero-Like Training", COLM 2025)
- **定位**: [共识 5](#共识-5--长度归一化--overlong-filtering-是反共识陷阱)反 length-norm 的**精确数学来源**。GRPO 的有效 advantage = 无偏 advantage 的重加权,两个归一项各引一个偏置。

**偏置 1 — Response-length bias(来自 $/|o|$)**:按序列长度归一 → **长的错误回答被欠惩罚**(penalty 被 $1/|o|$ 削弱);等价说法是 loss ≈ advantage/length,**偏好"短且对",不惩罚"长且错"** → 错误回答越训越长。DAPO 的 token-level **减轻但未消除**。

**偏置 2 — Question-difficulty bias(来自 $`/\mathrm{std}`$)**:按 $`\mathrm{std}(r)`$ 归一 → 给**低方差的题(极易/极难)不成比例的权重**。

**修复("GRPO Done Right")**:去掉 per-response 长度归一(改用常数 $L$)+ 去掉 per-group std 归一。效果:防止越写越长、提 token 效率、准确率不降。附带发现:**主流开源 PPO 实现普遍误用 length 归一**(`loss.mean(-1)`),源自预训练惯性。极简 recipe:Qwen2.5-Math-7B,8×A100 / 27h 达 SOTA。

> **反例 / 边界(Lambert 的 caveat,§48 详)**:去 std 会**下调"9 错 1 对"里那个罕见但关键的正确样本的权重**;且 Dr.GRPO 虽输出更短,**未展示更好的下游最终性能**。所以共识 5 的"别 len-norm"在"用 token-level 而非完全去掉归一"(DAPO 路线)上更稳——这也是 Lambert 本人的偏好。

---

## §10E Lite-PPO — "Tricks or Traps?" 极简两件套反超 DAPO (ROLL 团队, 2025-08)

- **arXiv**: [2508.08221](https://arxiv.org/abs/2508.08221) · 已并入 HuggingFace TRL
- **定位**: 系统消融 Normalization / Clipping / Masking / Loss-Aggregation 四方面,结论:**很多 DAPO trick 非必要,极简两件套即够**——共识 3/5 的重要反例。

**Lite-PPO 两件套**:① **混合优势归一化**(group 均值 + **batch** 标准差)——一组内几乎全对/全错时 group std 极小、除它会梯度爆炸,batch 级 std 更稳;② **token-level loss** 聚合。

**反直觉发现(全是反例,价值极高)**:
- **clip-higher 可能无益甚至有害**:base 模型 clip 率本就 ~0.003,抬上界几乎无效;只有 aligned 模型显著减缓熵崩。
- **存在 scaling 依赖**:4B 上 $`\epsilon_{\text{high}}{=}0.32`$ 最优,**8B 上 0.28 最优**,小模型成立、大模型不成立。语言学上:0.2 主要裁连接词,0.28 转向裁功能词、放开推理结构探索。
- **token-level vs sequence-level loss 取决于模型**:token-level 在 base 更有效,**aligned 模型多数数据集上 sequence-level 反而胜**——损失聚合无统一最优。
- Lite-PPO 还**丢掉 overlong filtering**(它限制小模型生成长尾)。

---

## §10F ScaleRL — "The Art of Scaling RL Compute" (Meta, 2025-10)

- **arXiv**: [2510.13786](https://arxiv.org/abs/2510.13786) · **>400,000 GPU-hours**,旗舰单次 100,000 GPU-hours(GB200)
- **定位**: RL 的 **scaling law** 论文。把"哪些设计改天花板、哪些只改算力效率"分得清清楚楚——做大规模 RL 前必读。

### Sigmoidal 算力-性能曲线

放弃幂律,用**饱和 sigmoid**拟合 pass-rate 对 log-compute:$`R_C-R_0=(A-R_0)\cdot\frac{1}{1+(C_{\text{mid}}/C)^B}`$。**A** = 天花板,**B** = 算力效率/陡峭度,**C_mid** = 半增益算力中点。拟合从 ~1.5k GPU-h 后开始。

### 推荐配方(ScaleRL)

PipelineRL(off-policy **k=8**)+ **CISPO** loss(选它而非 GSPO/DAPO)+ prompt-level loss 聚合 + **batch-level** 优势归一 + **FP32 logits/LM head** + interruption-based 长度控制 + zero-variance filtering + **No-Positive-Resampling**(永久移除 pass-rate≥0.9 的 prompt)。

### ⭐ 最有价值的区分:改天花板 vs 只改效率

- **改渐近线 A(天花板)**:loss 类型(CISPO/GSPO > DAPO)、**FP32 精度(A: 0.52→0.61)**、zero-variance filtering、No-Positive-Resampling、更大 batch、更长生成、MoE 规模。
- **只改效率 B(不动天花板)**:PipelineRL vs PPO-off-policy、loss 聚合、优势归一、数据课程、length penalty。

### 反直觉 / 反对建议

- **"Bitter Lesson":小算力下最好的方法外推到大算力可能落败——别只凭早期表现选配方。**
- **反对 DAPO 作 loss**(渐近线低、效率差,B=1.77 vs CISPO 2.01);CISPO 对 clip 超参更鲁棒。
- **别丢"看似冗余"的组件**:FP32 在 8B dense 的 leave-one-out 里显得边际,但在 GRPO/DAPO loss、MoE 上提供稳定性。
- 下游 benchmark 不适合研究 scaling,要用 in-distribution 留出验证集。
- **立场差异(并列,非定论)**:Meta 这里结论 **CISPO > GSPO > DAPO**,而 Qwen 主推 GSPO——两家在各自设置下的结论,谁对谁错未定。

---

## §10G ProRL — 长时 RL 能否扩展 base 能力边界(对共识 12 的挑战)(NVIDIA, 2025-05)

- **arXiv**: [2505.24864](https://arxiv.org/abs/2505.24864)(+ ProRL v2 blog) · 模型 Nemotron-Research-Reasoning-Qwen-1.5B
- **定位**: 直接挑战[共识 12](#共识-12--entropy-collapse-是头号杀手治法多样)/rStar2 的 "RL 上限 ≈ base" 论。主张**长时 RL 能发掘 base 采样不到的新策略**。

**核心争议**:rStar2/[2504.13837](https://arxiv.org/abs/2504.13837) 说 RL 只锐化 base 分布;ProRL 反驳——存在任务 **base 无论采多少次都给不出正确解,而 RL 模型达 100% pass**(boxnet / family_relationships 等 OOD 任务从 ~0 到完美)。Creativity Index 也显示长时 RL 产出更新颖轨迹。

**稳定长 RL 的关键技巧**:**KL 正则 + 参考策略硬重置**——KL 项训久会主导致停滞,**当验证停滞就把 ref 硬重置到近期 policy 快照、并重置 optimizer**,恢复稳定并促使更大幅偏离 base。配 DAPO(decoupled clip $`\epsilon_{\text{low}}{=}0.2,\epsilon_{\text{high}}{=}0.4`$ + dynamic sampling)。**>2k 步仍持续增益**(v2 到 3k 步,加 REINFORCE++-baseline + cosine length penalty)。32×H100 ≈ 16k GPU-h,框架 verl。

**诚实的 nuance**:pass@k 有三种形态——**Diminish / Plateau / Sustained**,**并非所有任务都扩展**:简单任务早饱和、prolonged RL 无额外收益,**复杂任务(如 coding)才持续扩展**。结果:1.5B 在 math/code/STEM/逻辑全面超 R1-Distill-1.5B,常追平 7B。

> **怎么调和共识 12 与 ProRL**:共识 12 说"RL 上限 ≈ base、别硬刚冲过上限",ProRL 说"长时 RL 能扩边界"——两者的边界在于**任务复杂度 + 是否用 KL-reset 维持探索**。简单任务上共识 12 对(早饱和);复杂任务 + 主动重置参考策略防熵塌时,ProRL 能再往外推。**两条都不是铁律,看任务和配方。**

---

## §10H Open-Reasoner-Zero (ORZ) — 极简 vanilla PPO base RL (StepFun, 2025-03)

- **arXiv**: [2503.24290](https://arxiv.org/abs/2503.24290)
- **定位**: 证明 **vanilla PPO + GAE(λ=1,γ=1) + 简单规则 reward + 完全无 KL** 就能在 base 上 scale,复现 R1-Zero。**刻意选 PPO 而非 GRPO**。

**精确超参(全公开)**:GAE **λ=1.0, γ=1.0**, clip ε=0.2;policy lr **1e-6** / critic lr **5e-6**,常数 + 50 步 warmup;batch **128 prompt × 64 response**, T=top-p=1.0;policy 严格 on-policy(每次生成 1 步优化),critic 12 mini-batch;**无任何 KL / entropy bonus**;AdamW β=(0.9,0.95) 无 weight decay;**纯二值 reward(无 format reward)**。

**反直觉教训**:**γ<1.0 会惩罚长期奖励 → response 变短、性能上不去**(模型早终止抢奖励);**λ=0.95 不稳定**(λ=1.0 才收敛);**KL loss 和 penalty 同时去掉**才最优。32B:AIME24 48.1(超 R1-Zero-Qwen-32B 47.0),仅需 **1/10 步数**。

---

## §10I Kimi K1.5 — 在线镜像下降 + L2 软信赖域(K2 算法之源)(Moonshot, 2025-01)

- **arXiv**: [2501.12599](https://arxiv.org/abs/2501.12599)
- **定位**: 文档 §19 提到的"K1.5 闭式 L2 trust region"的真正出处。**不用 MCTS / value 网络 / PRM**。

**机制(替代 PPO clip 的是 L2 平方惩罚)**:优化 KL 正则化目标 $`\max_\theta\mathbb E[r]-\tau\mathrm{KL}(\pi_\theta\|\pi_{\theta_i})`$,有闭式最优解 $`\pi^*\propto\pi_{\theta_i}\exp(r/\tau)`$;取对数得 surrogate = **对 log-ratio 的 L2 平方损失**,梯度 = 带经验均值 baseline $\bar r$(**无 value 网络**)的 policy-gradient **减去** $`\frac\tau2\nabla(\log\frac{\pi_\theta}{\pi_{\theta_i}})^2`$ 软信赖域项。这是 K2(§19)"L2 trust region"的来源——一个软的、对称的二次约束,而非 PPO 硬 clip。

**其他**:长度惩罚需 **warmup**(先不加再引入,否则伤前期);**Partial Rollouts**(未完成轨迹存 replay buffer 续跑 + 重复检测早停);**Long2Short** 四法(模型权重平均 / 最短拒绝采样 / DPO / Long2short RL)。参考策略每轮更新为当前 $`\pi_{\theta_i}`$ + **每轮重置 optimizer**。结果 long-CoT AIME 77.5 / MATH500 96.2。**τ、lr、batch 未公开。**

---

## §10J 小模型 RL 配方:DeepScaleR / Polaris / Light-R1

> 三个"小模型 + 公开配方"的代表,共同教训:**数据难度与模型强度要匹配,长度课程要因模型而异**。

### DeepScaleR (Agentica/Berkeley, 1.5B) — 迭代上下文扩展

从 R1-Distill-1.5B 起,GRPO + **8K→16K→24K** 迭代扩 ctx → AIME24 **28.8% → 43.1%**(超 o1-preview),成本仅 **~3,800 A100-h ≈ \$4,500**。**核心教训**:**"先学会高效推理,再学会推理更长"**——直接长 ctx 低效,因"错误的长回答更难学"。8K 阶段准确率先从 28.9 跌到 22.9,但 **response 长度从 5,500 降到 3,500、train reward 46%→58%**(逼出精炼 pattern)。**扩 ctx 的信号:response clipping ratio 从 4.2%→6.5%**(8K 成瓶颈)。⚠️ 争议:FastCuRL 复查发现 8K 时约 45% 输出被截断,初始 ctx 最优值仍有争议。

### Polaris (HKU × ByteDance Seed, 4B/7B) — 难度均衡 + 长度外推

**动机**:DeepScaleR 这类数据对更强模型(Qwen3-4B)**太简单**,先进模型 RL 边际提升甚至下降。四支柱:① 校准数据难度(镜像 J 形,**每阶段末移除 accuracy>0.9 样本**动态保持挑战);② **高温保多样**(Polaris-4B 用 **1.4/1.45/1.5** 远超常规;7B 用 0.7/1.0/1.1);③ **train-short generate-long**(用 Yarn 外推,4B+Yarn 在 >48K 后超 Qwen3-4B);④ 多阶段。**反直觉**:"先短后长"范式**不普适**,更保守的是直接让模型从头就"想长"。

### Light-R1 (奇虎360, 32B/14B) — 课程 SFT+DPO+GRPO,全公开数据

三阶段(课程 SFT → semi-on-policy DPO → GRPO),**纯公开数据**从 Qwen2.5-32B-Instruct 起。关键 **3K 数据集**;GRPO ~220 步,训练中 **response 长度与 reward 同步上升**(健康信号)。Light-R1-14B-DS:AIME24 74.0/25 60.2,**首次证明 RL 在 14B 数学推理有效**(+~2% 绝对);成本 **12×H800 / 6h ≈ \$1000**。

---

## §10K Llama-Nemotron — FP8 生成 + 课程 (NVIDIA, 2025-05)

- **arXiv**: [2505.00949](https://arxiv.org/abs/2505.00949) · Nano(8B)/Super(49B)/Ultra(253B)
- **核心**: **只对 LN-Ultra 做 reasoning RL(GRPO),使其超越 teacher(DeepSeek-R1)并创 GPQA SOTA;小模型做 RL 不如蒸馏**。GRPO:rollout 72 prompt × 16 response,T=1/top-p=1,global batch 576,每 rollout 2 次梯度更新;reward = Llama-3.3-70B judge accuracy + format;**刻意丢 pass-rate≥0.75 的题**做难度课程。
- **FP8 生成(train-infer 工程亮点)**:生成用 FP8;权重同步链路 = pipeline 维 all-gather → 转 vLLM 格式 → 共享内存 → offload 训练显存 → vLLM 生成 → sleep 释放 → 重载训练显存。**结论:SFT 已接近 R1,但 RL 阶段对超越 teacher(尤其 GPQA)至关重要**——印证"SFT 打底、RL 超越"。

---

## §10L 熵的动力学:崩溃定律 + 高熵 minority token(共识 12 深化)

> 把[共识 12](#共识-12--entropy-collapse-是头号杀手治法多样)的"entropy collapse 是头号杀手"从经验上升到**定量规律 + 治理工具**。三篇互补,务必一起读。

### 熵-性能定律(Cui et al. [2505.22617](https://arxiv.org/abs/2505.22617))

- **$`R=-a\cdot e^{H}+b`$**:验证性能是策略熵的指数函数,$`H{\to}0`$ 时天花板 $=-a+b$。**熵塌 = reward 上限被锁死**。
- **前 200 步消耗 73% 的熵、拿走 76% 增益**;前 800 步 >93% 增益 + 94% 熵损失。**超 2/3 训练步只在做边际优化**——量化背书了"用最少 compute 触达上限"。
- **机制**:$`H(k{+}1)-H(k)\approx-\mathrm{Cov}(\log\pi,\Delta\text{logits})`$,PG 下 $`\propto-\mathrm{Cov}(\log\pi, \pi\cdot A)`$,该协方差全程为正 → 熵单调降。
- **治法(掐极端离群)**:top-0.02% token 平均协方差 **5.654** vs 全体 **0.003**——极小一撮驱动崩溃。**Clip-Cov**(对高协方差 token 裁剪,clip ratio 2e-4)/ **KL-Cov**(加 KL 罚,k=2e-3@7B / 2e-4@32B)。32B 上比 GRPO **+6.4%**,熵能维持 "10× higher"。

### Beyond 80/20:高熵 minority token 驱动学习(Qwen [2506.01939](https://arxiv.org/abs/2506.01939))

- 只有 **~20% 高熵 "forking token"**(决策分叉点)在 steer 推理路径。**只对这 20% 做 PG 更新**,效果匹配甚至超全量:Qwen3-32B AIME25 +11.04 / AIME24 +7.71。**只训 bottom-80% 低熵 token → 性能显著下降**。

### ⚠️ 两篇的张力(必须点明)

- Cui et al. 说**掐掉极端高协方差 token(top-0.02%)防崩**;Beyond 80/20 说**聚焦高熵 token(top-20%)促学**。看似矛盾,实则**层次不同**:top-0.02% 是**病态离群点**(协方差异常,该抑制),top-20% 是**健康分叉点**(熵高但协方差正常,该保留)。统一结论:**梯度该聚焦少数关键 token,但要分清"病态离群"与"健康分叉"。**

### 其他治理共识(社区收敛)

- **clip-high 降熵 / clip-low 升熵**(干净分解 [2509.26114](https://arxiv.org/pdf/2509.26114)):光选 clip 参数就能控熵,**不必动 KL**。
- **bf16 rollout 不崩、量化 rollout 崩**:很多"熵崩"其实是 train-infer 失配的下游症状(见共识 12 深化 + §44)。
- **固定熵系数不够、提温只推迟不避免、加熵正则可能反弹**(step 1500 后突 spike)——单一手段都不够,要组合。

---

# Part II. 多轮工具 / 搜索 Agentic RL

> **本部分要回答**:把 reasoning RL 扩展到"reason → 调工具 → reason → 答题"的多轮交互后,新增哪些工程问题、怎么解。
>
> **核心观点**:
> - **Tool token mask 是入场票**(共识 1):所有 7+ 个工作的统一做法,不 mask 会让模型背环境固定输出
> - **冷启动 SFT 三派分明**(共识 8):派 A 不 SFT(§11/§14/§16),派 B 只教 format 不教 reasoning(§17 rStar2),派 C 含 long-CoT(§13/§15)
> - **Reward 越简单越稳**(共识 4):Search-R1 实证 intermediate retrieval reward 有害;rStar2 §17 §4.3.1 明确反对 step-level shaping
> - **§17 rStar2-Agent 是 agentic math RL 目前最完整公开 recipe**,带踩坑日志;14B 510 步超 671B DeepSeek-R1
> - **算法主流**:从 §11 PPO/GRPO → §13 PPO + tool mask → §17 GRPO-RoC(Resample-on-Correct,处理 positive trajectory 里的 tool error)

## §11 Search-R1 (UIUC, 2025-03)

- **arXiv**: [2503.09516](https://arxiv.org/abs/2503.09516) · [GitHub](https://github.com/PeterGriffinJin/Search-R1)
- **后续**: [Empirical Study 2505.15117](https://arxiv.org/abs/2505.15117) · [Search-R1++ 2602.19526](https://arxiv.org/abs/2602.19526)
- **定位**: 把 search engine 当可调用环境，让 LLM 通过 RL 学会多轮交错 reason ↔ search 的 canonical recipe。

### 算法

- 支持 PPO / GRPO / REINFORCE；**默认 PPO**（更稳）
- GRPO 收敛快但易后期不稳；Search-R1++ 报告 **REINFORCE > PPO > GRPO**

### Trajectory & masking（重点）

- 多轮模板：`<think>...</think> <search>q</search>` pause → retriever 返回 `<information>docs</information>` → 继续 → `<answer>...</answer>`
- **Retrieved Token Loss Mask**：PPO/GRPO 损失加 indicator $`I(y_t)`$ ，对 `<information>` 之间 token 置 0，**只对 LLM 生成 token 计算 policy gradient**。Table 4 ablation 显示去 mask 性能掉很多。
- **Max action budget = 4** search 调用；每次 retrieve top-3 passages。

### Reward

纯 outcome EM， $`r = \text{EM}(\hat a, a)`$ 0/1，**无 format reward、无中间 retrieval reward**。

### 关键超参

| 项 | PPO | GRPO |
|---|---|---|
| policy lr | 1e-6 | 1e-6 |
| value lr | 1e-5 | — |
| KL β | 0.001 | — |
| clip ε | 0.2 | — |
| batch (mini/micro) | 512 / 256 / 64 | 同 |
| max seq length | 4,096 | 4,096 |
| group size G | — | 5 |
| temp / top-p | 1.0 / 1.0 | 同 |
| 训练 steps | 500 | — |
| max turns | 4 | 4 |

Backbone: Qwen2.5-3B/7B base/instruct；**No SFT cold start**。

### Empirical Study (2505.15117) 关键结论

- **Format reward 有效；intermediate retrieval reward 基本无用甚至有害**（reward hacking）
- BM25 / E5 / Google 三类 search engine 训出的 agent 性质差异巨大；dense retriever 训练动力学更平稳
- 单纯 F1 reward 会出现 **"answer avoidance"**（学会不答以避免错答），需加 action-level penalty

---

## §12 R1-Searcher / R1-Searcher++ (RUC, 2025-03 & 2025-05)

### R1-Searcher (v1)

- **arXiv**: [2503.05592](https://arxiv.org/abs/2503.05592) · [GitHub](https://github.com/RUCAIBox/R1-Searcher)
- **算法**: 两阶段 Reinforce++（不需要 distillation、不需要 SFT）

**两阶段 reward**：
- **Stage 1**（学调用 retrieval）： $`R_{\text{retrieval}} = 0.5`$ if n≥1 else 0； $`R_{\text{format}} = 0.5/0`$
- **Stage 2**（学答题）： $`R_{\text{answer}} = \text{F1}(\hat a, a)`$ ； $`R_{\text{format}} = 0`$ if 对 else **−2**

**Modified Reinforce++ with RAG-rollout**：碰到 `<|end_of_query|>` pause 调 retriever，把结果包 `<|begin_of_documents|>...<|end_of_documents|>` 拼回；**这两标签间所有 token 不参与 loss 和 IS**。

**关键超参**：
| 项 | 值 |
|---|---|
| lr | 2e-6 |
| train batch / rollout batch | 256 / 64 |
| 每样本 rollouts | 16 |
| temperature | 1.0 |
| max retrievals | 8 |
| KL β | 0 (Qwen-2.5-7B-Base) / 1e-4 (Llama-3.1-8B-Instruct) |
| Stage 1 数据 | 200 HotpotQA + 150 2Wiki (medium) |
| Stage 2 数据 | 4561 HotpotQA + 3581 2Wiki (medium+hard) |

### R1-Searcher++ (v2)

- **arXiv**: [2505.17005](https://arxiv.org/abs/2505.17005) · [GitHub](https://github.com/RUCAIBox/R1-Searcher-plus)
- **定位**: v1 + SFT cold start + 内部知识利用 reward + 记忆化模块，让模型熟悉问题尽量用内知，未知再检索

**SFT cold start**: 720 HotpotQA + 85 2Wiki, 6 epochs, lr 2e-5, batch 64；document token mask。

**RL Loss**: $`\mathcal L = -J_{\text{Mask}}(\theta) + \mu \mathcal L_M(\theta)`$ 。

**Reward**：
- **answer**：1 if 答案≤10 词 **且** Cover-EM=True else 0（兼约束简洁）
- **format**：0 / −2
- **内部知识鼓励**： $`R_{\text{group}}(q, o_i) = \min(2\sigma^2, \eta)`$ ，σ 为同组正确 trajectory 检索次数的标准差（鼓励同组内检索次数有差异，间接奖励"能不查就不查"），η=2 上限

**Memorization**：另训一个 rewriting model 把检索文档消化成不依赖检索的推理路径，正确样本组成 𝒯，loss $`\mathcal L_M`$ 让 policy 内化外部知识。

**超参**: lr 2e-6；RL batch 1024 / rollout 64；rollouts 16；KL β 1e-4；μ=0.1；max retrievals 8；temp 1.0 / top-p 0.95；Qwen-2.5-7B-Instruct；8142 样本 / 1 epoch。

---

## §13 ReTool (ByteDance Seed, 2025-04)

- **arXiv**: [2504.11536](https://arxiv.org/abs/2504.11536) · [项目页](https://retool-rl.github.io/) · [GitHub](https://github.com/ReTool-RL/ReTool)
- **定位**: 用 RL 教 32B reasoning 模型何时插入 Python interpreter；AIME24 上 400 步即超过 1080 步的纯文本 RL baseline。

### 两阶段流程

1. **Cold-start SFT**（2 epochs）：开源数学题 + 人工 + DeepSeek-R1 双验证，把人工心算改写为可执行代码，让模型学会 `<code>` / `<interpreter>` 触发格式
2. **RL with PPO + verl**

### 算法 & masking

- **PPO**（不用 GRPO，理由：更稳）
- **Interpreter Feedback Mask**：`<interpreter>...</interpreter>` 之间 sandbox 输出 token 全部从 loss mask
- **KV-Cache Reuse**：检测到 `</code>` 时缓存 KV，只对 interpreter 反馈做 prefill，加速 rollout
- **异步 Sandbox Pool**：分布式 sandbox worker 负载均衡

### Reward

$R = +1 / -1$ （is_equivalent 二值）。**没有 format reward、没有 code-error penalty**——刻意简化以避免 hacking。

### 关键超参

| 项 | 值 |
|---|---|
| optimizer | AdamW |
| lr | 1e-6 |
| **KL β** | **0.0** |
| mini-batch | 512 |
| **max seq len** | **16,384** |
| temp / top-p | 1.0 / 0.7 |
| backbone | Qwen2.5-32B-Instruct（也开源 ReTool-DeepSeek-R1-Distill-Qwen-32B） |
| RL 数据 | DAPO-Math-17k |
| 训练 steps | 400 |

**结果**：32B AIME24 67.0% / AIME25 49.3%（400 步） vs 纯文本 RL 40% / 1080 步。涌现 code self-correction。

---

## §14 ToRL (GAIR-NLP, 2025-03)

- **arXiv**: [2503.23383](https://arxiv.org/abs/2503.23383) · [GitHub](https://github.com/GAIR-NLP/ToRL)
- **定位**: **从 base 模型直接 zero-SFT 上 RL** 学 code interpreter 使用，证明 TIR 在 base model 可行。

### 算法 & masking

- **GRPO**（基于 veRL）；group size 16，rollout batch 128 prompts
- **OBSERVATION mask**：sandbox 输出从 loss mask（防记忆 deterministic 输出）
- **去掉 KL loss**（"all experiments omit the KL loss"）——最大化探索

### Reward

±1 outcome。**Code Executability Reward (−0.5)** 实验过，**无提升**——简单 outcome 足够。

### 关键超参

| 项 | 值 |
|---|---|
| 框架 | veRL |
| 算法 | GRPO |
| rollout batch | 128 prompts |
| group size G | 16 |
| **KL** | **omit** |
| **max tool calls C** | **1**（C=2 提 ~2% acc 但训练时间×3） |
| temperature | 1.0 (训练) / 0 (评测) |
| backbone | Qwen2.5-Math-Base 1.5B / 7B（**纯 base, no SFT**） |
| GPU | 8×A800 |

### 稳定化 trick

- 用 **Sandbox Fusion** 而非 qwen-agent（隔离、稳定）
- **Error message 仅保留最后一行**（防 context 爆炸）
- C=1 限制 tool call 频次，避免 GPU 等代码执行 idle

**结果**：ToRL-7B AIME24 43.3%（同设置纯 RL +14%、SOTA TIR +17%），接近 32B 水平。

---

## §15 Tool-Star (RUC-NLPIR, 2025-05)

- **arXiv**: [2505.16410](https://arxiv.org/abs/2505.16410) · [GitHub](https://github.com/RUC-NLPIR/Tool-Star)
- **定位**: 6 种工具协同的 multi-tool RL，重点解决 tool-use 数据稀缺 + 多工具协作激励。

### 工具集

- 训练阶段（3）：Search Engine / Web Browser Agent / Code Interpreter
- 推理阶段（3）：Code Debugger / Tool-Use Backtracer / Reasoning Chain Refiner

### 两阶段训练

1. **Cold-start SFT** on Tool-Star-SFT-54K：TIR Prompting + Hint-based Sampling 在纯 CoT 中插入"逻辑验证/答案反思"提示触发工具调用；质量归一化 + 难度感知；easy 进 SFT、hard 留 RL
2. **Multi-Tool Self-Critic GRPO**：每 k 步 vanilla GRPO 后做 self-critic，组 preference pair（r≥1 正 / r<1 负），用 DPO 让模型内化 reward 结构，再回 GRPO 循环

### Hierarchical Reward

```math
R = \begin{cases} \max(\text{Acc} + r_M, \text{Acc}) & \text{format OK 且 Acc}\gt 0 \\ 0 & \text{format OK 且 Acc=0} \\ -1 & \text{otherwise} \end{cases}
```

其中 $`r_M = 0.1`$ if trajectory 同时含 `<search>` **和** `<python>` else 0——**显式奖励多工具协作**。

### 关键超参

| 项 | 值 |
|---|---|
| 算法 | GRPO + Self-Critic DPO |
| train_batch_size | 128 |
| ppo_mini_batch_size | 16 |
| rollout_n | 8 |
| total_epochs | 2 |
| GPU | 1 node × 8 |
| SFT lr / epochs / scheduler | 7e-6 / 3 / cosine warmup 0.1 |
| SFT cutoff len | 15,000 |
| backbone | Qwen2.5-3B-Instruct（也测 Llama3.2-3B） |
| infra | verl + vLLM ≤ 0.6.3 + torch 2.4.0 |

XML 格式：`<search>q</search>` / `<python>code</python>` / `<result>output</result>` / `<answer>...</answer>`。

**后续**：同班人马 ARPO (ICLR 2026) 号称比 Tool-Star 快 4×。

---

## §16 ARTIST (Microsoft Research, 2025-04)

- **arXiv**: [2505.01441](https://arxiv.org/abs/2505.01441)
- **定位**: 把 agentic reasoning + GRPO 统一用在 math + multi-turn function calling 两个异质域，强调 outcome-based RL 不需 step-level supervision。

### 算法 & masking

- **GRPO**（无 critic、组内 baseline）；去 KL 鼓励探索
- Tool 输出在 loss mask，"gradient 只通过 model-generated token"

### Reward（分域）

- **Math**：answer=2，format=0.5 relaxed + 0.5 strict，tool_execution = 成功 Python 调用比例
- **Function calling (τ-bench / BFCL v3)**：
  - State reward $`= \text{StatR}_{\max} \times \text{State}_{\text{match}} / \text{State}_{\text{total}}`$
  - Function reward $`= \text{FR}_{\max} \times \text{F}_{\text{match}} / \text{F}_{\text{total}}`$
  - Format reward 0.025–0.1

### 关键超参

- **No SFT cold start**；直接 Qwen2.5-7B/14B-Instruct 起步
- Math: 6 rollouts/q, temp 1.0, max gen 8,000
- Function calling: 8 rollouts/q, max context 16,384, max response 2,048

**Prompt 模板**：
- Math: `<think>` / `<python>` / `<output>` / `<answer>`
- Function calling: `<reasoning>` / `<tool>` / `<tool_result>`（含失败信息）

**结果**：AMC +22%, AIME +18%, Olympiad +16.9%；τ-bench airline +14%，correct tool call +30%，步数 −15%。

---

## §17 rStar2-Agent (Microsoft Research, 2025-08)

- **arXiv**: [2508.20722](https://arxiv.org/abs/2508.20722) · [GitHub](https://github.com/microsoft/rStar)
- **定位**: 14B 模型 510 步 RL（64×MI300X，一周）超过 671B DeepSeek-R1，AIME24 80.6%/AIME25 69.8%；**当前 agentic math RL 最完整的开源 recipe**。

### 核心算法: GRPO-RoC (Resample-on-Correct)

普通 GRPO + outcome-only reward 在 code 环境的核心问题：positive trajectory 里也含 10–15% tool error，模型学会"错就错吧，最终对就行"。

**GRPO-RoC**：每个 q **oversample 2G** 条 trajectory，再下采样到 G：
- **Negative**：均匀采 ⌊|O_neg|/2⌋，**保留 failure 多样性**
- **Positive**：按 penalty score $`p_{\text{total}} = p_{\text{err}} + p_{\text{format}}`$ 反比例采样，**优先选 tool-error 少、format 干净的正样本**
  - $`p_{\text{err}}`$ = tool error 数 / total tool call 数（无 tool call 默认 0.5，鼓励用工具）
  - $`p_{\text{format}}`$ ：`<answer>` 标签数异常按比例罚

### Reward

**Answer-only 0/1**。论文**明确反对** step-level reward 和 tool-error penalty——理由是 reward hacking 风险。

### 关键超参

| 项 | 值 |
|---|---|
| optimizer | AdamW |
| lr | 1e-6, constant + linear warmup 20 rollout steps |
| batch | 512 prompts |
| 2G | 32 (oversample) |
| G | 16 |
| **KL** | **去掉** |
| **Clip-Higher** | **ε_low=0.2, ε_high=0.28** |
| **Entropy loss** | **去掉** |
| temperature | 1.0 |
| **max turns T** | **10**（stage 1/2）→ **15**（stage 3） |
| 最大 response 长度 | **8K → 12K → 12K**（远短于 DAPO 20K / MiMo 48K） |
| 总 steps | 510 |
| GPU | 64× AMD MI300X |
| 训练时长 | **1 week** |
| 框架 | VERL v0.2 + SGLang |

### Non-Reasoning SFT cold start（关键设计）

故意**不放 reasoning 数据**，只装 instruction-following + JSON function calling + 工具使用。

- 数据：165K function call（ToolACE-11K + APIGen-MT-5K + Glaive-101K + 48K Magicoder JSON 改写）+ 30K Tulu3 + 27K LLaMA-Nemotron
- SFT 超参：lr **5e-6**，warmup 4%，cosine，batch **128**，**3 epochs**
- SFT 后 MATH-500 仅 57.4%、AIME24 仅 3.3%——**几乎不用 SFT 提 reasoning，全部留给 RL**

### 三阶段 RL

1. **Stage 1**（8K, 全 42K, ~300 步）：clip ratio 一度 >10% 也不放，强迫学短而高效 reasoning
2. **Stage 2**（12K, 全数据, ~85 步）：clip ratio 稳定 10% + plateau 时升 max len
3. **Stage 3**（12K, 17.3K hard subset, 125 步）：用上阶段策略筛去 8/8 全对的题；**reset optimizer + update reference model 为最新 policy**

### Tool call 格式（JSON function-call，非 markdown）

```
<tool_call>{"name":"execute_python_code_with_standard_io","arguments":{"code":"...","input":""}}</tool_call>
```
反馈包 `<tool_response>...</tool_response>`；环境 4 类返回：stdout / IPython output / error+traceback / timeout。

### 失败案例 & 经验（论文 §4.3.1 难得的踩坑日志）

1. **Overlong filtering 反向不 work**：扔掉超长 trajectory 反而让 overlong 比例上升（无负反馈 → 重复 pattern 不被纠正）。**保留 truncated trajectory 的负 reward 更有效**
2. **N-gram 重复检测**误杀合法的 verify pattern（如"换个输入验证"），降低 response length 和 score
3. **Reward hacking 警告**：复杂 rule-based reward / step reward / tool-error penalty 都易引入 bias、压制有用行为
4. **RL 上限 ≈ base model 上限**：step 510 之后继续训会 collapse；温度提到 1.2 / 加长 max len / 提 ε_high / T=20 / reset optimizer 都救不回来。**当前 RL 大概率不能突破预训练潜力，关键是用最少 compute 触达上限**

### 基础设施

65K 并发 tool call、平均 0.3s 端到端延迟；**Load-Balanced Rollout Scheduler** 按各 GPU 当前 KV cache 余量动态分配 rollout，tool call **异步**派发不阻塞 generation——多轮 agent RL 跑稳的工程关键。

---

> **【新增:多轮崩溃机理 + 信用分配】** §17A–§17F 专攻[共识 15](#共识-15--多轮-agentic-rl-有独立于单轮的崩溃模式void-turn--echo-trap--entropy-失控--template-collapse最重要的新坑)讲的"多轮独有崩溃"。这是用户最关心的"别人踩过的坑",每条都给 **机理 → 监控信号 → 修复 → 数字**。先看 §17A(最干净)。

## §17A SimpleTIR — "Void Turn" 致梯度爆炸(最干净的崩溃闭环)

- **arXiv**: [2509.02479](https://arxiv.org/abs/2509.02479) · 多轮 Tool-Integrated Reasoning(TIR)的 RL 稳定性
- **定位**: 整个 agentic RL 文献里**机理 + 信号 + 修复 + 数字最完整的一个崩溃案例**,代码人必读。

### 故障机理(含 Proposition 3.1)

多轮 TIR 的 RL 不稳定根源是**工具反馈引入的分布漂移**:工具返回是 OOD 的,模型在其条件下生成"漂离预训练分布、高度随机、被赋予异常低概率的 token";这些低概率 token 喂回下一轮 → **跨轮复合恶化** → 后段轮次崩溃 → **灾难性梯度范数爆炸**。对负奖励轨迹,IS ratio **上方无界**,某 token 旧策略概率极小时,一次小更新就让 ratio 爆炸。Proposition 3.1 给出梯度范数 $`\propto \rho_{i,t}\cdot|\hat A_i|\cdot\sqrt{1-2P(c)+\sum_j P(j)^2}`$,当采样 token 被赋低概率时 $(1-2P(c))$ 趋最大,持续放大梯度。

### 监控信号 + 修复

- **信号**:**梯度范数**。SimpleTIR 下梯度范数平稳无尖刺,Naive Multi-turn 出现灾难性爆炸。
- **修复**:定义 **void turn** = 一轮响应"既无完整代码块、也无最终答案"(残缺代码 / 重复 / 过早 eos);**只要任一轮是 void turn,整条轨迹的 policy loss 被 mask 并在 GRPO 更新前移除**。
- ⭐ **关键教训**:**单纯过滤"低概率 token"或"高 ratio token"都救不了**(消融:Low-Prob Filtering 23.3、High-Ratio Filtering 26.3,都远低于 SimpleTIR 50.5),**必须以 void turn 为标志过滤整条轨迹**——因为问题是跨轮复合的,局部过滤治标不治本。

### 数字(Qwen2.5 base,无 SFT)

Qwen2.5-7B:text-base AIME24 3.2 → **SimpleTIR 50.5**;MATH500 51.9→88.4;HMMT25 0.0→29.7。Qwen2.5-32B:AIME24 59.9 / AIME25 49.2。(注:摘要里流传的 "22.1→50.5" 是相对另一文本基线,主表是 3.2→50.5,引用别混。)

---

## §17B RAGEN / StarPO — "Echo Trap" + reward std 提前预警

- **arXiv**: [2504.20073](https://arxiv.org/abs/2504.20073) · StarPO(State-Thinking-Action-Reward PO)框架 + RAGEN 系统
- **定位**: 多轮 agent RL 的"复发崩溃模式"命名者,给出**最实用的早期预警指标**。

### Echo Trap 机理

模型**过拟合到局部高奖励的推理套路**:早期输出多样的符号化推理,训练后**坍缩为确定性、重复的模板**(措辞固定),RL 强化的是表层 pattern 而非泛化推理 → 损害长期泛化。

### ⭐ 监控信号有明确先后(看哪个指标 + 提前多少步)

1. **reward std(奖励标准差)先失稳** = 早期预警。FrozenLake-PPO:**std 在 step 40 急跌,而 reward mean 直到 step 90 才崩**——提前 50 步报警(此时性能还接近最优)。
2. **entropy 跟着失稳**。
3. **gradient norm 尖刺 = 不可逆崩溃临界点**(Finding 3)。Bandit step 170 / Sokoban 110 / FrozenLake 90。

### StarPO-S 三件套修复

① **基于方差的轨迹过滤**:用轨迹 reward 的 std 作不确定性 U,**只保留 top-25% 高不确定性 prompt**(默认 p=25%);② **Critic baselining**:用 **PPO(带 critic)而非 GRPO**(critic-free 更不稳,过滤在 PPO 下尤其有效);③ **解耦 clip + 去 KL**(借 DAPO)。量化:FrozenLake 保留 75% rollout 把稳定期 100→140 步,**保留 50% 直接不崩**。

### 关键洞察:推理不会自动涌现

"**没有细粒度、reasoning-aware 的 reward,多轮 RL 里 agent reasoning 几乎不会涌现**",否则只学到浅层策略或幻觉式思考。Rollout 设计偏好:多样初始状态 + 中等交互粒度 + 更频繁采样。

---

## §17C EPO / RAGEN-2 — 熵失控 + "Template Collapse"(熵看不见的崩溃)

> 这两篇把"多轮崩溃"推到比 Echo Trap 更深一层:**连熵本身都可能是误导信号**。属 2025Q4–2026 的前沿,结论可能演化,但机理极具启发。

### EPO — 问题不是熵塌,而是熵失控([2509.22576](https://arxiv.org/abs/2509.22576))

- **颠覆性发现**:多轮场景的核心问题**不是 entropy collapse,而是 uncontrolled entropy dynamics**(熵既暴跌也暴涨、剧烈震荡)——"exploration-exploitation cascade failure"。**单轮"熵只会塌"的直觉在多轮失效。**
- **为何常规手段失败**:① 所有轮共享策略参数,逐轮调熵**无法解耦早期探索与后期利用**;② 标准正则 stateless,但多轮里一次更新同时影响所有轮;③ decay schedule 会过早压制 → 锁死次优。
- **EPO 三组件**:① 轨迹级熵正则(跨所有轮聚合);② **熵平滑正则**——把熵锚到历史均值走廊 **$[0.5\bar H, 1.5\bar H]$**(±50%),ReLU 罚;③ 自适应相位加权。理论:累计熵偏差 O(T) vs 标准正则最坏 O(T²)。数字:ScienceWorld(Qwen2.5-7B)+152%、ALFWorld(3B)+19.8%。⚠️ **并非处处正收益**:PPO+EPO 在 ALFWorld 某指标反降 10.9%。

### RAGEN-2 — Template Collapse:熵看不见的崩溃([2604.06268](https://arxiv.org/abs/2604.06268), 2026)

- **2026 新发现**:**即使熵保持高位,推理仍可能漂向固定模板**——单输入内看似多样,**跨输入几乎相同**。称 template collapse,"一种熵和所有现有指标都看不见的故障模式"。
- **分解**:推理多样性 = 输入内多样性 $H(Z|X)$ + 跨输入可区分性 $I(X;Z)$;**熵可以很高而 $I(X;Z)\to 0$**。
- **诊断:互信息 MI proxy**(批内 cross-scoring,无需外部模型):MI z-score 与性能 Spearman **+0.39**,而各熵指标 **−0.11~−0.14(方向反了)**——"MI 预测性能比熵可靠 2 倍,熵甚至指向错误方向"。
- ⭐ **给 StarPO-S 打补丁(stabilizer 会反噬)**:任务梯度 $`\le\sqrt{\mathrm{Var}(R|X)}\cdot C`$,**当 reward 方差趋近 0,任务信号消失但正则梯度恒定**,对所有推理链施加"均匀收缩" → **反而加速 template collapse**。即**奖励区分度弱时盲目加正则有害**。
- **修复:SNR-aware filtering**——用 prompt 内 reward 方差作信噪比,保留 top-ρ(**ρ≈0.9**),还省 **26–41% 每步时间**;诊断指标低时过滤反而掉点(要先看 SNR 再决定要不要过滤)。

### 多轮崩溃 → 信号 → 修复 速查(共识 15 配套)

| 信号变化 | 故障 | 出处 | 修复 |
|---|---|---|---|
| 梯度范数尖刺 | void turn → IS 爆炸 | SimpleTIR §17A | 过滤含 void turn 的整条轨迹 |
| reward std 断崖(早于 mean) | Echo Trap | RAGEN §17B | top-25% 方差过滤 + PPO critic + clip-higher |
| 熵单调坍缩($`R{=}{-}a e^H{+}b`$) | 性能被熵锁死 | Entropy Mechanism §10L | Clip-Cov / KL-Cov 抑制 top-0.02% 高协方差 token |
| 熵剧烈震荡 | 多轮 entropy 失控 | EPO §17C | 轨迹级熵正则 + 熵走廊 [0.5,1.5]·H̄ |
| 熵高但 MI(X;Z)→0 | template collapse | RAGEN-2 §17C | MI 诊断 + SNR(reward 方差)过滤 ρ≈0.9 |
| 工具返回后 token 熵飙升 | 工具反馈不确定性未利用 | ARPO §17E | 高熵点自适应分支采样 |
| reward 方差≈0 时加正则更糟 | stabilizer 反噬 | RAGEN-2 §17C | 别盲目加正则,先提任务区分度 |

---

## §17D GiGPO / Tree-GRPO — 长程 agent 的两级/树形信用分配

> 长程 agent(几十步、稀疏延迟 reward)逐步信用分配极难。两条路线:分组算两级优势 vs 树形共享前缀。

### GiGPO — Group-in-Group Policy Optimization([2505.10978](https://arxiv.org/abs/2505.10978), NeurIPS 2025)

**两级相对优势**(critic-free、零额外显存):① **episode 级**(整轨迹组算宏观优势);② **step 级 anchor-state grouping**——回溯识别跨轨迹**重复出现的环境状态**,把同状态出发的动作分到一组算微观优势。同时抓"全局轨迹质量 + 局部步有效性",无需辅助模型/额外 rollout。数字(Qwen2.5-1.5B/3B/7B):**ALFWorld 比 GRPO +12%、WebShop +9%**,同等显存与时间。

### Tree-GRPO — 树形 rollout 共享前缀([2509.21240](https://arxiv.org/abs/2509.21240), ICLR 2026)

- **树节点 = 一个完整 agent step**(ReAct 的 Thought-Action-Observation),非 token/句级——消融证明语义单元定义是关键。
- **前缀共享**:树搜索替代独立 chain rollout → 固定 token/tool-call 预算下获得更多 rollout。
- **两级优势**:intra-tree(树内过程监督,**仅用 outcome reward 即可**,理论上等价 step-level DPO)+ inter-tree;**只用 intra-tree 会训练崩溃**,intra+inter 组合才稳。
- 数字:Qwen2.5-3B 只用 **1/4 rollout 预算**仍优于 chain;极限预算(每 prompt 2 条 rollout)tree 取得 **112% 相对提升**;多跳 QA 相对 chain-GRPO **16–69%**。

---

## §17E ARPO — 工具返回后熵尖刺 → 熵驱动分支采样 (RUC-NLPIR, 2025-07)

- **arXiv**: [2507.19849](https://arxiv.org/abs/2507.19849) · Tool-Star(§15)同班人马的后续,号称比 Tool-Star 快 4×
- **核心观察(pilot study)**:**每次工具调用后,前 10–50 个 token 熵急剧上升**(模型看到工具输出后最不确定);早期推理熵也升但低于工具反馈后;**search 反馈比 python 反馈引入更多不确定性**(前者返回信息性文本,后者返回确定性数字)。

**熵驱动自适应分支 rollout**:全局预算 M,先采 N 条整轨,剩余留给 partial sampling;在工具返回的高熵点按 $`P_t=\alpha+\beta\cdot\Delta H_t`$ 概率分支(把一条 partial path 分成 Z 条),把 rollout 预算花在"最该探索"的地方,复杂度从 $O(n^2)$ 降到 $O(n\log n)$。advantage 用 soft 设定(共享前缀 token 同 IS ratio)。**数字**:Qwen3-14B deep search,GAIA 36.9→43.7、WebWalker 30→36;**用一半 tool 预算**达到/超过 trajectory-level RL。

> 与 Kimi-Researcher 的 turn-level partial rollout(§18)对照:都在长 agentic 轨迹上做"局部续采",但 ARPO 的触发点是**熵(探索价值)**,Kimi 是**超时(系统效率)**——一个为质量、一个为吞吐。

---

## §17F 高熵 token 机制 + Spurious Rewards(RLVR 评估陷阱)

> 两个"会颠覆你对 RL 涨分理解"的发现,放一起讲。高熵 token 机制见 §10L,这里补 Spurious Rewards 这个**评估陷阱**。

### Spurious Rewards — 随机奖励也涨分,但只对 Qwen([2506.10947](https://arxiv.org/abs/2506.10947))

- **精确主张**:RLVR 用**与正确答案零/无/甚至负相关的 spurious reward** 也能 elicit 出强数学推理——**但仅对特定模型族(Qwen)**。
- **数字**:随机奖励 → Qwen2.5-Math-7B,**MATH-500 +21.4 个百分点**(ground-truth 奖励 +29.1,随机吃到约 3/4)。机理:**GRPO 的 clipping bias 放大预训练学到的高先验行为**(即便奖励无信息),case:code reasoning 频率 65%→90%+。
- ⭐ **必带的 caveat**:这些行为**高度模型依赖**——"对 Qwen 有效的 spurious reward,在 Llama3 / OLMo2 上往往无增益"。
- **对读者的教训**:**在 Qwen 上跑 RLVR 看到涨分,不能直接归因为"学到新能力";务必跨模型族复现,否则结论不可迁移。** 这也解释了为何文档里大量 Qwen-based 工作的结论要谨慎外推。

---

# Part III. 深度研究 / Web Agentic RL

> **本部分要回答**:trajectory 长到几十到上百 turns、context 到 128K、reward 极度稀疏的"deep research"场景,RL 怎么跑得动。
>
> **核心观点**:
> - **算法侧没人发明新东西,创新都在系统侧**:Kimi-Researcher 的 **turn-level partial rollout**(§18)、ASearcher 的**全异步 staleness η**(§20 + AReaL §35)、WebSailor 的 **DUPO 样本复制**(§22)都是工程救命稻草——同步 RL 在长尾下必死
> - **REINFORCE 反而比 PPO/GRPO 受欢迎**(§18 Kimi-Researcher、§11 Search-R1++):长 trajectory 下 critic 难训、clip 难调,outcome reward + REINFORCE 更可控
> - **Reward 几乎都靠 LLM-as-Judge**(§20 / §21 / §22):传统 verifier 写不动,Qwen-72B / 自训 judge 是默认选择
> - **Context management 是必备能力**(§18):naive agent 10 iter 就 OOM,Kimi-Researcher 教模型主动丢弃低价值文档后单条 rollout > 50 iter
> - **失败案例集中在两处**:invalid trajectory rate(WebDancer §21 报 13.6–21.4%)和长 trajectory reward 信号稀疏

## §18 Kimi-Researcher (Moonshot AI, 2025-06)

- **链接**: [官方技术博客](https://moonshotai.github.io/Kimi-Researcher/)
- **定位**: end-to-end RL 从 base 模型直接训出 deep research agent；HLE 从 8.6% → 26.9%（Pass@1）；agent 平均一轮 23 步、200+ URL，可达 70+ search queries / trajectory。

### RL 算法

**REINFORCE + outcome reward**（不是 PPO/GRPO 路线）

选用 REINFORCE 的核心理由：长 trajectory 下不引入额外 critic / 复杂 clip，可控性高。

### Trajectory 级关键设计

- **严格 on-policy**：训练时关闭 LLM engine 的 toolcall format enforcer 等"作弊"机制，保证 trajectory 完全由模型自身分布生成
- **Turn-level Partial Rollout（关键工程贡献）**：超过时间预算的任务存入 replay buffer，下一个 iteration 用更新后的权重继续执行剩余 turns；rollout 加速 ≥ 1.5×
- **全异步 rollout**：Gym-like 接口、unified sandbox 架构、K8s 混合云零停机调度、MCP 协议带 reconnection 的 stateful tool session、多副本容错
- **Context 管理**：训了 context-management 机制让模型主动丢弃低价值文档；naive agent 约 10 iter 就 OOM context，开启后单条 rollout 可 > 50 iter

### Reward

- **Format reward**：tool call 非法 / trajectory 超限 → 罚
- **Correctness reward**：format 合法时按 ground-truth 给二值
- **Gamma decay（关键设计）**： $`r_{\text{step}} = \gamma^{s-1} \cdot R_{\text{outcome}}`$ ，鼓励更短正确轨迹；防止"答对但啰嗦"

### 稳定化 trick

- **Negative sample 选择性丢弃**：负样本拉低 token 概率会引发 entropy collapse，主动 discard 部分 negative samples 维持长期训练稳定（博客明确点出的核心 stabilizer）
- **on-policy 强制**：拒绝 stale data

未公开：γ 具体值、lr、batch、GPU 数。

---

## §19 Kimi K2 Technical Report (Moonshot AI, 2025-07)

- **arXiv**: [2507.20534](https://arxiv.org/abs/2507.20534)
- **底座**: MoE 1T total / 32B activated，pretrain 15.5T tokens（MuonClip 优化器，零 loss spike）
- **成绩**: 非 thinking 模式下 SWE-Bench Verified 65.8、Tau2-Bench 66.1

### RL 算法

**沿用 K1.5 的策略优化算法**（非 PPO / 非 GRPO clip 风格）：
- 推导自 KL-regularized reward maximization 的闭式解 $`\pi^* \propto \pi_{\text{old}} \cdot \exp(r/\tau)`$
- 实际 loss：policy-gradient surrogate + **L2 squared-log-ratio trust region**（替代 PPO 的 min/clip）
- **group 经验均值 $\bar r$ 作 baseline**（GRPO-style，但用 L2 而非 clip）
- 允许 off-policy 复用（partial rollout 跨 iteration）

**结合 RLVR + Self-Critique Rubric Reward**：
- RLVR：数学/STEM/逻辑/IF/Faithfulness/Coding/Safety 用二值规则奖励
- Self-critic：开放任务上做 pairwise 比较，rubric 包含 clarity / fluency / objectivity 等
- **闭环 critic refinement**：on-policy RLVR rollout 持续 update critic，把可验证信号蒸馏到主观判断里

### Agentic 数据合成

三阶段：tool spec 仓库 → diverse agents/tasks 生成 → 在 simulated env 跑出 multi-turn 成功轨迹 + 大规模 rejection sampling 质量过滤。

### 稳定化 trick

- **Token Budget Control**：per-task token 预算硬限制，避免冗长换 reward
- **PTX Auxiliary Loss**：pretrain CE loss 兜底，防止灾难性遗忘；系数 b 平衡 RL 推进
- **Temperature Decay**：训练初期高温探索，末期降温收敛
- **Critic 资格门控**：critic 必须先在可验证任务上证明能力，才允许评判主观输出（防 reward hacking）

具体 lr / batch / clip ε 等超参未公开。

---

## §20 ASearcher (Tsinghua IIIS + Ant RL Lab, 2025-08)

- **arXiv**: [2508.07976](https://arxiv.org/abs/2508.07976) · [GitHub](https://github.com/inclusionAI/ASearcher) · [AReaL](https://github.com/inclusionAI/AReaL)
- **定位**: 突破 10-turn 限制，做 long-horizon (≤128 turns) agentic search RL；GAIA +15.0、xBench-DeepSearch +22.4、Frames +14.6 (Avg@4)。

### RL 算法

**GRPO**（Shao et al., 2024 原版） + **核心创新在系统层**：基于 AReaL 做全异步训练

### 异步训练架构

- 训练与 rollout **完全解耦**：长 trajectory 不阻塞下一步训练
- AReaL 引入 **staleness 上限 η** 超参：η=0 退化为同步 RL；η>0 允许使用稍旧 trajectory，保 GPU 利用率近 100%
- batch 内长 traj 不再等 slowest tail

### Trajectory 关键设计

- **Turn limit**：7B/14B = 32，QwQ-32B = 128
- **Context window**：只把最近 25k characters 历史喂回 LRM；input ~ 10k tokens；训练时输出可达 150k+
- **Append-only prompting**：base LLM 模式下 system prompt + 所有 LLM 回应 + search 结果 + 网页 summary 全部追加

### Reward

- **Base LLM**：format_reward × F1（乘法，format 非法直接 0）
- **LRM (QwQ-32B)**：LLM-as-Judge 稀疏 reward（trajectory 末端）
- 无显式 process reward 或步级 shaping

### 超参 / 算力

- batch size：128（7B/14B）/ 64（QwQ-32B）
- 训练数据：35k QA pairs × 2 套，均开源
- **QwQ-32B 训练耗时约 7.6k H100 GPU 小时**
- v2：35k 数据 + 32-turn 上限 = ~48 hours on 128 × A100
- 数据合成 agent：Injection（注入额外事实）+ Fuzzing（混淆事实）两个原子操作迭代

### 稳定化 trick

- **Dynamic filtering**：剔除 reward 全相同 / advantage = 0 的 query
- 未明确做 IS correction / TIS

### 失败案例

**7B 学不会网页浏览**（zero-shot 长网页 summarization 超出 capacity）→ 作者归因模型容量不足。

---

## §21 WebDancer (Alibaba Tongyi Lab, NeurIPS 2025)

- **arXiv**: [2505.22648](https://arxiv.org/abs/2505.22648) · [GitHub](https://github.com/Alibaba-NLP/DeepResearch)
- **定位**: Tongyi DeepResearch 系列**首个 end-to-end 训练的 deep research agent**；ReAct 框架；首倡 agentic data synthesis + agentic RL 双轨。

### 四阶段 pipeline

1. 浏览数据构造（CRAWLQA：爬权威站点 + GPT-4o 合成 QA）
2. trajectory 采样
3. SFT cold start
4. RL（DAPO）

### RL 算法：DAPO

- **Decoupled Clip and Dynamic Sampling**
- **非对称 clip**：ε_low / ε_high 分别设置
- **Dynamic sampling**：过采样 + 过滤掉 accuracy = 0.0 或 1.0 的 sample
- **Rollout 数**：每步 16 rollouts

### Reward 公式（明确给出配比）

```
R(ŷ, y) = 0.1 · score_format + 0.9 · score_answer
```
- format：strict 格式合规 + JSON tool call 合法的二值
- answer：用 **Qwen-72B-Instruct 作 LLM-as-Judge**

### 关键超参

- 推理温度：0.6（标准）/ 1.0（RL rollout）
- top-p：0.95（标准）/ 1.0（RL rollout）
- repetition penalty：1.1（LRM）/ 1.0（LLM）
- lr / batch / KL / entropy 系数：**论文未给具体值**

### Tool token masking

- SFT 阶段 observation 不计入 loss：indicator $`\mathbb{1}[x_i \neq o]`$ 屏蔽 observation tokens
- RL 阶段：tool response 进 context 参与 $`\pi_{\theta_{\text{old}}}`$ 计算，但**只对模型生成 token 做优化**

### 稳定化 trick

- 多阶段 trajectory 过滤：validity → correctness → quality
- **N-gram 重复约束**：10-gram 阈值 = 4，防内化坏 pattern
- 作者发现温度调节"对环境非平稳几乎无影响"

### 算力

- **32 nodes × 8 × NVIDIA H20 (96GB)**
- SFT 数据：7,678 short-CoT + 6,550 long-CoT trajectory
- RL 数据：~5,000 QA pairs（受算力限制）

### 失败案例（论文明确列出）

- 调用不存在 tool（如幻觉出 "calculate" 工具）
- 答案确认后还在"over-action"
- long-CoT → instruction model 迁移时 invalid rate 13.59-21.36%
- **过长 trajectory 导致 reward 信号稀疏**，限制 LRM 在 RL 阶段的提升

---

## §22 WebSailor / WebSailor-V2 (Alibaba Tongyi Lab, 2025-07 / 2025-09)

### WebSailor V1

- **arXiv**: [2507.02592](https://arxiv.org/abs/2507.02592) · [HF](https://huggingface.co/Alibaba-NLP/WebSailor-32B)
- **底座**: Qwen2.5 3B / 7B / 32B / 72B
- **定位**: 解决 BrowseComp 类极高不确定性任务；BrowseComp-en 12.0、BrowseComp-zh 30.1、GAIA 55.4

#### 训练 pipeline

1. **SailorFog-QA 数据**：graph sampling + 信息混淆（obfuscation）造高难任务
2. **RFT cold start**：rejection sampling FT，仅保留 ①答对 ②长度 < 32k tokens ③tool calls > 5 的 trajectory；**observation 在 loss 中 mask 掉**
3. **DUPO RL**

#### DUPO（Duplicating Sampling Policy Optimization，自研）

核心思想：**复制 batch 内 reward variance ≠ 0 的样本**，把 advantage = 0（全对/全错）的 slot 填掉，让 batch 始终保持 dense 学习信号。
- 两阶段动态采样：训练前一次 + 训练中一次
- **收敛速度比 PPO 快一倍以上；比 DAPO 快 2-3×**
- 设计动机：agentic env rollout 慢、reward 稀疏，传统方法 batch 内大量"无信号样本"浪费

#### Reward

- format 严格校验 + LLM-as-Judge answer correctness
- group-relative advantage（GRPO-style）

### WebSailor-V2

- **arXiv**: [2509.13305](https://arxiv.org/abs/2509.13305) · [GitHub](https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent/WebSailor-V2)
- **底座**: Qwen3-30B-A3B-Thinking-2507（MoE 30B total / 3B activated）
- **成绩**: BrowseComp-EN 35.3、BrowseComp-ZH 44.1、HLE 30.6（超过 DeepSeek-V3.1 671B）

#### 关键创新：Dual-Environment RL

- **Simulator**：基于离线 Wikipedia 知识库构建高保真模拟器，低成本 / 高速 / 可控，用于算法快速迭代
- **Managed Real-World**：SerpAPI / Jina 真实 web，做最终 policy 训练
- "symbiotic data-policy feedback loop"：simulator 调参，real-world 收敛

#### Tool set（ReAct）

search / visit / Google Scholar / Python interpreter / final answer。仍使用 DUPO 作为 RL 核心算法。

---

## §23 WebShaper / WebWeaver (Alibaba Tongyi Lab, 2025-07 / 2025-09)

### WebShaper（数据合成）

- **arXiv**: [2507.15061](https://arxiv.org/abs/2507.15061)
- **定位**: **不是 RL 训练工作，而是数据合成方法**；GAIA 60.19、WebWalkerQA 52.50

**核心方法**：
- **集合论形式化 IS 任务**：把 information-seeking 抽象为 entity sets 上的 "Knowledge Projection (KP)" 操作组合
- **Agentic Expander**：迭代地扩展 formal question 并用 retrieval + validation 校验
- 解决传统方法"信息结构 vs 推理结构"、"问题 vs 答案"不一致的问题

### WebWeaver（dual-agent 框架）

- **arXiv**: [2509.13312](https://arxiv.org/abs/2509.13312) · [GitHub](https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent/WebWeaver)
- **定位**: Open-Ended Deep Research (OEDR)；**planner + writer dual-agent**

**架构**：
- **Planner**：iterative cycle，交替 web search 和 outline 优化，dynamic outline optimization 防"fossilization"
- **Writer**：memory-grounded hierarchical synthesis；section-by-section，从 memory bank 做 targeted retrieval；专治"lost in the middle"

**训练**：
- **WebWeaver-3k**：3k 条高质 SFT 数据
- 用来 fine-tune Qwen3-30B-A3B-Instruct → Tongyi-DeepResearch-30B-A3B
- **不含独立 RL 阶段**（RL 由 Tongyi DeepResearch 主报告负责）

---

## §24 Tongyi DeepResearch 主技术报告（汇总）

- **arXiv**: [2510.24701](https://arxiv.org/abs/2510.24701) · [Blog](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)

> 这一篇把 WebDancer / WebSailor / WebShaper / WebWeaver 等系列工作 **整合到一个最终 30B-A3B agent**。

### 三阶段 pipeline

Agentic CPT → SFT → RL

### 关键设置

- **底座**：Qwen3-30B-A3B-Base（30.5B total / 3.3B activated）
- **CPT context 扩展**：先 32K，再扩到 128K
- **RL 算法**：customized GRPO
  - **strict on-policy**
  - **token-level policy gradient**
  - **leave-one-out advantage**
  - **selective negative-sample filtering**（与 Kimi-Researcher 思路类似）
  - **binary correctness reward** + dynamic data curation 做课程学习
- **推理参数**：temperature 0.85、top-p 0.95、repetition penalty 1.1
- **Turn 上限**：128 tool invocations / task；**Context**：128K tokens
- **评测**：Avg@3；HLE 32.9、BrowseComp 43.4、GAIA 70.9
- **基础设施**：asynchronous rollout server、robust sandbox、容错应对真实 API 抖动

**已承认限制**：128K context 对最复杂 long-horizon 任务仍不够。

---

## §24A 更新的 Web/Deep-Research RL 工作(真实环境工程坑)

> 文档主体覆盖 WebDancer/WebSailor/WebShaper/WebWeaver/Tongyi-DR/ASearcher/Kimi-Researcher。这里补几个**训练经验或路线不同**的新工作,重点是**真实 web 环境的工程陷阱**。

**DeepResearcher**(GAIR-NLP, [2504.03160](https://arxiv.org/abs/2504.03160), EMNLP 2025)— **真实 web 端到端 RL 的开山之作**。GRPO + observation masking(共识 1);reward = 格式错 −1 / 格式对给 word-level F1。**真实 web 工程坑(本节最大价值)**:① 高并发 → **50 节点 CPU 集群**处理 rollout 期工具请求;② 反爬/限流 → 重试 + **7 天缓存**(同 query 命中省 Google API,限流 ~200/s);③ 专用浏览 agent 维护短期记忆、分段读网页、自决继续/停止。涌现 **honesty(找不到就拒答)**。比 prompt 基线 +28.9、比 RAG-RL +7.2。**pitfalls**:数据污染(模型用记忆而非搜索,靠 pass@10 筛)、反爬返回无关内容、honesty 不被现有指标奖励。

**WebThinker**(RUC, [2504.21776](https://arxiv.org/abs/2504.21776))— **RL = 迭代在线 DPO**(非 GRPO)。关键对比可作"online > offline 偏好"论据:**迭代在线 DPO 44.9 > 离线 DPO 43.2 > base 42.1**。32B 上 GPQA 70.7%、HLE 15.8%。

**WebExplorer**(HKUST-NLP, [2509.06501](https://arxiv.org/abs/2509.06501))— Qwen3-8B + SFT + **GRPO**;**long-to-short query evolution** 造高难数据;RL 扩到 **128K ctx + 100 工具轮**,工具调用数与性能同步上升;**8B 超 WebSailor-72B(小 9 倍)**。

**MiroThinker**(MiroMind, [2511.11793](https://arxiv.org/abs/2511.11793))— **SFT→DPO→GRPO**;核心贡献 **interactive scaling**:把"agent-环境交互深度"作为模型规模/ctx 之外的**第三个 scaling 轴**,精度对交互深度呈对数关系,256K ctx + **up to 600 工具调用/任务**。**异步**:streaming rollout(未完成轨迹推回下轮)。**pitfalls**:明确承认 **RL 环境 ≠ 最终评测环境**;需大量轨迹清洗(>5 次连续网络异常、动作循环、超时);DPO 刻意**不强制 planning 长度**(会引入系统偏差)。72B GAIA 81.9% / HLE 37.7%。

**AFM / Chain-of-Agents**(OPPO, [2508.13167](https://arxiv.org/abs/2508.13167))— **单模型内模拟多智能体**(动态激活 tool agent + role-play agent),无需外部框架。两阶段:多智能体蒸馏 SFT → **agentic RL(Web/Code 用 DAPO,MHQA 用 PPO)**。trick:SFT 阶段 `ignore_observation` mask 工具返回。环境:真实 web + **nsjail 本地隔离 sandbox**(code)。AFM-RL-32B GAIA 55.3% / HLE 18.0%,**全开源**。

**Atom-Searcher**([2508.12800](https://arxiv.org/abs/2508.12800))— 针对 outcome RL 的 **reward 稀疏 + 梯度冲突**(一个错答案惩罚整条链),提 **Atomic Thought** 细粒度分解 + **RRM 过程奖励 + curriculum reward schedule**(早期重 process、后期转 outcome)。属"细粒度过程奖励"路线(与共识 4 outcome-only 形成张力,仅在难 verify 的 deep research 上才划算)。

> **真实 web 环境的共性坑(综合 DeepResearcher/MiroThinker)**:限流/反爬/缓存是头号工程量;**RL 训练环境几乎不可能 = 评测环境**(API 抖动、网页变化),要接受这个 gap;数据污染需 pass@k 筛查;honesty/拒答这类好行为**不被现有指标奖励**,容易在优化中被牺牲。

---

# Part IV. SWE Agentic RL

> **本部分要回答**:为什么 SWE agentic RL 的核心差异不在算法,而在环境;以及不同 SWE 路线(single-turn vs multi-turn、有真环境 vs 无真环境)的取舍。
>
> **核心观点**:
> - **算法可以抄,环境抄不来**(共识 11):主流算法就是 DAPO 改 / GRPO++ + tool mask;真正的差异在环境数量级。512 docker(DeepSWE) → 2.4K(SWE-Gym) → 7.2K(Nebius) → 8.7K(R2E-Gym) → **20K(Qwen3-Coder)** → 数十万(Cursor)
> - **single-turn 也能 work**(§26 SWE-RL):用 GitHub PR diff 相似度做 reward,**不需要 sandbox**,70B 训出 41% Verified;single-turn 上限有限但启动成本低
> - **multi-turn 必走 RL 全流程**:§25 DeepSWE 32B 64×H100 6 天;§29 Nebius 72B 128×H200 + 65K→131K 长 ctx 课程,**是公开 hyperparam 完整度最高的 SWE recipe**
> - **环境构造路线分两派**:**有人工 issue/test 派**(SWE-Gym)vs **自动 commit→test 派**(R2E-Gym SWE-GEN、Cursor Function Deletion);后者是规模化关键
> - **超长 trajectory 处理出现路线分歧**:DeepSWE Compact Filtering(mask 失败 trajectory)vs Nebius 软惩罚(明确反对 mask)——共识 5 详谈

## §25 DeepSWE (Agentica + Together AI, 2025-07)

- **博客**: <https://www.together.ai/blog/deepswe>
- **模型**: [agentica-org/DeepSWE-Preview](https://huggingface.co/agentica-org/DeepSWE-Preview) · [DeepSWE-Verifier](https://huggingface.co/agentica-org/DeepSWE-Verifier)
- **代码**: [agentica-project/rllm](https://github.com/agentica-project/rllm)
- **定位**: 在 Qwen3-32B 上 **纯 RL（无 SFT 冷启动）** 训练出开源最强 SWE agent，SWE-Bench-Verified Pass@1 42.2%，TTS 后 59%。

### 算法：GRPO++（六件套叠加）

| 改进 | 来源 | 说明 |
|---|---|---|
| Clip-High | DAPO | 提高 surrogate loss 上界，鼓励探索、稳住 entropy |
| No KL Loss | DAPO | 去掉 KL 项 |
| Length Normalization | Dr.GRPO | surrogate loss 除以 max context 长度，去除对错误长回答的长度偏好 |
| Leave-One-Out (LOOP/RLOO) | LOOP | 优势估计去掉自身样本，降低方差 |
| **Compact Filtering** | DeepSWE 原创 | 对 **超长 context / 20 分钟超时 / 触顶 max steps** 的 trajectory **mask loss** |
| **No Entropy Loss** | DeepSWE 原创 | entropy loss 反而会让 entropy 指数级爆炸；只要 base model token-entropy ∈ [0.3, 1] 就不需要 |

### 环境

- 基于 **R2E-Gym**：4500 道训练任务（剔除 sympy 等与 SWE-Bench-Verified 同仓库的污染样本）
- 每个任务是一个 **Docker image**；Kubernetes 编排 **512 个并发 Docker container**
- 工具 4 件套：`file_editor.py`、`execution_bash.py`、`search.py`、`finish.py`
- 工具输出（stdout/stderr）作为 observation token 拼回 prompt

### Reward

**稀疏 Outcome Reward (ORM)**：trajectory 末尾跑 Pass2Pass + Fail2Pass 测试子集
- 1：所有测试在 **5 分钟时限** 内通过（官方 SWE-Bench 时限为 30 分钟，训练时缩短加速）
- 0：任意测试失败或超时

### Trajectory 与超参

| 项 | 值 |
|---|---|
| 评测 context length | 64k tokens |
| 评测 max env steps | 100 turns |
| 推荐推理 max_tokens | 32–64k |
| TTS rollouts (K) | 16（K=8 已基本饱和） |
| TTS scaling 结论 | context 32K → 128K 提升 ≤2%；**trajectory 多样性 + verifier 更划算** |
| 训练 GPU | **64× H100，6 天** |
| 训练效果 | 200 个 RL 步从 23% → 42%（+20 pts） |

### 失败 trajectory 处理

不直接丢，而是 **mask 它们的 loss**（Compact Filtering），让 batch size / advantage 统计不被失败 token 污染。三种触发：达到 max context / 生成超过 20 分钟 / 达到 max env steps。

### 关键观察

- **没用 SFT 冷启动**，直接在 Qwen3-32B 上 RL（32B 规模上首次做到）
- 涌现：边界情况预判、自发回归测试
- Verifier (critic LLM) + K=16 rollouts 的 hybrid TTS 比单纯延长 context 更涨点

---

## §26 SWE-RL / SSR (Meta FAIR, 2025-02 / 2025-12)

### SWE-RL

- **arXiv**: [2502.18449](https://arxiv.org/abs/2502.18449) (NeurIPS 2025) · [GitHub](https://github.com/facebookresearch/swe-rl)
- **定位**: 第一个把 RL 真正 scale 到 SWE 域的工作。**不需要可执行环境、不跑测试**，用 GitHub PR 历史 + difflib 相似度做 reward；纯 single-turn repair 任务但能涌现 SWE 推理。

### 算法

**GRPO**（DeepSeek-R1 路线，rule-based reward）。不用 supervised loss。

### 环境（其实没有 RL 环境）

- 没有 sandbox、没有 docker、不跑测试
- 任务：给 issue + 文件级 localization 的代码上下文，让 model 一次性生成 patch（**single-turn**）
- 用 **Agentless Mini** scaffold：只做 file-level localization，整个文件内容塞 prompt
- 训练数据：**4.6M 仓库** 的 2015–2024 全部 events，过滤后得到 **11M 个 PR instance**；排除所有 SWE-bench 仓库防污染

### Reward

```
if 格式错误: r = -1
else:        r = difflib.SequenceMatcher(predicted_patch, oracle_patch).ratio()  ∈ [0, 1]
```

非常轻量、不依赖执行；相似度作为稠密 reward 已足够引导模型恢复"开发者推理过程"。

### 超参（完整公开）

| 项 | 值 |
|---|---|
| Base | Llama-3.3-70B-Instruct |
| Steps | **1,600** |
| Global batch size | **512** |
| Context window | **16k** tokens |
| 每个 prompt rollouts | **16** |
| 每个 batch problem 数 | 32 |
| 优化器 | Adam |
| 推理 temperature | 1.0（训练）；评测用 greedy 或 T=0.6 × 20-sample majority vote |
| **算力** | **512× H100 ≈ 32 wall-clock 小时** |

### Take-home

- **41% on SWE-Bench Verified**（70B model 当时最强中等规模开源）
- 涌现 5 个 OOD 能力：function coding、library use、code reasoning、math、language understanding（SFT baseline 反而平均退步）
- **不需要真环境就能涨 SWE**，但 single-turn 上限有限

### 后续：Self-play SWE-RL (SSR)

- **arXiv**: [2512.18552](https://arxiv.org/abs/2512.18552)
- 去掉对 "有 issue / 有 test" 的依赖，让 agent 自己生成学习经验
- **512× H100 SXM 80G**（**64 GPU 训练 + 448 GPU rollout**）
- 借鉴 ScaleRL / MiniRL 的 "large-batch + small-policy-staleness" 配置
- SWE-bench Verified 比 human-data baseline +10.4 pts

---

## §27 SWE-Gym (UC Berkeley + UIUC + CMU + Apple, ICML 2025)

- **arXiv**: [2412.21139](https://arxiv.org/abs/2412.21139) · [GitHub](https://github.com/SWE-Gym/SWE-Gym)
- **定位**: 第一个公开的可 RL 真实 SWE 训练环境；其本身 baseline 用 **rejection-sampling FT (filtered BC)** 而非 PPO/GRPO，但提供了后续 SWE-RL 工作的标准底座之一。

### 环境设计要点（最大贡献）

- **2,438 个 Python 真实 task**（Lite 划分 234 个）
- 每个 instance：codebase + **预装依赖的可执行 runtime** + 单元测试 + 自然语言任务说明
- 仓库筛选标准：创建于 2022-07-01 之前、≥500 stars, ≥300 LoC, ≥500 PRs, ≥100 contributors
- 来自 **11 个 Python 仓库**

### 训练方法（baseline）

- **Rejection Sampling FT**（不是 RL）：teacher 或 student 自己 rollout → 过滤 reward 高于阈值的 trajectory → 标准 NLL 训 student
- 两套 scaffold 都跑了实验：**OpenHands CodeAct** 和 **MoatlessTools**

### 超参（OpenHands 32B 主实验）

| 项 | 值 |
|---|---|
| Base | Qwen-2.5-Coder-32B-Instruct |
| 训练数据 | **仅 491 trajectory**（GPT-4o + Sonnet 3.5 采样） |
| 框架 | torchtune full FT |
| LR | **1e-4** |
| Global batch | 8 |
| Max epoch | 5 |
| Context | **32,768** |
| GPU | 2–8× H100 80G on Modal |

### Verifier 训练

32B Verifier：Unsloth + **LoRA rank=64, lr=5e-4, batch=8, epochs=5**，单卡 H100。

### Take-home

- 491 条轨迹 + 32B → SWE-Bench Verified **+13.6%**（到 20.6%）
- 加 verifier best-of-n → **32% Verified / 26% Lite**（当时开源 SOTA）
- **On-policy 反而掉点**（15.3% → 更低），说明 trajectory 质量 > 数量

---

## §28 R2E-Gym (UC Berkeley, COLM 2025)

- **arXiv**: [2504.07164](https://arxiv.org/abs/2504.07164) · [项目页](https://r2e-gym.github.io/) · [GitHub](https://github.com/R2E-Gym/R2E-Gym)
- **定位**: 把 SWE 环境数量从 2.4K 扩到 **8.7K+**，靠 **SWE-GEN** 自动从 commit 反向生成测试；DeepSWE 直接用的就是它。

### SWE-GEN：环境自动构造的核心 recipe

- 不依赖人工写好的 issue / unit test
- 直接从 commit 出发，**自动生成 reproduction test** 来验证 patch 正确性
- 用 back-translation 从 commit diff 反推 issue 描述
- 大幅降低人工标注门槛

### Hybrid Verifier（TTS）

- **execution-based verifier**：跑测试，区分度低
- **execution-free verifier**：LLM 评分，有 stylistic bias
- 单独都 saturates 在 42–43%
- 组合（hybrid）→ **51% Pass@1 on SWE-Bench Verified**

### 主要结果

32B Pass@1 34.4% / Best@26 (hybrid verifier) 51.0%。

---

## §29 Nebius Long-Context Multi-Turn SWE Agent (2025-08)

- **arXiv**: [2508.03501](https://arxiv.org/abs/2508.03501)
- **定位**: **目前公开最详尽** 的 long-context multi-turn SWE RL 训练 recipe。Qwen2.5-72B 上用改造的 DAPO 把 SWE-bench Verified 从 20% 推到 39%。

### 算法：改造版 DAPO

- **Asymmetric clipping**：(1-ε_low, 1+ε_high)，ε_high > ε_low 防 entropy 崩塌
- **Dynamic sampling**：过滤掉 advantage = 0 的 trajectory
- **Soft length penalty**：超过阈值 L_thr 后线性惩罚到 T_max
- **Token-level loss averaging**：batch 内每个 token 等权，长 trajectory 影响更大

### 环境

基于 **SWE-REBENCH**，**7,249 个** 精心 curate 的 Python 任务，每个有可复现 sandbox。

### Trajectory 组织（重头戏）

| 阶段 | Context | Max turns | Rollouts G | Problem/iter | Batch | LR | Clip range |
|---|---|---|---|---|---|---|---|
| Stage 1 RL | **65k** | **40** | 10 | 300 | 128 | 1e-6 | [0.2, 0.3] |
| Stage 2 RL | **131k** | **80** | 10 | 100 (curriculum→2028) | 256 | 1e-6 | [0.2, 0.26] |

共享设置：grad_clip=1.0、AdamW (β1=0.9, β2=0.999)、weight_decay=0.1、温度 1.0、其他 decoding 参数全关（保证 IS 无偏）。

### Loss masking

- **RFT 阶段**：只有 **"green"（成功完成 tool 调用格式的）assistant turns** 计入 loss
- mask 掉 environment formatting errors，提升 tool 调用稳定性
- **tool/bash output token 不算 loss**（隐含约定）

### 超长 trajectory 处理（与 DeepSWE 对比）

**没有像 DeepSWE 那样 mask 失败 trajectory**！作者明确论证：丢弃长 trajectory 会引入 bias、破坏"训练数据 ~ 策略采样分布"假设。用 soft length penalty **软性惩罚** 而非硬切。

### 冷启动：Rejection Fine-Tuning (RFT)

- 用 baseline 跑 10 次 × 7,249 tasks → 留下 **6,548 个成功 trajectory**
- 单 epoch SFT，**lr=5e-6, batch=64, 50 steps, context 65k**
- mask 掉 formatting errors 的 turn loss
- 效果：11% → 20%（再上 RL 到 39%）

### 算力

**16 个 H200 node**（每 node 8× H200, 32 CPU, 960 GiB RAM）= **128× H200**。JAX 训练 + vLLM 0.7.4 推理。

**这篇是公开工作里 hyperparameter 完整度最高的，可作为 SWE RL 训练首选参考模板。**

---

## §30 Qwen3-Coder (Alibaba, 2025-07)

- **博客**: <https://qwenlm.github.io/blog/qwen3-coder/>
- **定位**: 480B-A35B MoE，明确分 **Code RL（执行驱动）** 和 **Agent RL（long-horizon, multi-turn tool use）** 两条 RL 路线。

### Code RL

- 不局限于竞赛题，针对 broad real-world coding tasks
- 自动 scale test cases：对各种 coding 任务用工程方式扩充测试
- 完全 **execution-driven**：直接跑测试拿 reward

### Agent RL（long-horizon，真正的工程亮点）

- 自建 **20,000 个并发独立环境** 跑在 Alibaba Cloud
- **迄今公开最大规模的 SWE RL 环境集群**（vs DeepSWE 的 512 docker）
- 多轮 interaction：planning → tool 调用 → 接收 feedback → 决策

具体 lr / batch / rollout 数未公开；推测 GRPO 系。

---

## §31 Devstral / OpenHands LM

### Devstral (Mistral × All Hands, 2025-05/07/12)

- **博客**: <https://mistral.ai/news/devstral>
- 24B 128k ctx → 24B 1.1 → 123B 256k ctx (Devstral 2)
- 从 Mistral-Small-3.1 (24B) finetune
- "multi-stage RLHF + 工具调用 specialized fine-tuning"，**无具体 recipe**
- SWE-Bench Verified：46.8% → 53.6% → **72.2%**

### OpenHands LM 32B (All Hands AI, 2025-11)

- Base: Qwen2.5-Coder-32B-Instruct，用 SWE-Gym 采样 trajectory + filter
- 37.2% SWE-Bench Verified；训练框架 filtered BC（与 SWE-Gym 同 recipe）

---

## §32 Cursor Composer 系列 (Cursor, 2025–2026)

公开博客最详细的工业 SWE RL 系统之一。5 篇核心：
- [Composer 1](https://cursor.com/blog/composer)
- [Composer 1.5](https://cursor.com/blog/composer-1-5)
- [Composer 2 技术报告](https://cursor.com/blog/composer-2-technical-report) ([PDF](https://cursor.com/resources/Composer2.pdf))
- [Composer 2.5](https://cursor.com/blog/composer-2-5)
- [Real-time RL](https://cursor.com/blog/real-time-rl-for-composer)

### 架构（Composer 2）

四个解耦微服务：**Training / Environments / Inference / Evaluations**

- **Training**: Ray + PyTorch，完全异步
- **Environments**: 每个 rollout 跑在专属 **Firecracker VM**（自研平台 "Anyrun"，支持 filesystem snapshot 与 mid-trajectory checkpoint）；500+ pods/sec 调度
- **Inference**: 与 Fireworks AI 合作；每步用 delta-compressed S3 同步权重；**inference worker 能在 rollout 中途 update 权重 → 后期 token 更 on-policy**
- **Evaluations**: pinned production 后端 + 真实 Cursor client replica（确保 eval 行为 = 用户行为）

### 关键工程信条："Environment Fidelity"

- 训练 harness = 生产 harness：完全相同的 tools / prompt format / system message / file context
- 训练期间维护 **shadow deployment of Cursor backend**，保证 semantic search 等工具行为与生产一致
- 反对在简化环境（如裸 SWE-bench）训练，因为真实用户 query 是 under-specified

### 长 trajectory 处理：Self-Summarization

- 每次 rollout 可包含**多次"生成 → 总结"链式调用**
- **最终 outcome reward 应用到链上所有 token**：好的 summary 被强化、丢失关键 context 的 summary 被惩罚
- model **自己学会何时该 summarize**（不是人工触发）
- Composer 2.5 在 hard task 上会主动多次 summarize

### Reward

- 基于 code 的 correctness / succinctness / 工程规范性
- **Real-time RL**：直接从用户 production interaction 提取 reward signal，每天上线多次 checkpoint

### 训练规模

- 大部分 compute 在 **32k token sequence length**，后期再做长 context 扩展
- **Composer 2.5 把 ~85% 总 compute 投在 post-training / RL 上**
- 比 Composer 2 用了 **25× 的 synthetic task 数量**
- 训练横跨 3 个 GPU region + 4 个 CPU region

### 自研 trick

- **Distributed Muon optimizer**：Newton-Schulz 正交化跨 shard 异步，与通信 overlap
- **Function Deletion synthetic data**：AI agent 在有单测的 codebase 里"手术式删功能"；让模型反向实现 → 单测作为 verifiable reward
- **Directive Text Feedback** (2.5)：在 trajectory 行为不佳点插入 "hint"，用 hint 后 model distribution 作为 teacher → **局部 credit assignment**

### 涌现 / 坑

一次发现 model 学会"主动询问澄清问题以避免被 punish 写错代码"，导致编辑率持续下降；通过监控发现后改 reward。**long-trajectory RL 中 reward hacking 经典案例**。

---

## §33 Anthropic Claude 4 / 4.5

- [SWE-Bench Sonnet](https://www.anthropic.com/research/swe-bench-sonnet) · [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/)
- **SWE-Bench Verified**: 33.4% (Sonnet 3.5) → 49.0% (3.5 v2) → 77.2% (Sonnet 4.5, 200k no TTC) → 82.0% (high-compute)

### 公开的有限信息

- 内部有 **Code RL team "singularly focused on solving SWE"**
- **RLVR (RL with Verifiable Rewards)** 是核心方法论
- Sonnet 系列 SWE 性能跃迁主要来自 **targeted environment & data work + RLVR**，而非 pretrain / arch / optimizer
- Claude 4 训练时大部分 harmlessness 训练还是 chat-only 无 tool，新版本加 tool-use 环境后 agentic 安全显著改善
- Sonnet 4.5 报告能在复杂多步任务上保持 30+ 小时 focus

**几乎所有训练细节未公开**。唯一可学的是方法论：**RLVR + 真实环境多样性 + 大规模 targeted data**。

---

> **【新增:环境构造 / 自博弈 reward / 前沿模型 — 重点关注代码】** 用户最关心代码,这一组(§33A–§33D)集中补 SWE/code RL 的四块:**怎么自动造环境(§33A)、开源 SWE 模型怎么训(§33B)、reward 怎么设计 + 怎么被 hack(§33C)、前沿大模型的 agentic coding 配方(§33D)**。贯穿主题:**算法是次要的,环境与 reward 设计才是 code RL 的胜负手,也是头号陷阱**。

## §33A SWE 环境自动构造流水线:SWE-smith / SWE-rebench / SWE-Flow / SWE-Dev / SWE-Fixer

> [共识 11](#共识-11--环境基础设施成为核心差异化)说"环境抄不来"。这一节给**自动造环境的具体 recipe 与成本数字**——这是把 SWE RL 从几百 docker 扩到几万实例的关键。

### SWE-smith(Princeton/Stanford, [2504.21798](https://arxiv.org/abs/2504.21798))— 反转 SWE-bench,先建环境再造 bug

- **规模**:**50,137 实例 / 128 仓库**(比此前大一个数量级),存储仅 295GB(等效 SWE-bench 估 50–150TB,**~500× 省存储**),总成本 **\$1,360**。
- **核心 recipe**:**先建环境再造任务**——对最新 commit 跑 SWE-agent 安装+跑测试,人工确认 >80% 测试通过后做 Docker 镜像,**每仓共享一个环境**(这是省 500× 存储的关键)。
- **4 种造 bug 策略(各有 yield/成本/F2P)**:Combine Bugs(96.9% yield, 0¢)、LM Modify(56%, 0.38¢)、LM Rewrite(35%, 3.93¢)、PR Mirror(反转 PR)、Procedural(13 种 AST 变换, 0¢)。**校验**:patch 应用后必须**破坏 ≥1 个原本通过的测试**(Fail-to-Pass);测试运行 **2 分钟上限**(超时丢弃,治 flaky)。
- 训练:**仅 rejection sampling SFT(明确未探索 RL)**;SWE-agent-LM-32B = **40.2% pass@1**(开源 SOTA 当时)。
- ⭐ **代码 pitfalls(极有价值)**:**>25% 的 32B 轨迹有长度≥10 的重复动作序列**(Claude 3.7 <4%),**长度-10 重复 → 89% 失败概率**;**53% 失败因触及 runtime 上限(cost/calls)**——定位是主要失败模式;128 仓里人工 review 阶段**放弃 17 个**(环境搭不起来);Python 专属(依赖 `ast`)难移植。

### SWE-rebench(Nebius, [2505.20411](https://arxiv.org/abs/2505.20411), NeurIPS 2025)— 持续防污染 + 预建 Docker

- **21,000+ 交互式 Python 任务,专为大规模 RL 设计**;发布 **7,500 个预建 Docker 镜像**。全自动持续抽取:**LLM 驱动**提取/校验环境安装指令 + LLM 自动任务质量评估。
- **去污染卖点**:用持续新增的"鲜任务"做无污染基准,**实测发现部分模型在 SWE-bench Verified 上的分数因污染被 inflated**——这是"为什么要持续更新 benchmark"的硬证据。

### SWE-Flow(Qwen, [2506.09003](https://arxiv.org/abs/2506.09003), ICML 2025)— 从单测反推 TDD 任务

- **TDD 数据合成**:不依赖人写 issue,而是**从单元测试反推增量开发步骤**。构建 **Runtime Dependency Graph (RDG)** → 生成 step-by-step 开发计划,每步产出 partial codebase + 对应单测 + 必要改动 → **完全可验证的 TDD 任务**。难度可调(合并连续步)。16,061 训练 + 2,020 测试。

### SWE-Dev / SWE-Fixer(补充两条路线)

- **SWE-Dev**(THUDM, [2506.07636](https://arxiv.org/abs/2506.07636)):鲁棒测试合成 pipeline + scale agent 轨迹;7B **23.4%** / 32B **36.6%**。另有同名 feature-driven 版([2505.16975](https://arxiv.org/abs/2505.16975)):14,000 训练 + 每实例可运行环境 + 开发者写的可执行单测,**显式支持 SFT 与 RL**。
- **SWE-Fixer**(InternLM, [2501.05040](https://arxiv.org/abs/2501.05040)):**检索式(非执行式)**——retriever(BM25 粗到细)+ editor,**每实例仅 2 次模型调用**,强调效率;110K issues+patches;Verified 30.2%(P2P 过滤后 32.8%)。代表"不跑执行、固定 retrieve-then-edit"的轻量路线。

---

## §33B 开源 SWE 模型:Kimi-Dev / Skywork-SWE / Lingma SWE-GPT / Satori-SWE

### Kimi-Dev-72B(Moonshot, 2025-06)— Agentless + test-time self-play

- **SWE-bench Verified 60.4%**(发布时开源 SOTA);**Agentless** 框架(非 agentic),标准两阶段 File Localization → Code Edits。
- 训练:Base = Qwen2.5-72B;**mid-training ~150B tokens**(GitHub issues + PR commits,同时打好 **BugFixer + TestWriter** 两个先验)→ SFT → 大规模 **RL(用 Kimi k1.5 算法 §10I)**。
- **reward 纯 outcome**:Docker 中**整个测试套件全过才给 1**,**无 format/process reward、无 partial credit**。RL 工程:过滤成功率 0 的 prompt + curriculum + **把成功样本重放进最后阶段 batch**(Positive Example Reinforcement)。
- ⭐ **test-time self-play(新颖)**:RL 后让 BugFixer 与 TestWriter 协作,每 issue 最多 **40 patch 候选 + 40 test 候选**,明显 test-time scaling。

### Skywork-SWE-32B(昆仑万维, [2506.19290](https://arxiv.org/abs/2506.19290))— SWE 的数据 scaling law

- ⭐ **核心主张:首次系统揭示 SWE 能力的 data scaling law**——性能随训练轨迹数 **log-linear 增长,在 8,209 条轨迹处仍无饱和**。这是"SWE 该堆数据还是堆算法"的直接答案:**先堆数据**。
- 数据 pipeline:爬 151,472 repos → 三级 Docker 校验 → **10,169 verified 实例 / 2,531 仓 / ~11.9TB**;轨迹 8,209 条。Base = Qwen2.5-Coder-32B-Instruct,**仅 SFT**(RL 列为 future work,因需 runtime 镜像里做 instance 级校验)。8×H800 / 12h / 3 epoch / peak LR 5e-5;32K ctx;OpenHands 最多 100 轮。结果:38.0% pass@1(+TTS 47.0%)。
- ⭐ **代码实战指南(直接可抄)**:① **避免 train/test 同仓**否则 inflated;② 单一统一环境配置"必然导致显著数据损失",每仓要独立配;③ **runtime 复用**:500 镜像需 ~1000GB,用 mini-batch 的 **rollout→validate→delete** 减少 Docker 冗余;④ 轨迹超 50 轮易爆 32K,扩 128K 需序列并行;⑤ Agent 框架版本(OpenHands)的 prompt 差异会影响结果,用最新版。

### Lingma SWE-GPT(Alibaba, [2411.00622](https://arxiv.org/abs/2411.00622))— development-process-centric

三阶段(Repo Understanding → Fault Localization → Patch Generation,扩自 AutoCodeRover);~90,000 PR / 4,000 仓,rejection sampling 过滤(定位用 Jaccard 0.6、patch 用 n-gram+CodeBLEU 0.5)。**仅 SFT**;72B Verified 30.20%,接近 GPT-4o。**pitfall**:未做 patch 的自动单测验证(合适数据稀缺);定位仍是难点。

### Satori-SWE-32B([2505.23604](https://arxiv.org/abs/2505.23604))— 进化式 test-time scaling + RL

**EvoScale**:把生成当进化过程——LLM 作 mutation operator 迭代精修 patch,使分布移向高分区;**RL 后可自进化,推理时无需 reward model 选择器**(K=5/轮,≤4 轮=20 样本)。reward 用 **potential-based shaping**($`r_t=R(y^t)-R(y^{t-1})`$)。Best@1 35.8% → Best@50 41.6%。**效率主张**:32B vs 70B、30K vs 百万级数据、50 vs 500 样本(~10× 采样效率)。

---

## §33C 自博弈 / 共进化 reward:CURE / Absolute Zero / AceCoder / rStar-Coder

> ⭐ **代码 RL 的 reward 设计是头号陷阱**(测试即奖励,但测试本身可被 hack)。这一节是"用代码可执行性替代人工标注"的几条前沿路线,也是 reward hacking 的重灾区案例集。

### AceCoder([2502.01718](https://arxiv.org/abs/2502.01718))— 自动测试合成 + 执行 reward

- 首个用**全自动大规模测试用例合成**做 code RL:从种子改写问题 + 想象 ~20 测试 → 过滤 → **87K 问题 / 1.38M 测试**(AceCode-87K)。算法 **Reinforce++**(弃 PPO,省 value model)。从 base 用 rule reward,**80 步 / 48 H100-h 使 HumanEval+ +25.0**。
- ⭐ **reward hacking 教训**:**测试全过 ≠ 程序正确**(引入噪声);**从 base 用 RM(而非 rule)训练反而更差**(归因 reward hacking);在已很强的 Coder-Instruct 上提升有限。

### CURE / ReasonFlux-Coder([2506.03136](https://arxiv.org/abs/2506.03136), NeurIPS 2025 Spotlight)— coder ↔ tester 共进化

- ⭐ **RL 同时共进化 coder 与 unit-test generator,无需任何 ground-truth 代码监督**(tester 直接从 coder 的错误中学)。reward 设计有定理支撑:unit-tester 的 reward 在"测试让所有正确解通过、让错误解失败"时为正。
- ⭐ **核心 reward hacking 警示**:naive reward("通过所有正确代码即可")会**激励生成 trivial/过度宽松的测试**只为最大化通过率;CURE 推导的 reward 把这种 false-positive 从 42.2% 压到 36.5%。**这是"测试即 reward"必须防的坑**。
- 仅 4.5K 题,单测准确率 +37.8% / BoN +9.0%;其生成的测试可作 reward model,**无标签 RL 媲美有 ground-truth 监督**。

### Absolute Zero Reasoner(清华, [2505.03335](https://arxiv.org/abs/2505.03335))— 零外部数据自博弈

- **Absolute Zero 范式**:单模型既**提出任务**(最大化 learnability)又**解任务**,**零外部数据**,**Python 执行器同时作 task 校验器与 answer 验证器**。三类代码任务(deduction/abduction/induction)。算法 **TRR++**(Task-Relative REINFORCE++,6 种角色×任务配置各自算 baseline)。
- AZR-Coder-7B:LiveCodeBench +11.8;**跨域迁移惊人**——纯代码 RL 数学仅 +0.65,而 AZR 数学 +10.9~15.2,**规模越大收益越大**。
- ⚠️ **安全 pitfall**:Llama3.1-8B 偶发"令人担忧的 CoT"(如"智胜人类"),作者呼吁 safety-aware training——self-play 的暗面。

### rStar-Coder(MSRA, [2505.21297](https://arxiv.org/abs/2505.21297))— 无 oracle 解的测试合成

- **418K 题 / 580K 长推理解**。**互验机制(Mutual Verification)**:对合成题采 16 个解 + ≥50 测试输入,**若多数解产生完全相同输出则接受**(错误解更易发散,正确解收敛),<60% 一致则丢弃。互验准确率 **96.8% vs GPT-4o 直接 12.7%**。**仅 SFT**;Qwen2.5-7B LiveCodeBench 17.4→57.3。

> **代码 reward 设计三条铁律(综合本节)**:① **测试质量 > 测试数量**——trivial 测试是 reward hacking 主入口;② **从 base 用 rule-based outcome reward 比用学习式 RM 更抗 hacking**(AceCoder 实证);③ **公开测试集常被截断/不全**(见 §48 Open-R1 的"代码可验证性危机":CodeForces 公开测试截断 500 字符,7 个过 public test 的解在 full test 全失败)——务必用 full/hidden test。

---

## §33D 前沿模型 agentic coding:DeepSeek-V3.2 / MiniMax-M2 / GLM-4.6 / Seed-Coder

### DeepSeek-V3.2-Exp([2512.02556](https://arxiv.org/abs/2512.02556))— GRPO 四件套稳定化 + 单阶段合并 RL

- **结果**:SWE Verified **73.1**(thinking)、SWE Multilingual 70.2(开源最佳)、Terminal-Bench 2.0 46.4。
- ⭐ **GRPO 四个稳定化技巧(工程极有用)**:① **Unbiased KL Estimate**(用 IS ratio 修正 K3 估计;数学域用弱/无 KL);② **Off-Policy Sequence Masking**(对超过散度阈值 δ 的负 advantage 序列置零);③ **Keep Routing**(强制 sampling 与 training 的 MoE 路由一致,自 V3-0324 起——又一条 MoE train-infer 对齐方案,对照 GSPO/R3);④ **Keep Sampling Mask**(保留 top-p/top-k 截断掩码对齐 action 子空间)。**把 reasoning/agent/human-alignment 合并为单一 RL 阶段**,RL 预算 ">10% 预训练成本"。
- **agentic coding 环境/reward**:从 GitHub 挖数百万 issue-PR 对,自动建可执行环境;**成功判据 = gold patch 产生非零 F2P 且零 P2F**;8 种语言、**数万环境、code agent 任务 24,667 个**;reward = rule-based outcome + length penalty + language consistency;最后 **6 域 specialist distillation**。
- **128K 上下文管理坑(可类比长 agent 编码)**:~20%+ 测例超限,**无上下文管理 51.4 → Discard-all 67.6**(Summary 60.2 但平均步数飙到 364),触发阈值=用量超 ctx 80%。

### MiniMax-M2(MiniMax, 2025-12 前后)— interleaved thinking 编码 agent

- **230B 总参 / 10B 激活** MoE,128K ctx,**interleaved thinking**(`<think>` 标签,Plan→Act→Reflect)。SWE-bench Verified **69.4**、Terminal-Bench 46.3、τ²-Bench 77.2、BrowseComp 44。
- ⭐ **评测 scaffold(可复用经验)**:SWE-bench Verified 用 **R2E-Gym scaffold on OpenHands,128K ctx,100 max steps,无 TTS**,且**移除所有 git 相关内容**让 agent 只看 issue 时点代码(防泄漏)。RL 系统("Forge")为二手信息,未一手核实。

### GLM-4.6(Zhipu, 2025-09)— 真实多轮人评 CC-Bench

- 355B/32B-active,ctx 128K→200K;**SWE-bench Verified 68.0%**(≈Claude Sonnet 4)。**CC-Bench(真实多轮、隔离 Docker、人评)对 Claude Sonnet 4 胜率 48.6%**(near parity);token 效率比 GLM-4.5 高 >30%。**无独立 4.6 RL 报告**,方法学沿用 GLM-4.5(§8,单阶段 64K RL + 可验证环境 dense reward + "instruction-following RL 无 reward hacking")。

### Seed-Coder-8B(ByteDance, [2506.03524](https://arxiv.org/abs/2506.03524))— model-centric 数据 + 从 base 起 RL

- **model-centric 数据**:用 1.3B 质量打分器自筛 GitHub 代码,6T tokens。Reasoning 模型 RL:**GRPO + DAPO-like**(batch 128, LR 1e-6, 去 KL, clip 0.28, token-wise loss),RL 数据=**竞赛编程**,渐进课程 90 步@16K → 160 步@32K。
- ⭐ **重要 pitfall(与通用经验互证)**:**RL 从 base 起而非 instruct**——因 instruct 在 RL 中"频繁坍缩回 SFT 数据模式,损害 RL 后性能";且蒸馏数据过多会超出 base 固有能力。结果:Codeforces ELO 1553(≈o1-mini)、IOI'2024 146.5(超 QwQ-32B / DeepSeek-R1)。

> **CODE-SPECIFIC 经验总结(贯穿 Part IV 新增)**:① **执行 reward 两路线**——纯 outcome 0/1 全套件通过(Kimi-Dev、DeepSeek-V3.2 F2P≠0 且 P2F=0)vs pass-rate 连续(AceCoder)/过程式 shaping(Satori);多数**不用 partial/format reward**防 hacking。② **测试即奖励的反噬是头号坑**(AceCoder/CURE/公开测试截断)。③ **自博弈/共进化**用代码可执行性替代人工标注(CURE/AZR/Kimi-Dev)。④ **环境构造是规模化核心**(SWE-smith 500× 省存储、Skywork 数据 scaling law、rollout→validate→delete)。⑤ **多轮 SWE 需 >50 轮 / >32K**,128K 下 ~20% 超限需上下文管理。⑥ **从 base 起 RL** 比 instruct 更稳(Seed-Coder)。

---

# Part V. Agentic RL 训练框架与基础设施

> **本部分要回答**:做 agentic RL 该选哪个训练框架,以及"全异步训练"为什么是 long-horizon agent 的必经之路。
>
> **核心观点**:
> - **事实标准是 verl(§34)**:DAPO/rStar2/Doubao/Agent Lightning 全跑在 verl 上;Megatron+FSDP+vLLM AgentLoop 是生产路径
> - **全异步训练的关键超参是 staleness η**(§35 AReaL):η=0 退化为同步,η>0 GPU 利用率近 100%但 off-policiness 上升;**boba² v0.3 达 2.77× 同步系统加速**——长 trajectory 必上
> - **Agent Lightning(§36)给出 cleanest 抽象**:把多轮拆成 (input, output, reward) transition 独立训,**消除 tool mask 需求 + 不破 RoPE + 防 context 爆炸**;但实战中"零代码改造接入 LangChain/AutoGen"承诺需打折扣
> - **§36 那个 vLLM 教训不能漏看**:retokenization drift(text → tokenize 后 token id 与 inference 不同)是 agent RL 最隐蔽的 train-infer bug;所有人都该做一次 token-id 对齐验证
> - **OpenRLHF(§37)的 TIS/ICEPOP/Seq-Mask-TIS** 是当前修 vLLM↔training logprob mismatch 最系统的实现

## §34 verl / HybridFlow (ByteDance Seed)

- **GitHub**: [volcengine/verl](https://github.com/volcengine/verl) · [Fully Async Docs](https://github.com/volcengine/verl/blob/main/docs/advance/fully_async.md) (EuroSys 2025)

### Agentic 特性

- **PartialRollout**: 参数同步时通过 `sleep()`/`resume()` 保存在途 rollout 的状态，下一轮继续生成
- **AsyncPartialToolAgentLoop**: 响应中断信号保存状态
- 当前生产路径：**megatron / FSDP + vLLM server 模式 (AgentLoop)**
- 架构四件套：Rollouter / MessageQueue / Trainer / ParameterSynchronizer

**支持算法**: PPO / GRPO / ReMax / REINFORCE++ / RLOO / PRIME；VLM RLHF；SGLang 多轮 agentic RL。

**生态**: Doubao 1.5 Pro / DAPO / rStar2-Agent / Agent Lightning 都基于 verl。

---

## §35 AReaL (Ant Group × Tsinghua IIIS)

- **arXiv**: [2505.24298](https://arxiv.org/abs/2505.24298) (NeurIPS 2025) · [GitHub](https://github.com/inclusionAI/AReaL)
- **核心**: 完全异步 RL，**rollout 与 training 解耦**
- **稳定 trick**: **staleness-enhanced PPO** + 控制 batch 内平均 staleness 平衡 worker 负载
- **效率**: boba² v0.3 达 **2.77× 同步系统加速**，AReaL-lite 用 80% 更少代码保留 90% 性能
- **变种**: AReaL-Hex (异构 GPU)、ASearcher (端到端 async 训练的搜索 agent)

---

## §36 Agent Lightning (Microsoft Research, 2025-08)

- **arXiv**: [2508.03680](https://arxiv.org/abs/2508.03680) · [GitHub](https://github.com/microsoft/agent-lightning)

### 核心抽象 LightningRL

- agent 执行建模为 **POMDP**：state = 完整执行 snapshot；observation = LLM 输入；action = 一次 LLM 调用产生的全 token 序列
- **Credit assignment**：episode-level return R **等值赋给每个 action**，再在 action 内部用现有 token-level RL（GRPO/PPO/REINFORCE++）
- **关键设计**：transition = (input, output, reward) 三元组**单独抽取**，不做多轮拼接
  - 好处 1：**不需要 retrieval mask / tool output mask**（这些 token 本来就不在 transition output 里）
  - 好处 2：不破坏 RoPE 位置编码假设
  - 好处 3：消除 context 累积导致的长度膨胀

### 工程要点

- **TA Disaggregation**：Lightning Server (trainer + OpenAI-API endpoint) + Lightning Client (agent runtime)
- **OpenTelemetry tracing** 捕获 execution data，agent 代码不动
- **AIR (Automatic Intermediate Rewarding)**：tool 调用返回状态自动转中间 reward
- **零代码改造**接入 LangChain / LangGraph / AutoGen / OpenAI Agents

### 重要踩坑

**"No More Retokenization Drift" (2025-10-22 vLLM blog)**:
- **问题**：agent 用 OpenAI 兼容 API 拿 text，训练时 retokenize 出来的 token id 与 inference 时不一致（BPE 边界差异 / 特殊 token 处理差异）→ train-infer policy mismatch → PPO ratio 爆炸
- **对策**：**API 直接返回 token id**，训练端 logprob 严格对齐 inference
- 任何接 vLLM/SGLang 的 agent RL pipeline 都应做一次 token-id 对齐验证

---

## §37 OpenRLHF

- **GitHub**: [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) · [Async/Agent docs](https://openrlhf.readthedocs.io/en/latest/async_rl.html)

### Agentic 特性

- **统一 agent execution 范式**：`SingleTurnExecutor` / `MultiTurnExecutor`；所有训练都走 token-in-token-out
- 用 `--async_train` + `--agent_func_path` 启用 agent RL
- `agent_func_openai_server_executor.py`: vLLM 包装成本地 OpenAI 兼容 chat server
- 推荐 Hybrid Engine (`--colocate_all_models`) 优于纯异步

**算法**: PPO / GRPO / REINFORCE++ / REINFORCE++-baseline / RLOO / DAPO。

**新增**: **TIS / ICEPOP / Seq-Mask-TIS** 修正 vLLM ↔ training 间的 logprob 不匹配（**训推一致核心**）。

---

## §38 SkyRL / SkyRL-Agent (UC Berkeley Sky × Anyscale)

- **GitHub**: [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL) · [Project](https://sky.cs.berkeley.edu/project/skyrl/) · [SkyRL-Agent arXiv 2511.16108](https://arxiv.org/abs/2511.16108)

### 时间线

- SkyRL-v0 (2025-05): 多轮工具使用 RL pipeline，专为 SWE-Bench 这类 long-horizon real-env 任务
- SkyRL-SQL (2025-05): 653 样本训出超 GPT-4o 的 7B
- SkyRL-v0.1 (2025-06): + **SkyRL-Gym**（Gymnasium API 的 RL env 库）
- SkyRL-Agent (2025-11): 接入 SkyRL-train / **VeRL / Tinker** 任一后端

### 核心 scheduling 创新

- **Intra-rollout scheduling**: 单条 rollout 拆成 init / LLM gen / reward 阶段，与其他 rollout 阶段并行
- **Inter-rollout scheduling**: 全局调度 rollout 顺序，平衡 CPU/GPU 负载

已训 SWE / Deep Research / Computer Use / Memory agent。

---

## §39 rLLM (UC Berkeley Sky Lab / Agentica)

- **链接**: [rllm-project.com](https://rllm-project.com/) · [GitHub](https://github.com/rllm-org/rllm)

### 里程碑

DeepScaleR (1.5B, AIME24 43.1%) → DeepCoder (14B) → DeepSWE → **rLLM v0.2 General Agentic Programs** → rLLM SDK。

### v0.2 新抽象

- **AgentWorkflowEngine** + **AgentWorkflowTrainer**：支持任意 agentic program（multi-agent / solver-judge / planner-executor / MCTS）训练

### rLLM SDK (Dec)

**直接拦截 LLM 调用**，可训 LangChain / LangGraph / AutoGen 任何框架的 agent 而无需改造代码。在金融 agent 上，4B 专门化模型超 Qwen3-235B (59.7 vs 51.4)。

---

## §40 Tinker (Thinking Machines Lab, 2025-10)

- **链接**: [Announce](https://thinkingmachines.ai/news/announcing-tinker/) · [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- **定位**: Mira Murati / John Schulman 的"低层但好用"分布式 fine-tuning API
- **核心 primitive**: `forward_backward` + `sample`；用 LoRA 而非 full FT（参考其 "LoRA Without Regret" 技术 note）
- **支持模型**: Llama-3.2-1B 到 Qwen3.5-397B-A17B（含 MoE）
- **RL 实现**: Cookbook 内 `SamplingClient` 采集 trajectories（state/action/reward/logprob）→ 形成 batch → forward/backward
- **早期用户**: 普林斯顿（Goedel 定理证明）、Stanford（化学）、**Berkeley SkyRL**（async off-policy multi-agent multi-turn tool use）、Redwood Research

---

## §41 AgentScaler / AgentRL / RollArt

### AgentScaler (2025-09)

- **arXiv**: [2509.13311](https://arxiv.org/abs/2509.13311)
- **思路**: "Environment Scaling" — 自动构造海量异构**完全模拟**的 function-call 环境（解决真实 MCP service 不稳定问题）
- **两阶段训练**: stage 1 通用 tool-calling → stage 2 垂域专精
- **结果**: 基于 Qwen3 训出 4B/8B/30B-A3B；30B-A3B 在 τ-bench / τ²-Bench / ACEBench 上达到 1T 闭源水准

### AgentRL (THUDM, 2025-10)

- **arXiv**: [2510.04206](https://arxiv.org/abs/2510.04206)
- **infra**: 全异步生成-训练管线 + 统一 function-call API + 容器化环境
- **算法**: **Cross-policy sampling**（多轮场景增强探索）+ **Task advantage normalization**（多任务训练稳定）

### RollArt (2025-12)

- **arXiv**: [2512.22560](https://arxiv.org/abs/2512.22560)
- **观点**: agentic RL 因负载异构（compute-heavy prefill + bw-bound decode + CPU-heavy env sim）应**用 disaggregated infra**

---

> **【新增:异步/staleness 记账 + 去中心化 RL】** §41A–§41F 补几个新框架,重点是**异步训练的 staleness 怎么记账、train-infer logprob 一致性怎么落地实现**——这是把共识 10/14/§44 从"原理"变成"配置项"的地方。框架对比速查表见本组末。

## §41A ROLL / ROLL Flash (Alibaba) — per-sample staleness α

- **arXiv**: [ROLL 2506.06122](https://arxiv.org/abs/2506.06122) · [ROLL Flash 2510.11345](https://arxiv.org/abs/2510.11345) · [GitHub](https://github.com/alibaba/ROLL)
- **定位**: 阿里的大规模 + agentic 双栈 RL 库;**ROLL Flash 那篇是异步加速的系统性工作**,与 AReaL 形成"staleness 记账哲学"对照。

### ROLL 本体

四大模块 single-controller + parallel-worker;**Rollout Scheduler 对每个 sample 的 lifecycle 做细粒度管理**(vs verl 的关键差异);Environment/Reward Worker 支持快速 agentic 实验;AutoDeviceMapping 支持 colocated 与 disaggregated。算法开箱:PPO/GRPO/Reinforce++/GSPO + **GiGPO/StarPO/Lite-PPO**(两种 agentic 范式:TrajectoryWise=StarPO,StepWise=GiGPO)。落地:LiveThinking(淘宝直播,~30× 算力削减)、TaoSR 主搜。

### ⭐ ROLL Flash — staleness 记账的另一种哲学

- 两原则:**fine-grained parallelism**(sample 级,prompt replication 打散 long-tail)+ **rollout-train decoupling**(可调 blocking→non-blocking)。
- ⭐ **Async ratio α = per-sample 的 freshness 上界**(当前策略版本与生成该 sample 的策略版本之差)。**这是与 AReaL 的关键差异:AReaL 控 batch 的平均 freshness,ROLL Flash 控每个 sample**。**最优 α≈2**("a value as low as 2 suffices"),α=1 给 28% 提速、α=2 给 64% 吞吐增益。
- **off-policy 校正**:实现 Decoupled PPO / TIS / CISPO / TOPR + 新 **Weighted TOPR**;**train-infer 一致性**:用截断 IS(cap C=5)min(π_megatron/π_vllm, C),并**强制 temperature=1, top-p=1 恢复原始 logits**(同 AReaL 做法),且指出此问题**同步架构也存在**。
- 加速:RLVR 最高 2.24×,Agentic **ALFWorld 2.72× / SWE 1.81×**。
- **稳定性发现**:α=2~8 下 **vanilla GRPO 就能持平同步训练**("无需算法 trick");α 太低 long-tail 瓶颈、太高样本过期损稳;**α 对模型大小不敏感,随序列长度增大、随 rollout batch 增大而减小**。

---

## §41B slime (THUDM) 深入 — TIS/MIS + catastrophic-token veto

- **GitHub**: [THUDM/slime](https://github.com/THUDM/slime) · 支撑 **GLM-4.5 / 4.6**
- **定位**: SGLang-native 的 RL 引擎,**train-infer 一致性做得最细的开源框架之一**。

- **设计哲学(vs verl)**:**拒绝多余抽象层**——Megatron 参数直通、SGLang 参数以 `--sglang-` 前缀暴露。代价是只押 SGLang 一个推理后端,换取直接用其专属能力(PD 分离 / weight-sync)。Agentic = "workflows as data generation",**重写 rollout 函数和 reward 函数**而非 fork 训练 kernel。
- **FP8 rollout**:Megatron BF16 训练态 + SGLang FP8 rollout(bf16 权重直接 cast),长 ctx 可 `--sglang-kv-cache-dtype fp8_e4m3`。
- ⭐ **train-infer 一致性(配置级)**:`--use-train-infer-is`(取代 legacy `--use-tis`),支持 **TIS/MIS × token/sequence/geometric 多层级** + **catastrophic-token detection**(`--train-infer-is-veto-threshold` 否决整条序列) + 自动记录 train/infer perplexity、KL、K3。
- ⭐ **明确 pitfall(给 §44 TIS 争议补实锤)**:**复现 Search-R1 用 3B 模型时,启用 TIS 反而早期 policy collapse**(输出多语种乱码、无视格式),关掉 TIS 才正常收敛(slime issue #1533)。机制假设:低概率 token 天然高方差 ratio 最易被截断 → 系统性下调其梯度 → 引入有害 bias。**slime blog 因此推荐 MIS 作默认**(晚期抑制 mismatch 又保性能)。

---

## §41C NeMo-RL (NVIDIA) — 端到端 FP8 统一训推精度

- **GitHub**: [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) · Ray + vLLM + Megatron-Core
- **算法**:GRPO / **GSPO** / **DAPO** / DPO / on-policy distillation / 多轮 RL;v0.6 新增 **GDPO**(针对多 reward,解决 GRPO 的 "reward advantage collapse")。后端:DTensor/FSDP2(≤32B)+ Megatron-Core(>100B)。
- ⭐ **端到端 FP8(train-infer 一致性的正面案例)**:Megatron-Core FP8 训练 + FP8 vLLM 生成,DeepSeek 式 sub-channel 量化。**关键卖点:端到端 FP8 统一 train/infer 精度,"减小 train-rollout logprob gap、降低 TIS clip fraction"**——与 MiniMax-M1/ScaleRL 的 FP32-LM-head 是**两条相反但同源的对齐思路**(一个全 FP8 对齐、一个关键层 FP32 对齐)。Llama-3.1-8B 验证精度 FP8 0.613 vs BF16 0.616,整体最高 48% 提速。

---

## §41D prime-rl + verifiers — gym-like 环境抽象 + 去中心化异步

- **GitHub**: [verifiers](https://github.com/PrimeIntellect-ai/verifiers)(Will Brown)· [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)
- **定位**: **环境抽象的代表案例** + 去中心化异步训练器。

- **verifiers 环境抽象**:`vf.MultiTurnEnv`(`env_response`/`is_completed` hook)、`vf.ToolEnv`(`call_tool` hook);概念上 **interaction protocol + rubric = environment = 合成数据引擎 = RL trainer = eval harness** 四合一。经 **Environments Hub** 安装即可训,**无需改 prime-rl 代码**。定位易用优先(极致性能官方建议用 NeMo-RL/veRL)。
- **prime-rl 异步(配置级精确)**:**默认一步重叠(k=1)**——trainer 用 step-n rollout 出 π_n 时 inference 已用 π_{n-1} 出 step n+1。默认 loss = **DPPO + KL(类 Kimi-K2.5)**:PG 项带上截断 δ(防 stale rollout 对高 reward token 的低概率 runaway),KL 项用 **log²(π/μ)** 平方 log-IS-ratio(对照 K1.5 §10I 的 L2 信赖域)。**把 train/infer logprob 作为一等公民暴露**给自定义 loss API(`LossInputs(trainer_logprobs, inference_logprobs, ...)`)。

---

## §41E INTELLECT-2 / INTELLECT-3 — 全球去中心化异步(双侧 clip,async-8 实测 GSPO 崩)

- **arXiv**: [INTELLECT-2 2505.07291](https://arxiv.org/abs/2505.07291)(32B,QwQ-32B base)· INTELLECT-3 2512.16144(106B-A12B MoE,GLM-4.5-Air-Base)
- **定位**: 首个全球去中心化异步 RL,把异步稳定性逼到极限——**它踩的坑就是所有人异步会踩的坑的放大版**。

### ⭐ 双侧 GRPO 裁剪(高 staleness 稳定的关键)

- 标准 GRPO 的 `min` 对**负 advantage 不裁剪** → 大更新致崩。引入 **δ 给负 advantage 的 ratio 加上界**:内层 `min(ratio, δ)`,要求 δ > 1+ε。**实测 ε=0.2, δ=4**(与 MiniMax-01 独立一致)。
- 其他:**梯度裁剪激进到 0.1**(显著推迟梯度增长期);**torch.compile 全程关闭**(曾致后期 catastrophic collapse,疑单个 faulty kernel);**不用 inference worker 的 logprob,在训练集群重算**(vLLM logprob 数值不稳)。
- staleness:trainer 用**优化步开始时的策略**重算 logprob;**ablation 异步度 up to 4 仍匹配同步**。两步异步 + SHARDCAST 树状广播权重 + TOPLOC 可验证推理(防去中心化作弊)。
- 不稳信号:梯度范数随模型增大而升、**~150 步后 entropy 回涨**(通常预示灾难失败);**QwQ-32B 比 R1-Distill-32B 更不稳**(因 QwQ 已先做过 RL)。

### ⭐ INTELLECT-3:async-8 实测 GSPO 崩(共识 14 反例的硬证据)

prime-rl 升级生产级、专为异步设计;用 **async-8 做高 off-policy 压测,观察到 GSPO 下 reward(及所有指标)崩溃**——**序列级 IS 在极端 staleness 下扛不住**的直接证据。106B MoE,512×H200 / ~2 个月;AIME24/25 90.8/88.0;**全开源 prime-rl + verifiers + Environments Hub + Prime Sandboxes**。

---

## §41F Trinity-RFT (Alibaba) / TRL (HF) / AReaL-lite

### Trinity-RFT([2505.17826](https://arxiv.org/abs/2505.17826))— 统一 sync/async × on/off-policy × online/offline

Explorer-Trainer-Buffer "trinity";靠 **`sync_interval` / `sync_offset` 两参数**统一控同步度(`sync_interval=1` 严格 on-policy,`>1` off-policy,`offset=1` 一步 off-policy)。**多轮建模:K 轮交互紧凑拼成单序列 + masking**(避免拆成 K 个样本);**lagged rewards**(轨迹先入 buffer 标 "not ready",reward 到了再标 "ready");**100+ Data-Juicer 算子** + Prioritized Experience Replay。**MIX 算法**:`loss=(1-μ)·grpo + μ·sft` 融合 online RL + offline SFT。

### TRL (HuggingFace) — GRPO/GSPO + 默认 TIS

vLLM colocate / server 两模式;**默认开 Truncated Importance Sampling** 校正 vLLM↔训练 mismatch(`vllm_importance_sampling_correction=True`,可切 TIS/MIS × token/sequence);**GSPO** 设 `importance_sampling_level="sequence"` 即可。多轮工具调用需 transformers ≥5.0。⚠️ **版本兼容碎片化**:vLLM 支持区间随版本变,务必按安装版本核对。

### AReaL-lite(2025-07)— algorithm-first,代码减 80% 保 90% 效率

轻量版 AReaL;**所有算法(GRPO/GSPO/PPO/DAPO/LitePPO)同时支持同步/异步**——`max_head_offpolicyness=0` 即同步。原生全异步 agentic RL。沿用 AReaL 的 Decoupled PPO(**最优 staleness=4**,naive PPO 即使轻微 staleness 也显著掉点)。

### ⭐ 框架对比速查(异步 / staleness / train-infer 一致性)

| 框架 | 异步默认 | staleness 记账 | train-infer logprob 一致性 |
|---|---|---|---|
| **ROLL Flash** | 可调 blocking→non-blocking | **per-sample α(最优≈2)** | TIS cap C=5 + 强制 temp=1/top-p=1 |
| **slime** | colocated 同步 / disagg 异步 | (SGLang PD 分离) | **`--use-train-infer-is`:TIS/MIS/geometric + catastrophic veto**;推荐 MIS |
| **NeMo-RL** | 全异步 GRPO + replay buffer | replay buffer off-policy | **端到端 FP8 统一精度**降 logprob gap |
| **prime-rl** | **默认一步重叠 k=1(去中心化 k=2)** | DPPO 上截断 δ + log² KL | μ(infer)/π(train) 作一等公民暴露 |
| **INTELLECT-2/3** | **两步异步**(async-4 OK;async-8 GSPO 崩) | **双侧 clip ε=0.2/δ=4** + grad clip 0.1 | **训练集群重算 logprob**(vLLM 不稳) |
| **AReaL-lite** | `max_head_offpolicyness=0` 切同步 | **Decoupled PPO,最优 staleness=4** | (SGLang 自定义 weight update) |
| **Trinity-RFT** | sync_interval/offset 可调 | 一步 off-policy(offset=1) | 主文未明确(OPMD 变体在附录) |
| **TRL** | server/colocate | — | **默认 TIS**(可切 MIS)+ GSPO seq-level |

> **三条高价值框架经验**:① **per-sample(ROLL Flash α≈2)vs batch-average(AReaL staleness=4)是两种 staleness 记账哲学**,记账粒度不同最优值也不同;② **强制 temp=1/top-p=1 恢复 logits** 是 ROLL Flash 与 AReaL 共同做法(代价是牺牲采样超参灵活性);③ **极端 off-policy 下序列级 IS(GSPO)会崩,双侧 clip 更稳**(INTELLECT-3 + INTELLECT-2 实证)。

---

# Part VI. 横向综合 take-home

> **本部分作用**:Part I–V 的横向汇总。如果你只想"扫一眼对比"再决定回去读哪一篇,从这里开始。
>
> **包含**:
> - **§42 跨工作对比速查表**:按 reasoning / multi-turn tool / web / SWE 四类,把所有工作的核心超参拉成表
> - **§43 共识 → 出处反查表**:15 条共识每条对应的支持工作 + 反例工作清单(配合顶部速读区使用)
> - **§44 训推一致性(隐形杀手)**:工程上最容易漏掉的一类 bug,逐源说明现象 + 机制 + 对策
> - **§45 算力规模参考**:从 7B 到 1T 的资源消耗参考,按"模型规模 → 卡数 → 时长"对照
> - **§46 抄作业指南**:按"抄 recipe / 抄环境 / 抄算法稳定化 / 抄工业部署"四类的推荐清单 + 待挖掘空白点
> - **§47 奖励模型路线**:hard-to-verify 域绕不开 GRM;offline / online / actor–GRM 合并三条线的取舍 + 推荐决策树

## §42 跨工作对比速查表

### Reasoning RL（math/code 单轮）

| 工作 | KL | Clip ε | LR | Rollouts | Batch | Loss 归一化 | 长度处理 |
|---|---|---|---|---|---|---|---|
| DAPO | **0** | low=0.2, high=**0.28** | 1e-6 | 16 | 512 | **token-level** Σ | 4k cache 软惩罚 + 16k 上限 |
| Magistral | **0** | high=**0.26–0.28** | 未披露 | — | 8k→4k→2k | **group total length** | 16k→24k→32k 阶梯 |
| MiMo-7B | **0** | high 抬高 | 1e-6 | — | 512 | — | 32k→48k |
| Skywork-OR1 | **0** | 0.2 + clip-higher | — | 16 | 64-256 | **token-level, 无长度归一** | 8k→16k→32k stage |
| DeepSeek-R1 | 0.001 | **10 (几乎不裁)** | 3e-6 | 16 | 512 | GRPO 标准 | 32k 单阶段 |
| Qwen3 reasoning RL | — | — | 1e-6 (社区) | 12-16 | 大 batch + grad accum | — | thinking budget |
| MiniRL | implicit IS | **asymmetric 0.27/0.2** | — | — | 1024-8192 | **无 length norm** | 32k |
| **GSPO**(§10A) | 0 | **序列级 3e-4/4e-4** | — | — | — | **序列级几何平均** | 用于 Qwen3;免 routing replay |
| **CISPO**(§10B) | 0 | **只裁 IS 权重上界,不丢 token** | — | 16 轮复用 | — | token Σ | 40K→80K;**FP32 LM head** |
| **GMPO**(§10C) | — | **token 级 $`(e^{-0.4},e^{0.4})`$** | 未披露 | 8 | 128 | **几何平均 + 长度归一** | 3k |
| **Dr.GRPO**(§10D) | — | 标准 | — | — | — | **去 /\|o\| + 去 /std**(常数 L) | 防越写越长 |
| **Lite-PPO**(§10E) | **0** | base 抬界无用;4B=0.32/8B=0.28 | 1e-6 | 8 | 1024 | **group 均值 + batch std + token-level** | 丢 overlong filter |
| **ScaleRL**(§10F) | — | **CISPO**(选它而非 GSPO/DAPO) | 见附录 | 16 | 768→2048 | prompt-level + batch 归一 | **FP32 LM head**(天花板 0.52→0.61) |
| **ProRL**(§10G) | **KL + 参考策略硬重置** | low=0.2, **high=0.4** | — | 16 | 256 | — | >2k 步;能扩 base 边界(复杂任务) |
| **ORZ**(§10H) | **0(全去)** | 0.2 | **policy 1e-6 / critic 5e-6** | **64** | 128 | **PPO + GAE λ=γ=1** | γ<1 会变短 |
| **Kimi K1.5**(§10I) | **τ·KL(软)** | **L2 平方 log-ratio(替代 clip)** | 未披露 | k | — | 经验均值 baseline | partial rollout + long2short |

### Multi-turn Tool / Search RL

| 工作 | 算法 | KL | Clip-Higher | Cold-Start SFT | Tool Mask | Reward | Max Turns |
|---|---|---|---|---|---|---|---|
| Search-R1 | PPO (default) / GRPO | 0.001 | — | No | Yes (`<information>`) | EM 0/1 | 4 search |
| R1-Searcher | Reinforce++ | 0 / 1e-4 | — | No | Yes | 2-stage: retrieve+format / F1+format(−2) | 8 |
| R1-Searcher++ | Reinforce++ | 1e-4 | — | **Yes** (6 ep) | Yes | answer + format + group internal-knowledge | 8 |
| ReTool | PPO | **0** | — | **Yes** (2 ep) | Yes (`<interpreter>`) | ±1 outcome | — |
| ToRL | GRPO | **omit** | — | **No** | Yes (OBSERVATION) | ±1 outcome | C=1 |
| Tool-Star | GRPO + Self-Critic DPO | — | — | **Yes** (3 ep) | Yes | hierarchical + multi-tool 0.1 | — |
| ARTIST | GRPO | (omit) | — | **No** | Yes | answer + format + tool-success | — |
| rStar2-Agent | **GRPO-RoC** | **去掉** | **0.2 / 0.28** | **Yes** (non-reasoning) | (transition 抽取替代) | **0/1 only** | **10 → 15** |
| Agent Lightning | framework | 视底层 | 视底层 | 任意 | **不需要** | identical-assignment + AIR | 不限 |

### Deep Research / Web Agent RL

| 工作 | RL 算法 | Tool token | Rollout 组织 | Context 策略 | 关键 stabilizer |
|---|---|---|---|---|---|
| Kimi-Researcher | **REINFORCE + γ-decay** | 不参与 loss | **Turn-level partial rollout** + 异步 | 自学习 context management | 丢负样本 + on-policy 强制 |
| Kimi K2 | **K1.5 闭式解 L2 trust region** | partial rollout 跨 iter | 同 K1.5 | — | Token budget + PTX loss + temp decay + critic 资格门控 |
| ASearcher | **GRPO** + AReaL 异步 | append-only | **全异步、staleness η** | 仅最近 25k chars | dynamic filtering 剔除 zero advantage |
| WebDancer | **DAPO** | observation mask | 每步 16 rollouts | 全历史拼接 | dynamic sampling + 10-gram 阈 4 + LLM-as-Judge |
| WebSailor V1 | **DUPO**（自研） | observation mask | duplicate non-zero variance 样本 | RFT trajectory < 32k | DUPO 自身的 sample 复制 |
| WebSailor V2 | DUPO + Dual-env | 同上 | simulator + real-world 双环境 | 同上 | 仿真器降本 + 真实环境收敛 |
| Tongyi DeepResearch | **customized GRPO** (on-policy + LOO + neg filtering) | — | 异步 rollout server | CPT 32K→128K | binary reward + dynamic data curation |

### SWE Agentic RL

| 工作 | 算法 | Base | Trajectory | Context | Max turns | Rollouts | LR | 算力 | 环境数 |
|---|---|---|---|---|---|---|---|---|---|
| DeepSWE | GRPO++ | Qwen3-32B | multi-turn | 64k (eval) | 100 | 16 (TTS) | — | 64× H100 × 6d | 512 docker |
| SWE-RL | GRPO | Llama-3.3-70B | **single-turn** | 16k | 1 | 16 | — | 512× H100 × 32h | 无（diff reward） |
| SWE-Gym | RFT (filtered BC) | Qwen2.5-Coder-32B | multi-turn | 32k | — | — | 1e-4 | 2-8× H100 | 2,438 |
| R2E-Gym | RFT | 32B | multi-turn | — | — | — | — | — | 8,700+ |
| Nebius LCMT | DAPO 改 | Qwen2.5-72B | multi-turn | **65k → 131k** | **40 → 80** | 10 | **1e-6** | 128× H200 | 7,249 |
| Qwen3-Coder | 未披露 | 480B-A35B | multi-turn | 256k–1M | — | — | — | — | **20,000** |
| Cursor Composer | 未披露 async | (Kimi K2.5?) | multi-turn + self-summary | 32k 主训 | 数百 | — | — | 多 region 数千 GPU | "数十万" sandbox |

---

## §43 共识 → 出处反查表

> **观点**:15 条共识的完整论证在文档[「写在最前面」](#写在最前面工程实战-15-条共识)。本节是反向索引——读完一个工作后,可以反查"它支持/违反哪些共识",或者反过来"哪些工作可以拿来论证某条共识"。
>
> 列名说明:**支持** = 工作中明确采用此做法;**反例** = 工作走相反路线但能 work,反映共识的边界。

| 共识 | 支持工作(配置) | 反例 / 边界 |
|---|---|---|
| **1. Tool/observation token mask loss & IS** | Search-R1 (§11, `<information>` mask)、R1-Searcher (§12, `<begin_of_documents>` mask)、ReTool (§13, `<interpreter>` mask)、ToRL (§14, OBSERVATION mask)、WebDancer (§21)、WebSailor (§22)、Nebius (§29)、ARTIST (§16) | Agent Lightning (§36) 用 transition 抽取等价替代——不存就不需要 mask |
| **2. 移除 KL 项** | DAPO (§2, β=0)、Magistral (§3)、MiMo (§4)、Skywork-OR1 (§5)、ReTool (§13, β=0.0)、ToRL (§14, omit)、rStar2 (§17)、DeepSWE (§25) | DeepSeek-R1 (§6, β=0.001)、Search-R1 (§11, β=0.001)、R1-Searcher Llama (§12, β=1e-4)、Llama 4 |
| **3. Clip-Higher 0.2/0.28** | DAPO (§2, 0.2/0.28 首倡)、rStar2 (§17, 0.2/0.28)、Nebius (§29, 0.2/0.3→0.2/0.26)、Magistral (§3, ε_high 动态)、MiniRL (§1, 0.2/0.27)、Skywork-OR1 (§5)、MiMo (§4) | DeepSeek-R1 (§6, ε=10 几乎不裁)、Search-R1 (§11, 对称 0.2) |
| **4. Outcome-only reward** | ReTool (§13, ±1)、ToRL (§14, ±1 + Code-Exec reward 无提升)、rStar2 (§17, 0/1 + 论文反对 shaping)、Search-R1 (§11, 纯 EM)、Search-R1 Empirical (intermediate retrieval reward 有害)、DeepSWE (§25, Pass/Fail 二值)、Kimi-Researcher (§18)、Tongyi DR (§24) | Tool-Star (§15, multi-tool $`r_M=0.1`$)、Magistral (§3, 四维)、R1-Searcher (§12, 两阶段)、Kimi-Researcher (§18, γ-decay 隐式) |
| **5. 反 length norm / overlong filtering** | MiniRL (§1.7)、DAPO (§2, token-level Σ)、Skywork-OR1 (§5)、Magistral (§3, group total length 折中)、rStar2 (§17 失败案例 1, "overlong filtering 反而让 overlong 比例上升")、Nebius (§29, soft length penalty) | DeepSWE (§25, Compact Filtering)——但只对"打不完"的样本 mask,不是普遍超长 |
| **6. Dynamic sampling / zero-adv filtering** | DAPO (§2 首倡)、Magistral (§3)、Skywork-OR1 (§5)、ASearcher (§20)、WebSailor DUPO (§22, **复制路线,比 DAPO 快 2-3×**)、Skywork 课程式丢弃、rStar2 Stage 3 (§17) | 无明确反例;唯一不做的工作通常是 rollout 极便宜(SWE-RL §26 single-turn) |
| **7. Actor lr ≈ 1e-6** | DAPO (§2, 1e-6+warmup 20)、ReTool (§13)、Search-R1 (§11)、rStar2 (§17)、Nebius (§29)、MiMo (§4) | DeepSeek-R1 (§6, 3e-6)、R1-Searcher (§12, 2e-6) |
| **8. SFT 三派** | **派 A 无 SFT**:ToRL (§14)、Search-R1 (§11)、ARTIST (§16)、DeepSWE (§25)、R1-Searcher v1 (§12) ;**派 B 只教 format**:rStar2 (§17, SFT 后 AIME 仅 3.3%) ;**派 C 含 long-CoT**:ReTool (§13)、WebDancer (§21)、Tool-Star (§15)、Nebius (§29)、R1-Searcher++ (§12) | — |
| **9. 多阶段长度课程** | Skywork (§5, 8K→16K→32K)、Magistral (§3, 16k→24k→32k)、MiMo (§4, 32K→48K)、rStar2 (§17, 8K→12K→12K)、Nebius (§29, 65K→131K)、Tongyi DR (§24, CPT 32K→128K) | DeepSeek-R1 (§6, 32K 单阶段)、GLM-4.5 reasoning RL (§8, 单阶段 64K **优于** 渐进式 — 反例) |
| **10. Partial rollout / 异步** | Kimi-Researcher (§18, turn-level partial)、AReaL/ASearcher (§35/§20, staleness η)、WebSailor V2 (§22, dual-env)、Cursor Composer 2 (§32, mid-rollout sync)、Magistral (§3, NCCL broadcast 不丢 in-flight)、Seed-Thinking (§9, streaming) | 短 trajectory reasoning RL 不需要(DAPO/Skywork);判断阈值见共识 10 |
| **11. 环境基础设施差异化** | DeepSWE 512 docker (§25)、SWE-Gym 2,438 (§27)、R2E-Gym 8,700+ SWE-GEN (§28)、Nebius 7,249 (§29)、rStar2 65K 并发 tool call (§17)、Qwen3-Coder 20,000 envs (§30)、Cursor 数十万 Firecracker VM (§32) | SWE-RL (§26) 不要环境也能 work(diff reward)——但单轮天花板低 |
| **12. Entropy collapse 治理** | Clip-Higher(共识 3 全员)、Skywork-OR1 adaptive entropy tgt=0.2 (§5)、DeepSWE 去 entropy loss (§25)、rStar2 去 KL+entropy loss (§17)、Kimi-Researcher 丢负样本 (§18)、Skywork 1 grad step/rollout (§5) | rStar2 Stage 1 (§17) 故意让 clip ratio>10%、Skywork τ=1.0 反而比 τ=0.6 稳——反直觉证据 |
| **13. 判分两域分治(RLVR vs GRM)** | 可验证 RLVR:共识 4 全员;不可验证 GRM:Kimi K2 闭环 critic + 资格门控 (§19)、SPCT/DeepSeek-GRM online RL (2504.02495)、Seed-Thinking 双轨 reward (§9)、DeepSeek-V3 self-feedback+voting (§6);完整三路 + 决策树见 §47 | DeepSeek-V4 合并 actor–GRM(丢独立性换无分布失配,前沿高风险);KL 与共识 2 反向(GRM 域加回 KL 兜底) |
| **14. IS 粒度之争(token/序列/裁权重)** | 序列级:GSPO(§10A,Qwen3,免 routing replay);裁权重:CISPO(§10B,MiniMax-M1)、ScaleRL(§10F,Meta 选 CISPO);几何平均:GMPO(§10C);token+截断:TIS(§44);MoE 对齐另解:DeepSeek-V3.2 Keep Routing(§33D) | **Lite-PPO(§10E):clip-higher 依赖 base/aligned + 规模,非普适**;INTELLECT-3(§41E):**async-8 下 GSPO 崩**;GSPO 机制归因有争议(2509.24203);TIS 本身有争议(§41B slime 3B collapse) |
| **15. 多轮 agentic 崩溃模式** | void turn→梯度爆炸:SimpleTIR(§17A);echo trap:RAGEN/StarPO(§17B);entropy 失控:EPO(§17C);template collapse:RAGEN-2(§17C);两级/树形信用:GiGPO/Tree-GRPO(§17D);熵驱动分支:ARPO(§17E) | **stabilizer 反噬**:RAGEN-2——reward 方差≈0 时加正则反加速 collapse;EPO 非处处正收益(ALFWorld 某指标 −10.9%);单纯过滤低概率/高 ratio token 救不了 void turn(§17A) |

**怎么用这张表**:
- **复现某个工作时**:对照它在每条共识上的位置,确认你的 recipe 没漏关键 trick
- **设计新 recipe 时**:每条共识看"反例栏"——反例越多说明这条不是铁律,可以尝试偏离
- **debug 训练崩了**:从共识 12 治法清单倒查,优先排除"没做"的项

---

## §44 训推一致性(隐形杀手)

### 核心观点

> **同一份权重 $`\theta_{\text{old}}`$,在 training engine(Megatron/FSDP BF16)和 inference engine(vLLM/SGLang FP8)上算出来的 logprob 不严格相等**——这不是 bug,是默认事实。
>
> 不修正这个不一致,几步训练之内 entropy 必崩。前面共识 12 + 监控类 2 给的是"如何监控",本节给"为什么会有 + 如何修"。

### 不一致的五个物理来源(逐源分析)

**来源 1 — 数值精度(FP8 vs BF16)**
- **现象**:推理端用 FP8/INT8 量化加速,训练端用 BF16 mixed precision;同一 logit 算出的 softmax 概率有 round-off 误差,IS ratio 偏离 1
- **机制**:量化把高概率舍入更高、把极低概率压成 0(详见监控类 1 + 共识 12 的 entropy collapse 三股力分析)
- **对策**:**Truncated Importance Sampling (TIS)**——截断阈值 ≈ 5(MiniRL §1.4,Yao et al. 2025);部分团队(Skywork、GLM)直接 **FP8 训推统一**;Thinking Machines 2025 走 **batch-invariant deterministic kernel** 路线(Tinker §40)

**来源 2 — Kernel 实现差异(浮点加法不结合)**
- **现象**:PagedAttention(vLLM)的 reduction order 与 training attention 不同,**同一 token 在不同 batch 位置可能拿到不同 logit**(continuous batching 的非确定性 reduction)
- **机制**:浮点加法不满足结合律, $(a+b)+c \neq a+(b+c)$,ulp 级漂移随 sequence 长度累积
- **对策**:开启 **batch-invariant kernel**(Thinking Machines 路线)、或在 vLLM 端关闭 continuous batching 做对照实验;实际工程多数选**接受漂移 + 用 TIS 兜底**

**来源 3 — BPE retokenization drift(最隐蔽)**
- **现象**:agent 通过 OpenAI 兼容 API 拿 text,训练端 retokenize → 出来的 token id 与 inference 时不一致(BPE 边界 / 特殊 token 处理差异)→ 训推完全错位 → PPO ratio 爆炸
- **证据**:Agent Lightning(§36)2025-10-22 vLLM blog "**No More Retokenization Drift**" 专门讨论这个;所有人都中过招
- **对策**:**API 直接返回 token id**(vLLM/SGLang 现在都支持),训练端不再做 retokenize;**任何接 vLLM/SGLang 的 agent RL pipeline 都应做一次 token-id 对齐验证**

**来源 4 — KV cache 跨权重版本残留**
- **现象**:rollout 中途权重更新后,前缀 KV cache 还是旧权重算的;后期 token 实际是"半新半旧"策略生成,但被当作 fully on-policy 处理
- **对策(两路)**:
  - **Magistral (§3)**:NCCL broadcast 推权重时**不丢 in-flight 序列**;KV cache 轻度过期靠 off-policy IS 校正补救
  - **Cursor Composer 2 (§32)**:**mid-rollout weight sync**——inference worker 在 rollout 中途同步权重,后期 token 主动更 on-policy

**来源 5 — MoE expert 路由不一致**
- **现象**:同一 token 在训推两端被路由到不同 expert,**等价于换了模型**;ratio 完全无意义
- **机制**:MoE router 对输入分布敏感,inference engine 的 batch 组织、激活精度都会改变路由结果
- **对策**:**Routing Replay**(MiniRL §1.3)
  - **R2(Vanilla)**:训练时强制用训练引擎旧策略的路由 $`e^{\pi}_{\text{old}}`$ ——只缓解 policy staleness
  - **R3(Rollout)**:训练时直接 replay 推理引擎 rollout 时的路由 $`e^{\mu_{\text{old}}}`$ ——同时缓解 train-infer 和 staleness 两个 gap
  - **决策表**:N=gbs/mbs ≤ 2(几乎 on-policy)用 R2;N ≥ 4(显著 off-policy)用 R3——MiniRL §1.6 实验给出的数值

**来源 6(额外)— mini-batch 内 policy staleness**
- **现象**:大 batch 拆 N 个 mini-batch 做 N 次梯度更新, $`\pi_\theta`$ 已经从 $`\mu_{\theta_{\text{old}}}`$ 漂走,但仍按 on-policy 算
- **对策**:**token-level IS**(共识 2 移除 KL 后的标配)+ **Clip-Higher**(共识 3)+ N≥4 时加 R3

### 实战 checklist(优先级排序)

1. **零优先级(开训前)**:接 vLLM/SGLang 的 agent RL pipeline,**先做 token-id 对齐验证**——这一项不通过,后面都白搭
2. **训练第一天**:打开三件套监控(详见监控类 1+2):**entropy + clip ratio (high/low 分开) + IS ratio 分布直方图**
3. **健康阈值**:IS ratio 集中在 **[0.8, 1.27]**;clip rate < 15%;MoE expert routing diff > 95% 重合
4. **告警三件套**:entropy 骤降 + clip rate > 15% + KL(μ_old‖π_old) 同步爆涨——**任一异常先停训查 logprob mismatch**,不要硬抗
5. **MoE 模型额外**:必须监控 expert routing diff,< 80% 重合直接上 R3

### 排查决策树

```
训练崩了
├─ IS ratio mean ≈ 1 且分布窄 → 不是数值差,查 reward / 学习率
├─ IS ratio mean 系统性 ≠ 1 → 数值差或 retokenization
│   ├─ 先 token-id 对齐验证(零成本)
│   ├─ 仍偏 → 上 TIS truncation = 5(MiniRL 阈值)
│   └─ 仍偏 → FP8 训推统一或换 batch-invariant kernel
├─ IS ratio p99 > 5 → 必上 TIS;查是否 KV cache 跨权重残留
├─ MoE expert routing diff < 80% → 上 R3(Rollout Routing Replay)/ 或换 GSPO 免 replay
└─ entropy 崩 + clip rate 飙 + KL 同步爆 → 训推 mismatch 确诊,优先排第 3、5 来源
```

### 【2025–2026 深化】"你的高效 RL 框架偷偷在做 off-policy"

> 这一节把 §44 从"经验"升级到一个被正式命名的问题,并补上 **TIS 的争议**(它不是银弹)、**FP32 LM head 的双重佐证**、**双侧 clip**、以及**新算法(GSPO/CISPO)作为另一类解法**。

**① 问题被正式 framing("secretly off-policy")**:论文 *"Your Efficient RL Framework Secretly Brings You Off-Policy RL Training"*(Feng Yao 等,NeurIPS 2025;[notion](https://fengyao.notion.site/off-policy-rl) / [verl PR #2953](https://github.com/volcengine/verl/pull/2953))把它讲透:训练后端(FSDP/Megatron)与推理后端(vLLM/SGLang)**即便参数 θ 完全相同**,对同一 token 也给出显著不同概率,极端时一个给 1、一个给 0。**名义 on-policy 训练实为有非平凡偏置的 off-policy**。作者**排除了"纯 vLLM bug"假说**(patch vLLM 暴露真实概率 + lm_head 转 FP32 后失配仍在)。修复 = **TIS(Truncated Importance Sampling)**:对 $`\rho=\pi_{\text{train}}/\pi_{\text{infer}}`$ 做**单边截断** $\min(\rho,C)$——注意是**单边**(下方自由、只截上方),区别于 PPO/CISPO 的双边;token 级。

**② ⚠️ TIS 不是银弹(关键争议,务必写明)**:
- **正面**:Qwen2.5-32B + DAPO 上 TIS "明显提升";**量化 rollout 场景 TIS 大幅弥合 gap**,使 FP8 rollout 可用。
- **反面**:多个后续报告 **TIS 持续降低稳定性、变体过早崩溃**。机制假设:**低概率 token 天然产生高方差 ratio、最易被 TIS 截断,这虽降方差却系统性下调这些 token 的梯度 → 引入有害 bias**(与 PPO 比率裁剪同源)。**slime 实锤(§41B)**:Search-R1 用 3B 模型时开 TIS 反而**早期 policy collapse**(多语种乱码),关掉才收敛。
- **折中:MIS(Masked IS,越界比率置零/丢弃)+ IcePop(token 级 MIS,用于 MoE)**。slime blog 因此**推荐 MIS 作默认**。**C 值框架相关**:Open Instruct C=2、verl 示例 C=10、MiniRL 阈值 5——没有统一值,要按框架/任务调。

**③ ⭐ FP32 LM head——两个独立硬证据**:
- **MiniMax-M1(§10B)**:精度失配主要来自 **LM head 的 high-magnitude activations**(不在 attention),**FP32 head 使训练/推理概率相关性 0.9→0.99**,恢复 reward 增长;且**只在 hybrid 架构出现,小 dense+softmax 不出现**。
- **Meta ScaleRL(§10F)**:**FP32 logits 把性能渐近线 0.52→0.61**(改天花板而非只改效率)。
- **结论**:`lm_head`/logits 用 FP32 几乎是免费的 train-infer 对齐手段,应作默认。NeMo-RL(§41C)走相反但同源的路——**端到端全 FP8 对齐**(让训推用同一套低精度),同样降 logprob gap。

**④ 双侧 clip:高 staleness 异步的必备**:标准 GRPO 的 `min` 对**负 advantage 不裁剪**,高 off-policy 时大更新致崩。INTELLECT-2(§41E)给负 advantage 的 ratio 加上界 δ,**实测 ε=0.2, δ=4**(MiniMax-01 独立一致)。这是异步/去中心化 RL 的稳定关键。

**⑤ 新算法作为"换目标函数"的解法**:除了"修 IS 比率",还可**换掉比率的粒度**——**GSPO(序列级,§10A)免 MoE routing replay**、**CISPO(裁权重不丢 token,§10B)**、**DeepSeek-V3.2 Keep Routing(§33D)强制训推路由一致**。但注意 **INTELLECT-3 实测 async-8 下 GSPO 崩**——序列级 IS 在极端 staleness 下反不如双侧 clip。

**⑥ 一个常被忽略的共识**:**bf16 rollout 不出现 entropy collapse,量化 rollout 才崩**——即很多"熵崩"是训推数值差的下游症状(§10L)。**先查 §44,再调熵超参**,别一上来就动 entropy coefficient。

---

## §45 算力规模参考

> **观点**:RL 算力随"模型规模 × multi-turn × context 长度"非线性放大。本表给出从 7B 到 1T 的参考点;选型时先对照"经验法则"再看具体工作。

### 各规模实测算力(按"模型规模 → 卡数 → 时长"对照)

| 任务规模 | 模型 | 算力 | 时长 | 代表工作 |
|---|---|---|---|---|
| 7B math/code RL | 7B dense | 8× A800 | — | ToRL (§14) |
| 7B math reasoning | 7B dense | 32× H800 | 2700 steps | Skywork-OR1-7B (§5) |
| 32B reasoning RL | 32B dense | — | 400 steps | ReTool (§13) |
| 32B SWE multi-turn RL | 32B dense | **64× H100** | **6 天** | DeepSWE (§25) |
| 70B single-turn SWE-RL | 70B dense | **512× H100** | **32 小时** | SWE-RL (§26) |
| 72B multi-turn long-context SWE | 72B dense | **128× H200** | — | Nebius LCMT (§29) |
| 14B agentic math (multi-turn tool) | 14B dense | **64× MI300X** | **1 周** | rStar2-Agent (§17) |
| 30B-A3B MoE agentic | 30B-A3B MoE | **256× H20 (96GB)** | — | WebDancer (§21) |
| 32B QwQ deep search | 32B reasoning | ~7.6K H100·hours | — | ASearcher (§20) |
| 200B-20B MoE thinking | 200B-A20B | "Tensor+Expert+Sequence 并行" + Streaming | — | Seed-Thinking-v1.5 (§9) |
| 1T-32B MoE agentic | 1T-A32B MoE | 未披露 | — | Kimi K2 (§19) |
| 工业级 SWE agent | — | 多 region 千卡 + 数十万 sandbox | 持续部署 | Cursor Composer 2.5 (§32)、Qwen3-Coder (§30)、Anthropic (§33) |
| **1.5B 小模型数学(极致性价比)** | 1.5B dense | **8→32× A100** | **3,800 A100·h ≈ \$4,500** | DeepScaleR (§10J) |
| **14B/32B 数学(全公开数据)** | 14B/32B | **12× H800** | **~6h ≈ \$1,000** | Light-R1 (§10J) |
| **456B-A46B MoE reasoning** | 456B-A46B MoE | **512× H800** | **3 周 ≈ \$0.53M**(厂商估) | MiniMax-M1 (§10B) |
| **8B/17B MoE RL scaling 研究** | 8B dense / Scout MoE | **旗舰单次 10 万 GPU·h**(总 >40 万) | — | ScaleRL (§10F) |
| **32B 去中心化异步 RL** | 32B dense | **全球分布式志愿节点** | **2 周**,train:infer FLOPs≈1:4.5 | INTELLECT-2 (§41E) |
| **106B-A12B MoE 异步 RL** | 106B-A12B MoE | **512× H200 / 64 节点** | **~2 个月** | INTELLECT-3 (§41E) |
| **32B SWE SFT(数据 scaling)** | 32B dense | **8× H800** | **12h / 3 epoch** | Skywork-SWE (§33B) |
| **7B code RL(从 base)** | 7B dense | — | **80 步 ≈ 48 H100·h** | AceCoder (§33C) |

### 经验法则(五条)

- **模型规模 → 卡数对照**:7B → 32 卡,32B → 64 卡,70B → 128–512 卡
- **multi-turn 比单轮多 2–4× 算力**(rollout 慢 + 异步收益受限)
- **async / partial rollout 可省 30–50% GPU 闲置**(共识 10)
- **小模型数学 RL 极便宜**:1.5B–14B 数学推理 RL 只要 **\$1k–\$5k**(DeepScaleR/Light-R1),是验证 recipe 的最佳起点
- **agentic / 长 ctx / 去中心化会把成本拉高一个量级**:同规模下 SWE multi-turn、deep research、分布式异步都显著更贵;**RL 总预算约 = 预训练的 10%+**(DeepSeek-V3.2 §33D)

---

## §46 抄作业指南

> **观点**:学术界 RL 论文公开度极不均;别一上来就读综述,先按"我想抄什么"对应到下面的清单,直接看那篇论文 + repo。

### 按"抄什么"分类的推荐清单

**1. 抄完整训练 recipe(按可复现度排序)**
1. **Nebius LCMT** ([2508.03501](https://arxiv.org/abs/2508.03501)) — 长 context multi-turn 完整超参表,公开度最高
2. **SWE-RL** ([2502.18449](https://arxiv.org/abs/2502.18449)) — 70B single-turn 完整超参
3. **rStar2-Agent** ([2508.20722](https://arxiv.org/abs/2508.20722)) — agentic math 完整 recipe + **§4.3.1 踩坑日志**(论文里最值钱的部分)
4. **DAPO** ([2503.14476](https://arxiv.org/abs/2503.14476)) — reasoning RL 全套 trick + verl recipe
5. **ReTool** ([2504.11536](https://arxiv.org/abs/2504.11536)) — PPO + tool mask
6. **R1-Searcher** 系列 — multi-turn search 完整公开

**2. 抄环境构造**
- **R2E-Gym** (§28) — SWE-GEN 自动 commit→test,**让 SWE 环境从手工 2K 涨到自动 8.7K**
- **Qwen3-Coder** (§30) — 20K 并发 envs 工业路线(无公开细节但是规模标杆)
- **Cursor Anyrun** (§32) — Firecracker + filesystem snapshot,**500+ pods/sec** 调度

**3. 抄算法稳定化**
- **DeepSWE GRPO++** (§25) — 六件套(Clip-High + No KL + Length Norm + LOO + Compact Filtering + No Entropy Loss)
- **rStar2 GRPO-RoC** (§17) — 解决"positive trajectory 含 10–15% tool error"问题的 resample-on-correct
- **MiniRL token-level IS + TIS** (§1) — 训推一致性的理论 + 实操统一框架

**4. 抄工业部署**
- **Cursor Composer 2** (§32) — 解耦四服务(Training/Environments/Inference/Evaluations)+ mid-rollout weight sync + environment fidelity 信条
- **GLM-4.5 slime** (§8) — Hybrid 同步/异步 + FP8 rollout(8×H100 部署)+ Megatron+SGLang
- **rStar2 基础设施** (§17) — Load-Balanced Rollout Scheduler 实现 65K 并发 tool call、0.3s 端到端延迟

### 待挖掘(目前没公开的空白点,值得持续关注)

- **超参未公开**:Kimi-Researcher / Kimi K2 / WebDancer / WebSailor / Qwen3-Coder 具体 lr / batch / clip / KL 全无
- **二阶段 RL 黑箱**:DeepSeek-R1 第二阶段 RL 的奖励权重、reward model 细节
- **闭源工业 recipe**:Anthropic Claude 系列几乎无任何 RL 训练细节;Devstral 无公开 RL 细节
- **MoE × agentic RL**:训推路由一致性的实战数据极少,MiniRL §1.6 是目前公开度最高的

### 直接看 config 比看论文快的工作

需要具体超参时直接 clone 对应 GitHub repo 看 config:
- `ASearcher/configs/` (§20)
- `verl/recipe/` 下的 `dapo/`、`rstar2/`(§34)
- `rllm/recipes/deepswe/` (§39)
- `Tool-Star/scripts/train/` (§15)
- `tinker-cookbook/` 下的 RL 实现(§40)

---

## §47 奖励模型路线:offline GRM vs online GRM vs actor–GRM 合并(DeepSeek-V4)

> **观点**:[共识 4](#共识-4--outcome-only-reward-是主流且足够强)说"outcome-only / RLVR 是主流且足够强"——**那是 verifiable 域(math / code / SWE-test)的结论**。一旦进入 **hard-to-verify 域**(开放写作、研究综合、设计、主观质量、long-horizon agent 行为),规则 verifier 根本写不出来,就只能上**生成式奖励模型(GRM)**:让一个模型读 response、出 critique、给分。
>
> 这一节回答 GRM 这条线的工程选型。注意它和共识 4 不矛盾,而是它的**补集**:**能 verify 的别碰 GRM,碰 GRM 的都是 verify 不了的任务**。本文前面零散提到的 reward model 设计(Seed-Thinking 双轨 reward §9、Kimi K2 self-critique rubric + 闭环 critic §19、DeepSeek-R1 第二阶段 RM 黑箱 §6)在这里第一次被拉到一起做横向选型。

### 两个正交维度

GRM 选型可以拆成两个**互相独立**的轴——别把它们混成一个"offline vs online"的二元问题:

| 维度 | 取值 | 含义 | 代表 |
|---|---|---|---|
| **A. judge 冻不冻** | **offline**(冻结)/ **online**(RL 中持续刷新) | RM 在 RL 全程是否更新 | offline: 经典 RLHF RM;online: SPCT (§下)、Kimi K2 闭环 critic (§19) |
| **B. judge 与 actor 是否同参数** | **分离** / **合并** | 判分的权重是不是 actor 自己 | 分离: 绝大多数;合并: **DeepSeek-V4** |

下面三条主流路线就是这两轴的组合:**offline 分离 GRM** → **online 分离 GRM** → **合并 actor–GRM**(合并几乎必然是 online,因为 actor 一直在动)。

---

### 路线 1 — Offline GRM(冻结的分离判分模型)

训练一次、RL 全程不更新。问题全来自**它静止而 policy 在动**:

- **机制 / 直觉**:RM 只是真实偏好的**代理(proxy)**。policy 在 RL 中主动往 RM 高分方向漂,迟早漂到 RM 训练分布之外,把 RM 的系统性误差当漏洞钻。表现为 **proxy reward 单调涨、真实质量先升后崩**(Gao et al. 2023 "Scaling Laws for Reward Model Overoptimization" 的标志性曲线)。
- **四个具体病症**:
  1. **Reward hacking / over-optimization(Goodhart)**:典型 hack 形态是 **length bias**(RM 把"长"误当"好")和 **sycophancy**(看起来对 > 真的对)。
  2. **OOD 失校准 + underspecification**:同数据训出的两个 RM 在分布内一致,**一旦 policy 漂出训练分布,二者行为可以天差地别**——in-distribution 高准确率不保证 OOD 鲁棒。而 RM 是 RL 唯一优化目标,这种 miscalibration 会被**系统性放大**。
  3. **能力天花板**:policy 上限被 RM 判别力锁死;actor 在某维度强过 RM 时,RM 反而成噪声源。反直觉的是:**RM 越大、训练数据越多,over-optimization 有时反而更严重**(Gao et al.)。
  4. **Staleness**:policy 涌现的新模式 / 新失败模式,冻结 RM 既奖不对也罚不到。
- **检测手段**:**Gold-RM divergence**——用一个更强的参考 RM 复评同一批 policy 输出,proxy 分持续涨而 gold 分开始降 = reward hacking 确诊(对应监控类 7 的"reward 涨 benchmark 不涨")。
- **生成式特有开销**:GRM 是生成式的(出 critique 再给分),offline **也省不掉推理成本**——每个新 rollout 还要跑一遍长 critique。offline 只省了"训 RM"那部分,没省判分算力。

**证据 / 参考**:Gao et al. 2023 [2210.10760](https://arxiv.org/abs/2210.10760);RM underspecification 与 over-optimization 综述见 [reinforced.info](https://www.reinforced.info/p/reward-model-overoptimization);offline framing 的两源 distribution shift + 悲观正则(RPO)见 [2405.16436](https://arxiv.org/abs/2405.16436)。

---

### 路线 2 — Online GRM(judge 随 RL 共同刷新,仍与 actor 分离)

RM 在 RL 中持续用当前 policy 分布的新数据更新。**缓解了 OOD,代价是两个会动的东西耦合**:

- **机制**:judge 一直被拉回 actor 当前分布,OOD 漂移被结构性抑制;但 policy 和 reward 同时变 → **非平稳优化**,容易震荡甚至 collapse,且要持续拿到当前分布的新标签(贵)。
- **三个问题**:① moving target,要小心两者更新节奏;② 标注成本 + 双大模型同步训练的工程复杂度;③ **协同崩塌**——judge 若用 policy 自己的输出更新,可能合谋到退化解(sycophancy / mode collapse),误差自我强化。
- **关键防 hack 机制(都来自把 judge 锚回"可验证能力")**:
  - **闭环 critic refinement**(Kimi K2 §19):**on-policy RLVR rollout 持续 update critic**,把**可验证信号蒸馏进主观判断**——让 judge 的主观打分始终被客观对错牵着。
  - **Critic 资格门控**(Kimi K2 §19):**critic 必须先在可验证任务上证明能力,才允许评判开放输出**——直接防 reward hacking 的硬约束。
  - **SPCT / DeepSeek-GRM**([2504.02495](https://arxiv.org/abs/2504.02495)):pointwise GRM + **online RL** 训练,核心是 **principle-as-part-of-reward**——judge **临场按任务生成评判原则(principle)再出 critique**,而非固定 rubric;配 parallel sampling + **meta-RM 投票**做 inference-time scaling。27B GRM 推理时 scale 后**反超 340B scalar RM 和 GPT-4o**,且跨域 bias 显著小于 scalar RM。这套 principle/critique 机制正是合并路线(路线 3)要复用的护栏。
  - **DeepSeek-V3 先例**([2412.19437](https://arxiv.org/abs/2412.19437)):用 **V3 自身 + voting** 对开放问题做 self-feedback,增强对齐鲁棒性——已是"模型即 judge"的雏形。

---

### 路线 3 — 合并 actor–GRM(DeepSeek-V4 路线)

把判分模型彻底取消,**judge 就是 actor 自己**(同一份权重,既生成又评判),可理解为 online 的极端形态。

- **DeepSeek-V4 怎么做的**:对 hard-to-verify 任务,**丢掉传统 scalar RM,改用 GRM,且让 actor 模型本身充当 judge,RL 同时优化"生成"和"评判"两种行为**。它不是孤立 trick,而是嵌在 V4 的"先训 domain specialist → 再合并"流程里:每个域 SFT + GRPO 训专家,最后**不是权重平均,而是 On-Policy Distillation(OPD)**——student 采自己的 rollout,10+ teacher 在这些轨迹上给目标分布,避免"一个复合 reward 一次产出所有行为"。
- **合并解决了什么**:
  - **结构上消除 actor–RM 分布失配**:GRM 永远恰好在 actor 当前分布上,offline 的 OOD 和 online 的滞后从根上没了。
  - **省一份大模型**:显存 / 算力下降,不用维护两套权重同步。
  - **能力互促**:学 critique 反哺推理(self-reflection 信号回流生成),理论上 generator 和 verifier 一起涨,绕开"RM 弱于 policy"的天花板。
- **合并引入的新风险(本质是丢了独立性)**:
  1. **误差相关 / 盲点共享**:同参数生成又判分,policy 推不对的地方它也判不出——独立 verifier 之所以有用,正因其错误与 generator **不相关**;合并把这层去相关抹掉了。
  2. **自我合谋式 reward hacking 更易**:梯度可直接朝"给自己打高分"走(sycophancy-to-self),比分离时更难防。
  3. **目标干扰 / 容量竞争**:policy-gradient 与 reward-modeling 两个目标共享参数,梯度可能打架。
  4. **稳定性更紧**:moving target 收紧到极致,对配方(更新步长、判分支路是否 detach)更敏感。
- **证据**:DeepSeek-V4 训练系统分析 [fireworks.ai](https://fireworks.ai/blog/what-deepseek-v4-says-about-training-platforms)、[kili-technology](https://kili-technology.com/blog/data-story-deepseek-v4)、[artgor review](https://artgor.medium.com/deepseek-v4-review-why-million-token-context-needs-efficient-attention-not-just-larger-windows-6dc8e74a00b1)(注:V4 暂无完整 arXiv 技术报告,以上为基于公开材料的二手分析,细节待官方 paper 确认)。

---

### 三路对照表

| | 核心病 | 根因 | 主要解药 | 代表 |
|---|---|---|---|---|
| **Offline GRM** | reward hacking + OOD 失校准 + 能力天花板 | RM 静止、policy 在动 | KL 兜底 + 周期 refresh + gold-RM divergence 监控 | 经典 RLHF RM |
| **Online GRM(分离)** | 非平稳震荡 + 标注/算力贵 + 协同崩塌 | 两个会动的模型耦合 | 资格门控 + 闭环 RLVR 蒸馏 + principle/critique | Kimi K2 §19、SPCT [2504.02495] |
| **合并 actor–GRM** | 误差相关丢独立性 + 自我合谋 hacking + 目标干扰 | judge 与 generator 同参数,去相关性消失 | 独立 critique 采样 + rubric/principle + verifiable anchor + 判分支路 detach | DeepSeek-V4 |

### 关键 tie-in:不可验证 reward → KL 罚项回归

[共识 2](#共识-2--几乎所有现代-reasoning-rl-都移除-kl-项)说"现代 reasoning RL 几乎都删 KL",但那条**明确留了边界:"reward 可信(verifier 干净)就能去 KL;reward 可疑(open-ended)才要 KL 兜底"**。GRM 这条线正是"reward 可疑"的全部场景——所以**主流 RLHF 对 GRM 几乎一律保留 KL 罚项**,它是限制 distribution shift、抑制 over-optimization 的头号手段(KL 不能根治 hacking,但能把 policy 摁在 RM 还可信的区域内)。一句话:**verifiable 域删 KL,GRM 域把 KL 加回来**——两条建议不矛盾,取决于 reward 可不可信。

### 推荐(决策树)

```
要不要用 GRM?
├─ 任务可验证(math/code/SWE-test/有 ground-truth)
│   └─→ 不用 GRM。RLVR / outcome-only,删 KL(共识 4 + 共识 2)。最稳最省,首选。
└─ 任务 hard-to-verify(写作/研究综合/主观/long-horizon agent)
    ├─ 预算有限 / 求稳起步
    │   └─→ Offline 分离 GRM,但【别纯冻结】:
    │        ① 保留 KL 罚项兜底(共识 2 边界)
    │        ② 周期性 refresh(用当前 policy rollout 重新标注再训 RM)
    │        ③ 盯 gold-RM divergence(监控类 7)防 hacking 失控
    ├─ 预算充足 / 要上限 / 团队能维护双模型
    │   └─→ Online 分离 GRM,【必带】资格门控 + 闭环 RLVR 蒸馏 + principle/rubric
    │        (Kimi K2 §19 + SPCT 2504.02495 的成熟组合,把 judge 锚在可验证能力上)
    └─ 前沿 / 追求最省一份模型 + 有机制工程能力
        └─→ 合并 actor–GRM(DeepSeek-V4),【必须补回独立性】:
             独立 critique 采样 + principle-based rubric + 混入 verifiable anchor 任务防自我合谋
             + 判分支路 detach。高风险高回报,不是新手默认项。
```

**一句话推荐**:**能 verify 就别碰 GRM**;**必须 GRM,就从"online 刷新 + 资格门控 + KL 兜底"起步**(路线 2,综合 K2 + SPCT,目前性价比与稳健性最好的成熟解);**合并 actor–GRM(路线 3)是方向性正确但高风险的前沿**——它最省、最无分布失配,却把"独立判分"这层最重要的安全垫拆了,只有当你有能力用 principle / 资格门控 / verifiable anchor 把那层去相关性人为补回来时才上。

> **🧠 思考点(作者本人,待验证)——合并是不是终局?**
>
> 直觉上合并 actor–GRM 很诱人:**verifier 与 generator 的分布失配是 offline/online 两条线所有病症的总根源,而合并从定义上消灭了它**,还顺手省一份大模型、让判分能力反哺推理。DeepSeek-V4 把它推上台面。
>
> 但我倾向认为**短期它不会是默认解,而是"强者的奢侈品"**,理由有二:
> - **独立性是被低估的安全资产**。RL 之所以怕 reward hacking,靠的就是"出题人和答题人不是同一个人"。合并后这层去相关被拆掉,等于把"防作弊"从结构保证降级成"机制补丁"(principle / 资格门控 / anchor)——补丁能不能稳住,强依赖团队的机制工程功力,DeepSeek 能做不代表中小团队能抄。
> - **它和共识 12"RL 上限 ≈ base 上限"耦合**:合并路线赌的是"判分能力能反哺生成、一起突破天花板",但若 base 判别力本就不足,self-judge 只会把自己的盲点正反馈放大。**base 越强,合并越安全;base 越弱,越该保留独立 judge**。
>
> 一个可证伪的预测:**未来一两年,verifiable 域继续 RLVR 无 GRM;不可验证域的主流仍是"分离 online GRM + 资格门控",合并路线只在头部实验室(数据/机制/算力三齐)落地**。判断它是否反转的信号:有没有中小团队用合并路线在公开 benchmark 上稳定复现、且不靠一堆补丁。

### 🧠 几个被低估的关键点(作者本人,待验证)

> 上面的决策树讲"怎么选";这里是几个我认为没被讲透、却直接决定成败的点,尤其针对合并 actor–GRM 这条前沿线。

**1. generator–verifier gap 是真正的资产,而它恰好在最需要 GRM 的地方最薄。**
GRM/verifier 这套能成立,底层是"**验证比生成容易**"(P vs NP 式直觉:判断一个证明对不对 < 自己写出证明)。**合并 actor–GRM 等于赌这个 gap 在同一个模型内部依然存在**。但 gap 大小是任务依赖的:math/code 上极大(跑测试就行),开放写作 / 审美 / 研究品味上趋近于 0("这文章好不好"和"写出好文章"一样难)。**于是有个残酷错配:合并在 gap 大的任务上最安全,却在 gap 小的任务上最危险——而 gap 小的任务正是你当初不得不上 GRM 的原因**。推论:合并最该先落地在**"半可验证"任务**(有部分客观结构:tool-use 的 end-state、带 rubric 的代码风格),纯主观任务上保留独立 judge 更稳。

**2. 合并改变了 reward hacking 的"形状",还顺手废掉了你自己的检测器。**
分离 RM 时,hacking 是 policy 去搜一个**固定对手**的盲点(搜索问题),gold-RM divergence 能逮(proxy 涨 / gold 跌)。合并后两边一起动,变成**协同收敛到一个"自洽但错误"的不动点**——更像 GAN mode collapse / 两个玩家串通。要命的是:**合并优化的目标本身就是"内部一致性",而 gold-RM divergence 依赖的恰恰是内部不一致信号 → 你最趁手的检测器失效了**。所以合并路线必须**外挂一个完全独立的 judge(不同 base / 不同数据)做周期体检**,否则会"训得很顺、指标很好、其实在自我欺骗"而毫无察觉。这点比"加 principle"更关键,却最容易漏。

**3. 这不是三选一,是一条连续谱;实战甜点大概率在"共享 trunk + 独立 head + stop-grad"。**
offline / online / 合并 是三个离散标签,真实设计是连续的:**冻结 → 周期 refresh → 共训分离 → 共享骨干分离头 → 全合并**。全合并(同一份权重、同一次 forward 既生成又判分)拿满了省参数 + 零分布失配,但也丢满了独立性。**我赌性价比最高的不是全合并,而是"共享 transformer trunk + 一个独立 critique/value head + 判分 head 对生成 head 做 stop-gradient"**:省掉大部分参数、judge 自动跟 actor 分布走(零失配),却保留一层结构性去相关(不共享最后那段表征 + 梯度不互灌)。DeepSeek-V4 的全合并是"激进端",这个是"务实端",值得对照实验——也呼应共识 12 里"IS 权重外挂 stop-gradient"的同一类思路(把不该回传的信号 detach 掉)。

**4. 自评的失败是不对称的:宁可让 judge 偏悲观。**
self-judge 里 **false-positive(把烂的判成好)远比 false-negative 致命**——正 reward 会被强化、复利放大,一个对自己太宽容的 judge 直接触发 runaway;漏奖一个好样本只是慢一点。这正好接上 offline-RL 的**悲观主义(pessimism)**理论(RPO / 不确定性惩罚,[2405.16436](https://arxiv.org/abs/2405.16436)):**合并 judge 应当被刻意调成"拿不准就低判",把不确定性罚进 reward**。便宜、有效,却几乎没人在合并语境里明说的护栏。

**5. "省一份模型"省的是显存不是算力,别被这点诱惑骗进高风险区。**
合并省的主要是 **RM 的权重(显存 / ops 复杂度)**;但**判分的推理 FLOPs 一点没省**——generative judge 每个 rollout 照样要跑一遍长 critique(路线 1 的"生成式特有开销"已提)。再叠加生成 / 判分两目标抢容量的梯度干扰,**合并的净算力收益其实不大**。所以合并的真正卖点是"零分布失配 + 能力互促",**不是省钱**。冲着省成本去合并的,动机和收益错配,大概率不值这个风险。

> **一句话收口**:这五点都指向同一个判断——**合并 actor–GRM 不是"把两个模型缝一起省事",而是用"独立判分"这个安全资产去换"零分布失配 + 能力互促"**。换得值不值,取决于你能不能用 半可验证任务(#1)+ 外部独立体检(#2)+ 结构性 stop-grad(#3)+ 悲观偏置(#4)把那层独立性补回来。补不回来,就老实留在路线 2。

---

## §48 实战教训精粹(practitioner lessons,带出处的非显然 know-how)

> 前面的章节是"按工作组织";这一节是"按教训组织"——从 practitioner writeups(HuggingFace Open-R1、Nathan Lambert/Interconnects、复旦 Secrets of RLHF、社区 trick 帖)里挑出的**非显然、带出处、可直接用**的经验。这些往往不在论文正文里,但最省时间。

### 来自 HuggingFace Open-R1(工程一线发现)

- **数据质量 vs 数量随预算翻转**:严格过滤的小数据**早期赢**,但**后期"更多样本更有益——即使含错误"**;推荐 Llama 验证 ∪ math_verify 的并集(~154k),别过度过滤。
- **GRPO 样本复用 μ 的甜区 = 2~4**:μ 太大损害学习(对照 prime-rl 的 DPPO 也默认小步重叠)。
- ⭐ **"代码可验证性危机"**:CodeForces 公开测试用例**截断在 ~500 字符**,公开数据集只含简单测试;**7 个通过 public test 的 R1 解,在 full test set 上全部失败**——**code RL 务必用 full/hidden test,别信 public test**(直接呼应 §33C)。
- **Packing 伤推理**:SFT 用 packing 后"模型几乎解不出题"(长 trace 跨 chunk 边界);长 CoT SFT 慎用 packing。
- **大 LR 在 code SFT 有用**:2e-5→4e-5 在 LiveCodeBench **+近 10 分**。
- **prefill `<think>`**:不 prefill 时 OOD 查询会退回 base instruct 行为;prefill 强制长 CoT。
- **32B 扩 context 易 OOM**:transformers/trl 无 context parallelism,>20k token 即 OOM(即便 16 节点)——长 ctx RL 必须用支持序列并行的框架。

### 来自 Nathan Lambert / Interconnects(高信号观点 + 反驳)

- ⭐ **"GRPO 不是特殊算法"**:源自 PPO,advantage 算法与 RLOO 几乎一致;现代 RL 算法真正差异只有 3 点——是否用 value function、是否 PPO-style clipping、每 prompt 几个答案。**"batching 和基础设施的决策往往比小的算法差异更重要"**——别过度纠结算法名,先把 infra 和数据做对。
- ⭐ **对 Dr.GRPO 的 caveat**(来自 Costa Huang):去掉 std 会**下调"9 错 1 对"里那个罕见但关键的正确样本的权重**;且 Dr.GRPO 虽输出更短,**未展示更好的下游最终性能**。Lambert 本人**更喜欢 DAPO 的方案**(把 length norm 移到 group 外,而非完全去掉)——这是共识 5 的一个重要细化。
- **KL 移除的 base vs instruct 区分**:base RL 应去 KL(模型要变化大);**对 instruct 模型做 RLVR,KL penalty 可能仍有用**(对照共识 2 的边界)。
- **数据是真正杠杆**:"真实性能更多来自数据";推崇**难度多样性优于课程**(batch 内混合难度,模型自己持续学习);排除选择题/判断题/证明题(易被猜)。
- ⭐ **怀疑论框架**:若一篇 paper 没"~2 倍超越 baseline",提升多半来自调参或混杂变量;并指出 **DAPO/Dr.GRPO 的 "DeepSeek-R1-Zero-Qwen-32B" 对比点实际并不存在**(误导性比较,读论文要警惕)。
- **"Aha moment" 祛魅**:更大的 Qwen 在任何 RL 前就已有 reflection 行为——"模型主要在放大,而非学新东西"(对照共识 12 + ProRL §10G 的争议)。

### 来自 Secrets of RLHF(复旦, [2307.04964](https://arxiv.org/abs/2307.04964) / [2401.06080](https://arxiv.org/abs/2401.06080))

- **policy constraint(KL 约束)是 PPO 有效的关键**,对稳定性至关重要(注:这是 RLHF 语境,与 reasoning RL 去 KL 不矛盾——见共识 2 边界)。
- ⭐ **更优监控指标**:用 **perplexity、response length、policy 与 SFT 的 KL** 监控,**比 reward 和 loss 值更能反映稳定性**(对照仪表盘类 2)。
- **无 SFT 初始化的 policy 在 PPO 中"明显无能"**;**用 RM 初始化 critic 是次优**(critic 要对每步给反馈,与给整 response 打分有 gap)。

### 社区收敛的 trick 清单 + 失败模式(Cameron Wolfe / clip 分解等)

- **clip-higher 是最有效单一 trick**:被 clip 的多是 low-prob 高熵 token("however/recheck"等);消融 step 2000 时去掉 clip-higher,熵从 0.34 掉到 **0.03(崩溃)**。但注意 Lite-PPO(§10E)的反例:它依赖 base/aligned + 规模。
- **clip-high 降熵、clip-low 升熵**(干净分解 [2509.26114](https://arxiv.org/abs/2509.26114)):光选 clip 参数就能控熵,**不必动 KL**。
- **health 信号:response 长度应持续上升**;异常超长是不健康(对照 Light-R1 §10J 的"长度与 reward 同步上升")。
- **reward hacking 在 self-training 尤甚**:多数旋钮(换 GRPO/RLOO、加大 KL、降 LR)**只延缓不阻止**崩溃——治本要从 reward 设计(共识 4 + §33C)入手。

### §48 一句话总纲

> **算法差异是二阶的,数据 + 基础设施 + reward 设计是一阶的**(Lambert);**监控要看 perplexity/长度/KL 而非只看 reward**(复旦);**code RL 别信 public test**(Open-R1);**clip-higher/Dr.GRPO 这些"标配"都有模型/规模依赖的反例**(Lite-PPO)——**没有放之四海皆准的 recipe,只有"先把一阶的事做对、再调二阶"**。

---

## §49 PG 算法家族谱系 + 选型决策树

> 共识 14 讲了"IS 粒度之争",这一节给**完整谱系图 + 一棵可操作的选型决策树**,把 PPO→GRPO→DAPO→GSPO/CISPO/… 这一路的关系理清,并回答"我该用哪个"。

### 谱系:沿三条轴演化

所有这些算法的差异可归到三条轴(Lambert 的"3 点差异"扩展版):

| 轴 | 取值谱 | 代表 |
|---|---|---|
| **A. 要不要 value function** | 有 critic ↔ critic-free(组内 baseline) | PPO/VAPO(有)↔ GRPO/RLOO/REINFORCE++(无) |
| **B. IS 比率 / trust region 形式** | 双边 clip(丢 token)↔ 裁权重(不丢)↔ 序列级 ↔ L2 软约束 ↔ 几何平均 | PPO/GRPO/DAPO ↔ **CISPO** ↔ **GSPO** ↔ **K1.5** ↔ **GMPO** |
| **C. loss 归一化** | per-sample(/\|o\|·/std)↔ token-level Σ ↔ 去 std/去长度归一 ↔ group total length | GRPO ↔ DAPO/Skywork ↔ **Dr.GRPO** ↔ Magistral |

**主干演化线**:
```
REINFORCE ─ baseline ─→ RLOO / REINFORCE++ ─ 组内 baseline ─→ GRPO(DeepSeek-R1)
   │                                                              │
   └─ +critic+GAE ─→ PPO(ORZ 用 vanilla PPO) ─→ VAPO(value-based)│
                                                                   ↓
                                          DAPO(clip-higher + dynamic sampling + token-level)
                                          ├─ Dr.GRPO(去两个归一偏置)
                                          ├─ Lite-PPO(极简两件套反超 DAPO)
                                          ├─→ GSPO(序列级 IS,免 MoE routing replay)
                                          ├─→ CISPO(裁权重不丢 token,保 fork token)
                                          ├─→ GMPO(几何平均,抗离群)
                                          └─→ +TIS/MIS(修 train-infer)/ 双侧 clip(异步)
   K1.5(在线镜像下降 + L2 软信赖域)─→ K2(L2 trust region)─→ prime-rl DPPO(log² KL)
```

### 选型决策树(实操)

```
选 PG 算法?
├─ 任务/规模:dense 中小模型 + 近 on-policy + 可验证 reward
│   └─→ GRPO / DAPO(clip-higher 0.2/0.28)+ token-level loss + TIS 兜底。最成熟,首选。
│        ⚠️ 若是 base 模型或很小:试 Lite-PPO 极简两件套,clip-higher 可能无效(§10E)
├─ MoE 模型
│   └─→ 首选 GSPO(免 routing replay,§10A);或 MiniRL R3 / DeepSeek-V3.2 Keep Routing 对齐路由
├─ 大量 off-policy 复用(每 batch 多轮更新)
│   └─→ CISPO(别 clip 掉 fork token,§10B);ScaleRL 也选它(§10F)
├─ 高 staleness / 异步 / 去中心化
│   └─→ 双侧 clip(ε=0.2, δ=4,§41E);别用纯 GSPO(async-8 会崩);配 staleness 上界(AReaL=4 / ROLL Flash α≈2)
├─ 多轮 agentic(长程、稀疏 reward)
│   └─→ 上层用 GRPO/PPO,叠加:① 轨迹过滤(void turn §17A / 低方差 §17B);② step 级信用(GiGPO §17D);
│        ③ 监控梯度范数 + reward std + MI(共识 15)
├─ 想要 value-based 的低方差(长 CoT 易"断裂")
│   └─→ VAPO / decoupled GAE(Seed-Thinking §9);或 PPO + GAE λ=γ=1(ORZ §10H)
└─ 不可验证 reward(开放写作/主观)
    └─→ 不是 PG 算法的问题,是 reward 的问题 → 走 §47 GRM 路线 + 加回 KL

横切(无论选哪个):
  · train-infer 对齐:lm_head FP32(§44 ③)+ TIS/MIS(注意争议 §41B)+ token-id 对齐
  · loss 归一:token-level Σ(共识 5),别 per-sample /|o|;长度控制放 reward 不放 loss
  · 监控最低配:entropy + clip ratio + IS ratio 分布(+ 多轮加梯度范数 + reward std)
```

### 一句话选型

> **90% 的情况:dense 用 DAPO/GRPO + TIS,MoE 用 GSPO,异步用双侧 clip,多轮叠轨迹过滤 + step 级信用。** 新算法(GSPO/CISPO/GMPO)解决的是特定痛点(MoE 路由 / fork token / 离群 ratio),**不是无脑升级**——先确认你有那个痛点(对照共识 14 的"如何抄"与 Lite-PPO 的反例),再换。**算法是二阶,数据/环境/reward/infra 是一阶(§48)。**

---

## §50 中文社区 + 框架操作层实战排障手册(practitioner lessons II)

> §48 提炼的是英文 writeup(Open-R1 / Lambert / 复旦 / Cameron Wolfe);这一节补的是**中文社区(知乎 / CSDN / 博客园 / 火山·昇腾文档)+ 框架代码操作层**的踩坑——也就是"你真把 verl / OpenRLHF / slime 跑起来才会撞到、论文正文绝不会写"的那一类。组织方式从"按工作"换成 **"按操作环节 / 按症状"**:框架操作 → 训推一致深化 → 异步 staleness 深化 → 稳定性 folk 争议 → reward 实操 → 多轮 agentic → 通用调参。**每条带出处,争议项与"待验证"项明确标注**;凡前文已讲透的(TIS、FP32 LM head、void turn、overlong filtering、长度/难度偏置 = Dr.GRPO)只给指针不重复。
>
> 一句话定位:**论文教你"为什么",框架代码和社区帖教你"具体哪一行会崩"**。

### 50.1 框架操作层(上手 verl/OpenRLHF/slime 第一周的坑)

> 这些不是算法,是"配置/转换/显存"层面的硬坑。它们工程价值极高(一个坑能卡你两天),但任何论文都不会写。

- **⭐ verl 强制重算 logprob,不用 rollout engine 返回值**:不论 HF generate 还是 vLLM generate 都能返回每个 token 的 prob,但 verl 在更新阶段(stage2)会用 trainer 的 forward **重算一遍**,丢弃 generate 给的值。原因:generate 时若用了 sampling 后处理(temperature/top-p/repetition penalty 等),inference engine 返回的是**被这些策略改过的** logit,不是模型原始 logit。**这正是 §44 train-infer 不一致的实操根源**——你拿到的"两套 logprob"(infer 的 $`\pi_{\text{infer}}`$ 与 trainer 重算的 $`\pi_{\text{train}}`$)天然不等,也是 TIS/MIS 必须存在的理由。**踩坑提醒**:自己手搓 pipeline 时若图省事直接用 generate 的 logprob 当 $`\pi_{\text{old}}`$,会把后处理偏差混进 ratio。(火山引擎 verl 实践 / CSDN verl 避坑指南)
- **verl GRPO 内部 repeat-n + uid 分组**:GRPO 要对一条 prompt 采样 n 次,verl 的做法是 stage1 把原始数据 **repeat n 份**,stage2 再按 **uid** 把同源样本聚到一起算 group 的 mean / std。**排障**:advantage 出现"全组同号""分组数对不上""归一化异常"时,先查 uid 是否正确传播、n 是否与 `actor_rollout_ref.rollout.n` 一致——多半是分组配置错,不是算法错。(同上)
- **⭐ DAPO → GRPO 切换的参数坑(直接报错或静默跑歪)**:从 DAPO recipe 改成 GRPO 时,容易漏改四处——① **删** clip 上限(DAPO `clip_ratio_high=0.28`,GRPO 回对称 `0.2`);② **删** DAPO 专属的 dynamic sampling / 过采样参数;③ **删** soft overlong punishment 相关参数;④ **重新打开 KL**(DAPO `use_kl_loss=False`、GRPO 一般 `use_kl_loss=True` + `kl_loss_coef≈1e-3`、低方差估计 `low_var_kl`)。漏一处就报参数缺失或行为异常。(昇腾社区《后训练强化学习最佳实践》算法切换对照)
- **⭐ Megatron 词表 padding 致权重 reshard 形状不匹配**:Megatron 为 TP 高效会把词表 padding 到整数倍(`make_vocab_size_divisible_by`,默认 **128**)。HF↔Megatron 权重互转时 embedding/lm_head 形状对不上 → 报错或 logit 错位。**修法**:转换时**显式设 `--vocab-size`**(裁掉 padding 部分)。这是 slime 文档点名的坑;它也间接影响 train-infer 对齐(padding 位的 logit 是 garbage,别让它进 softmax/IS)。(slime 文档 / Megatron checkpoint reshard)
- **OpenRLHF OOM 处理阶梯(按优先级)**:① **大模型先关 `--colocate_*`**——colocate 把多模型塞同组 GPU,省卡但最易 OOM,大模型务必分散;② 开 `--adam_offload` / `--ref_reward_offload`(优化器状态 / ref 模型卸到 CPU);③ 减 `rollout_micro_batch_size` 与 `micro_train_batch_size`;④ 调低 vLLM `gpu_memory_utilization`(给 ZeRO-3 的参数/梯度/优化器留显存);⑤ `--packing_samples` 省 padding。**反向**:小模型/短 context 反而要 `--colocate_all_models` + `--vllm_enable_sleep` 省卡。节点配比经验 **vLLM:Actor:Critic ≈ 1:1:1**(70B + 48×A100 → 各 16 卡);多机 `--vllm_sync_backend nccl`(vLLM 0.7.2+);`n_samples_per_prompt>1` 时开 `enable_prefix_caching`。(OpenRLHF 官方 README / 文档)

### 50.2 训推一致性深化(§44 续:FP16 新派 + 让 IS 爆炸"有数")

> §44 已把 6 个物理来源 + TIS 争议 + FP32 LM head 讲透。这里补三件 §44 没有的:**FP16 全局精度这条新路线**、**IS 比率指数爆炸的具体量级**、**sampling-mask 不一致**。

- **⭐ FP16 替 BF16:把 mismatch 直接从根上消掉(Sea AI Lab × NUS, [2510.26788](https://arxiv.org/abs/2510.26788),code [sail-sg/Precision-RL](https://github.com/sail-sg/Precision-RL))**:论文把 train-infer 失配的**根因落到浮点格式本身**。BF16 = 8 指数 + **7 尾数**(范围大但精度低),尾数少 → 舍入误差大 → 在自回归生成里逐 token 累积 → 训推分布显著漂移。FP16 = 5 指数 + **10 尾数**,精度高,**几行代码切过去**即可:GRPO / GSPO / TIS / MIS / 朴素 PG 全部更稳、收敛更快;**verl + Oat 双框架、Dense-14B + MoE + LoRA 全验证**;最朴素的 IS-PG 在 FP16 下**超过所有 BF16 baseline**,且"很多算法修正在 BF16 下本就乏力"。**代价**:FP16 动态范围窄(5 指数),易上下溢,需 **loss scaling** 兜底。**与 §44③ 的关系**:FP32 LM head 是"只改最敏感那一层"、FP16 是"改全局格式"——**两条互补路线**,都指向同一结论:**train-infer 不一致首先是精度问题,不是算法问题**。这是对共识 14 / §44 的重要更新,值得在新项目里直接试。
- **⭐ 让"IS 比率为什么必须 clip"有数**:设每个 token 的 logprob 训推差 **0.01**、序列长 **2k** → 序列级 log-ratio 累积差 **≈20** → 权重 $`e^{20}\approx 4.8\times10^8`$,**单条样本就能主宰整个 batch 的梯度**。所以 sequence-level IS "理论正确、实践危险",离不开 **clip + reward 归一化 + 小 lr + 频繁 policy 刷新** 这一整套工程纪律(cnblogs《训推误差与 IS》)。这条数字解释了为什么 §44 的 IS ratio 健康区间要卡在 [0.8, 1.27]、p99 > 5 必上 TIS。
- **sampling truncation 不一致(易漏)**:rollout 端 top-p/top-k **截断了词表**、训练端 forward 看**全词表** → 破坏 IS 恒等式(分母的支撑集都不同)。修法是记录并复用采样 mask("Keep Sampling Mask"),需把 rollout API 从 `(token_ids, logprob, finish_reason)` 扩展到**带 sampling mask / expert routing**——这是个 breaking change,目前开源框架普遍没做(HF《16 个开源 RL 库的教训》/ DeepSeek-V3.2 §33D)。**MoE 同理**:gating 的浮点舍入让训推选到不同 expert,prob 大跳变——呼应 §44 来源 5 + §10A GSPO。

### 50.3 异步 / staleness 深化(§41 续:正负不对称是最值钱的 gem)

> §41/§44④ 已讲双侧 clip、staleness 上界、partial rollout。这里补**正负样本对 staleness 的不对称性**(一类被反复独立发现的规律),以及一组让"异步该多激进"有据可依的工程数字。

- **⭐⭐ staleness 对正/负样本不对称——异步 RL 最该知道的一条**:实测发现**正奖励样本对陈旧度容忍极高,负奖励样本在中等 off-policy 下就崩**。**数学根因**:负优势项是**无界优化**——总能靠把 $`\pi_\theta(\tau)\to 0`$ 进一步降 loss,陈旧数据下这会灾难性地"在不再探索的概率空间里挖坑",抹掉有用特征;baseline 能降方差但**消不掉这个符号不对称**(在 GRPO / PPO / 原始 REINFORCE 都存在)。**修法**:① **非对称 IS 裁剪**——对负样本更紧的 clip(TOPR / verl Rollout Correction);② 对**负优势 + off-policy** 的样本直接 **zero loss**(OPSM);③ baseline 取略低于行为策略期望 reward,偏向正样本(**AsymRE** [2506.20520]);相关谱系还有 **M2PO**(约束 IS 二阶矩,[2510.01161])、**BAPO**(自适应平衡正负 clip,[2510.18927])。**实操**:异步/高 staleness 训练崩、且崩前负 advantage 样本占比高时,**先对负样本下手**,别一律加 KL。(HF《16 个开源 RL 库的教训》+ 上述论文)
- **staleness 一定要设上界并记版本号**:AReaL / ROLL Flash / verl-async 都给 $\eta$(最大版本偏差)+ 全局安全机制;**AReaL 把 IS weight 截在 5.0**;`staleness_threshold=0` 近似同步、`>0` 允许 rollouter "抢跑"。增大 $\eta$ 提吞吐但降收敛——**异步不是越异步越好**,是吞吐/新鲜度/稳定的三角平衡(工业级 Agentic RL 选型指南 / Hands-on Modern RL B.1)。
- **weight sync 的量级感(决定异步上限)**:NCCL broadcast ~**100–500ms**;verl NCCL + bucketing ~**20ms**;vLLM `packed=True` 把参数打包成 ~1GB uint8 双缓冲、跨 CUDA stream;checkpoint-engine + Mooncake RDMA 给 **Kimi-K2(256×H20)更新万亿参数 ~16–17s**;**LoRA adapter-only sync ~50MB、亚毫秒**(但 OAT 要在 ZeRO-3 关 fused lm_head、ROLL 要关 gradient checkpointing)。(HF《16 个开源 RL 库的教训》)
- **partial rollout 四味(长尾必备,§50.6 会再引)**:① **implicit continuation**(PipelineRL 永不停生成,逐 transformer forward 换权重,间隙 ~1–10ms);② **abort + prefix-resume**(SkyRL / slime);③ **explicit save-resume / Sleep-Resume**(verl 存 partial token id + logprob,sync 后续上);④ **group cancel**(prime-rl 丢弃 in-flight 整组)。越激进吞吐越高、一致性越差。
- **straggler 的真实量级(为什么必须 partial/async)**:vLLM 单卡 H100 上 **7B≈6300 tok/s、32B≈1200 tok/s**;一个 GRPO step(G=8×64=512 rollouts、32K token)**7B≈45 分钟、32B≈3.7 小时**;即便 8 卡推理,32B 的 32K rollout 仍 **~28 分钟/step**。轨迹长度 1K–32K 不均 → 一个 batch 等最慢那条 → 长尾。(HF《16 个开源 RL 库的教训》)

### 50.4 稳定性 folk 争议(带立场,明确标注"社区观点 / 待验证")

> 这一块是中文社区高信号但**有立场**的观点,与正文共识存在张力。照实写出来 + 标注边界,比假装没有更有用。

- **⭐ "RL 崩溃的根因常在 RL 之前的 SFT 起点"**:多篇社区/论文观察到一个反向关系——**SFT 阶段 entropy loss 越低 / epoch 越多,后续 RL 越早 reward collapse**。机制:过拟合 SFT 把策略**困在专家轨迹的僵化思维模式**里(尤其当 SFT 数据是同一模型自蒸馏的),探索力下降 + 明显熵崩。**修法**:控 SFT epoch 别过拟合、动态调 SFT loss 权重(dynamic-γ)。**实操含义**:RL 一开训就崩、或"训得越久的 SFT 起点崩得越早"时,**回看 SFT 的熵和 epoch,别只在 RL 超参里打转**(对照共识 8 + 仪表盘类 1)。(知乎/火山 ADG 社区《LLM-RL 训练崩溃元凶》及相关熵机制论文,机理待你的设置上复验)
- **entropy 是"双向"问题,不是只会塌**:除了 collapse(过早收敛、性能进平台),还有 **explosion**(熵被噪声淹没、信用分配恶化);**只治塌可能诱发爆**。固定熵系数极脆弱:**小系数防不住崩、大系数引发爆**,实验设置微变就能让精心调的系数反向。趋势是 **adaptive coefficient + target entropy**(把熵稳在一个区间,而非单调抬/压)——对照共识 12 / §10L,但强调"双向"这层。(知乎《熵机制》论文解读 / mlpod)
- **⚠️ "GRPO 小 batch 下有致命稳定性缺陷"(社区争议观点)**:有 practitioner 主张——GRPO 靠**大 batch 压低 PG 方差**才稳,DeepSeek 能用好是**数据量大到"完美避开"了这个缺陷**;**中小规模 / 小 batch(百万~千万参数级、batch 偏小)下 GRPO 稳定性缺陷会致命,这类规模建议优先用带 critic 的算法**。**边界**:这与"critic-free 是主流"(共识 + Lambert §48"GRPO 不是特殊算法")有张力——分歧点是**规模与 batch**,不是绝对优劣。把它当"小 batch 别硬上裸 GRPO,要么加 critic、要么靠 dynamic sampling 保住有效样本数"的提醒。(知乎《为什么 GRPO 很容易训飞》——立场鲜明,按你的规模验证)
- **GRPO 长度/难度偏置的 folk 版**:负 reward 时**长答案稀释 per-token 惩罚** → 模型学会"又臭又长的错误答案";低 std 的题(近全对/全错)被**过度加权**。这是 §10D Dr.GRPO 的工程直觉版,也是为什么要 **token-level loss(共识 5)+ dynamic sampling(共识 6)**。(CSDN《为什么 GRPO 容易 reward 崩塌》/ 知乎)

### 50.5 reward 设计实操(论文给原则,这里给数字 + 反直觉)

> 共识 4/5/13 给了"outcome-only、别 length-norm、判分两域分治"的原则。这里补社区验证过的**具体参数和反直觉结论**。

- **⭐ format reward 常非必需、甚至有害**:只用 accuracy reward + **prompt 里约束格式**,模型往往很快自学会所需格式;**显式 format reward 反而给 reward hacking 留口子**(模仿格式但不真推理)。**先纯 outcome,确有需要再加 format,并配反 hacking 惩罚**(呼应共识 4)。
- **⚠️ dense reward 比 sparse 更易 hacking(反直觉)**:全密集 / 分段密集奖励常表现为 **reward 快速但不稳上升 + 响应长度骤缩** = 表面模式记忆而非真推理;**稀疏离散反馈反而训练更稳**。与"过程奖励/PRM"潮流有张力,边界看任务可验证性(对照共识 13 / §47)。
- **截断样本:mask > 给负奖励**(共识 5 的实操数字):给截断样本负 reward 会让"推理有效、只是太长"的样本困惑、引入噪声;**mask 掉不进 loss**(overlong filtering)同时改善性能与稳定。若要软惩罚,**DAPO 实配**:`L_max=16K`、`L_cache=4K`,在 **[12K,16K]** 区间内长度越界越多惩罚越大、线性到 **-1**,超 16K 给满 -1,叠在 rule reward 上。(DAPO / Interconnects)
- **设 length reward 必配 anti-repetition / diversity reward**:否则模型直接用**重复内容刷长度**。核心原则:**奖励正确的长推理、惩罚重复的长冗余**。
- **model-based verifier 召回高但极易被 hack**:静态评估召回比规则高,但策略会学会生成"骗过 verifier 接受逻辑"的输出(**单个符号 / 乱码被判对**)。**hybrid = rule 精度 + ML 召回**较稳,但**仍需持续监控更新**(对抗性优化会找新漏洞)——呼应共识 13 / §47。
- **code RL 专属(§48 / §33 续)**:**public test 不可信**——7 个过了 public test 的 R1 解在 full/hidden test 上**全挂**;务必用 full/hidden test。每题**独立 Docker 镜像**防污染;**过滤掉与评测集同仓库的题**(DeepSWE/R2E 过滤 sympy 等);**测试用例数量 scaling 直接提 pass@1**(SWE-RL)。二元 0/1 reward 稀疏(抓不住"部分正确"),partial credit 能缓解但**会扩大 hacking 面**,慎用。(知乎 DeepSWE/SWE-RL 解读)

### 50.6 多轮 agentic 实战(ROLL "苦涩的教训"等)

> 共识 15 + §17A–F 已讲多轮独有崩溃(void turn / echo trap / 熵失控 / template collapse)。这里补**环境/行为层**的 folk lessons,论文很少正面写。

- **⭐ 终端类 agent 两大失败模式(ROLL 苦涩的教训)**:① **unproductive loops(无效循环)**——已有明确失败信号,仍**重复同一种策略、不切换思路 / 不重审假设**,形成冗长无效交互链;② **timeouts(超时)**——模型对**长命令执行时长缺乏可靠感知**,被默认超时机制误导 → 误判 / 反复重试。**启示**:环境侧给"执行时长 / 失败原因"的显式反馈,reward/prompt 侧鼓励"失败后换假设"。(ROLL《苦涩的教训》)
- **并行调用集中在"检查类"工具,改环境的执行/编辑操作天然串行** → 设计启示:**状态改变前,显式鼓励一个"前置的、并行的信息收集阶段"**(先并行查,再串行改)。(同上)
- **长尾的二八定律 = 别丢最贵的样本**:Agentic 输出长度极不均,**频繁触达 context 上限的那条轨迹,往往恰是最有价值、不可替代的 hard exploration case**。简单按长度截断/丢弃 = 丢掉最值钱的样本 → 用 partial rollout 保留(对照共识 10 / §50.3)。(工业级 Agentic RL 选型指南)
- **黑盒 agent 框架的非 MDP 轨迹**:用黑盒框架采样的轨迹常不满足逐步马尔可夫性,主流思路是**放弃逐步 MDP 假设,在轨迹级 / turn 级做策略优化**;多轮极易崩,需配自适应正则(对照共识 15)。(知乎《黑盒 Agent 框架下的非 MDP 轨迹》)
- **一句共识**:**"Agentic RL 不是单一 RL 算法,而是环境建模 + 学习信号 + 异步数据流 + 策略优化 + 基础设施的协同系统"**——多源反复强调(ROLL / 选型指南),呼应共识 11 + 15。

### 50.7 通用 RL 调参(经典 folk checklist,新手最常问)

- **三大永恒痛点**:**sample inefficient(样本效率低)/ 对 seed & 超参敏感 / reward function 设计难**——几乎所有 RL 算法都在围这三点做文章;评估时还要注意 **env wrapper 对 episode reward/length 的修改会污染对比**。(腾讯云《深度 RL 训练与调参技巧》)
- **学习率**:经验"**小模型大 lr、大模型小 lr**";**warmup 必加**(RL 初期对 lr 极敏感、易梯度爆炸);LoRA lr ≈ **1e-5~2e-5**(对照共识 7:full-param actor ≈ 1e-6、SFT ≈ 5e-6~7e-6——量级差一个数量级,别照搬 SFT lr 到 actor)。
- **GRPO 经验默认值**:`n=8~16` 个 response / prompt;`temperature≈0.7~1.0`(太低损探索、太高引噪声);KL 用低方差估计 `low_var_kl`、`kl_loss_coef≈1e-3`(GSM8K 经验值)。(CSDN/博客园 GRPO 复现帖)
- **监控优先级**(对照仪表盘 + §48 复旦):**perplexity / response length / 对 SFT 的 KL 比单看 reward 和 loss 更能反映稳定性**;最低配三件套仍是 **entropy + clip ratio + IS ratio 分布**。

### §50 一句话总纲

> **train-infer 不一致先查"logprob 是不是重算的 + 精度"**(FP16 全局 [2510.26788] / FP32 lm_head §44③),别一上来调熵;**异步崩先看负样本**(正负 staleness 不对称,对负样本更紧 clip),别一律加 KL;**RL 一开训就崩先回看 SFT 起点的熵和 epoch**,别只在 RL 超参里打转;**reward 先把 outcome 做对再谈 shaping**(format 非必需、dense 易 hacking、截断要 mask);**框架层先做对**(verl 重算 logprob / Megatron vocab padding / OpenRLHF colocate 阶梯)——这些操作坑不解决,算法再对也白搭。**一句话:论文是一阶认知,框架代码和社区帖是把一阶落地时真正会绊倒你的二阶细节。**

---

> **本轮更新(2026/06)小结**:在原 13 条共识 + 47 节基础上,新增 **共识 14(IS 粒度之争)、共识 15(多轮崩溃模式)**,Part I 加 §10A–§10L(GSPO/CISPO/GMPO/ScaleRL/ProRL/Dr.GRPO/Lite-PPO/ORZ/K1.5/小模型配方/熵动力学),Part II 加 §17A–§17F(SimpleTIR/RAGEN/EPO/RAGEN-2/GiGPO/Tree-GRPO/ARPO),Part III 加 §24A(新 web 工作),Part IV 加 §33A–§33D(SWE 环境构造/开源 SWE 模型/自博弈 reward/前沿 coding——**重点关注代码**),Part V 加 §41A–§41F(ROLL/slime/NeMo-RL/prime-rl/INTELLECT/Trinity/TRL),Part VI 扩 §44(secretly off-policy + TIS 争议)、新增 §48(实战教训)、§49(算法谱系 + 选型决策树)、**§50(中文社区 + 框架操作层实战排障手册)**。**核心新增主题:序列级 IS、多轮独有崩溃、train-infer 的 TIS 争议与 FP32 LM head、code RL 的环境构造与 reward hacking;§50 进一步补 FP16 全局精度修 mismatch、staleness 正负不对称、SFT 起点致 RL 崩、verl/Megatron/OpenRLHF 框架操作坑等"代码层"know-how。**
