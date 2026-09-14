# DSpark 解读

> **Paper**：《DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation》
> **Authors**：Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong et al.（Peking University + DeepSeek-AI）
> **Repo**：deepseek-ai/DeepSpec（开源 DeepSpec 训练框架 + DSpark / DFlash / Eagle3 实现 + DSpark checkpoints）

---

## 前置知识：写给小白的投机解码入门

> 这一节不需要任何背景，先用大白话把「投机解码（Speculative Decoding）到底在干嘛、凭什么又快又不掉质量」讲明白。懂的人可以直接跳到下面的 Take Home Message。

~### 第一件事：大模型为什么打字这么慢？

你跟大模型聊天时，它的回答是**一个字一个字往外蹦**的。这不是装样子，而是它真的只能这么写——

大模型写字的规矩是：**每写一个字，都要把前面所有内容从头读一遍，才能决定下一个字写什么。** 写「今」之前读一遍，写「天」之前再读一遍，写「天气」前又读一遍……要写 100 个字，就得这样从头折腾 **100 次**。每折腾一次都很慢，所以整体就慢。

我们把「从头折腾一次」叫做**一次前向**（forward pass）。记住这个词就行：**大模型每吐一个字 = 跑一次前向，而一次前向很贵。**

### 第二件事：一个反常识的便宜——「捎带」几乎不要钱

这里有个关键的小秘密，是整个投机解码能成立的根基：

> **大模型跑一次前向，检查 1 个字和检查 5 个字，花的时间几乎一样。**

听起来很怪？打个比方就懂了：

> **大模型跑一次前向，就像开一趟大卡车去仓库。** 这趟车油费很贵（车大、路远），但贵的是「开这一趟」本身。你这趟车里拉 1 箱货还是拉 5 箱货，油费几乎没差别——反正车都开了。
>
> 普通的逐字生成，就是**每次只拉 1 箱货**，开一趟、拉一箱、再开一趟、再拉一箱……白白浪费了卡车的空间。

> **拓展：这趟车的「油费」到底花在哪？** 关键词只有一个：**memory-bound（受显存带宽限制）**。
>
> 不是「函数启动」那种开销。大头是**搬权重**——大模型几千亿参数、几百 GB 全存在显存里，每跑一次前向都得把这几百 GB 从显存完整搬到计算单元过一遍，慢就慢在这一段「搬运」。而权重一旦搬进来，顺手算 1 个字还是 5 个字几乎不要钱（算力本来就闲着）。所以**贵的是搬权重、不是算字**，这就是「油费跟拉几箱货无关」的物理真相。（函数/内核启动也有固定开销，但比搬权重小一两个数量级，是 §4.3 的 ZOS 顺带优化的次要项。）

所以大模型每次「只写 1 个字」其实非常亏：车明明能装 5 箱，你只放了 1 箱。**那能不能想办法，让这一趟车把 5 个字一次拉回来？**

### 第三件事：投机解码的妙招——找个小助手先猜

既然大模型「一趟车能顺便检查好几个字」，那就别让它从零想了，**找个又小又快的小模型先把后面几个字猜出来，大模型一趟车把这几个字一起检查掉。**

* **小模型（草稿模型，drafter）**：又笨又快，但写得飞快。负责**先蒙**接下来几个字。
* **大模型（目标模型，target）**：又准又慢，就是你真正想要的那个。负责**一次性检查**小模型蒙的对不对。

继续用打工的比方：

> **小模型是手快的实习生，大模型是把关的专家。**
> 实习生啪啪啪先把草稿写出来：「今天天气真好」。
> 专家开一趟车（跑一次前向），把这 6 个字一起扫一遍，逐个判断「这个字是不是我自己也会这么写」。

### 第四件事：一轮，到底怎么走？（看这张图）

```
 ① 小模型飞快蒙 5 个字：       今   天   天   气   真
                              （它写得快，这步很便宜）
                                       │
                                       ▼
 ② 大模型开一趟车，一次检查这 5 个字：
                              今✓  天✓  天✓  气✓  真✗
                              （前 4 个它认可，第 5 个"真"它不同意）
                                       │
                                       ▼
 ③ 规则：从第一个不对的地方"一刀切"
        · 前面连续对的「今天天气」→ 全部采纳（4 个字白赚！）
        · 出错的「真」以及它后面的 → 全部作废
        · 大模型顺手把它自己想写的那个字补上 →「很」
                                       │
                                       ▼
 ④ 这一轮：白赚 4 个 + 大模型补 1 个 =「今天天气很」共 5 个字
    可大模型只开了 1 趟车！下一轮从「很」接着蒙……
```

**为什么这就快了**：老办法逐字写「今天天气很」要开 **5 趟车**；投机解码同样 5 个字只开了 **1 趟车**（外加小模型那次很便宜的猜）。**车开得越少 = 越快**，小模型蒙得越准、一次被采纳的字越多，省的趟数就越多。

> **⚠️ 记牢一条铁律：草稿一旦有一个字被拒，它后面的字——不管对不对——全部作废。** 这条规则贯穿全篇。
>
> 为什么？假设草稿是「今天天气**真好**啊」，V4 把第 5 个字「真」拒了、改成「很」。那第 6 个字「好」当初是小模型**接在「真」后面**蒙的；可现在前文已变成「…天气很」，「好」的前提整个塌了，再问它对不对毫无意义。所以从第一个被拒处**一刀切**，后面无条件全扔。
>
> 这条铁律有个重要后果——**越靠前的字越金贵**：第 1 个字若被拒，整块草稿当场全废、这一轮白干（只白赚到那 1 个 bonus）。所以 DSpark 才拼命把**第 1 个字蒙准**；也正因如此，并行 drafter 的**「后缀崩塌」才那么要命**——靠后某字错一个，它后面的全跟着陪葬（这正是 §1 串行头要解决的）。

### 第五件事：幕后——模型自始至终只会做一件事

上面图里「大模型怎么检查、怎么补字」，底层简单得出奇。记住一句话：**语言模型（不管是 V4 还是小模型）一辈子只会做一件事——给它前文，它吐出「下一个字该是什么」的一张概率表。** 出草稿、验证、修正，全是这同一个动作换着用：

* **出草稿** = 让小模型干这件事再抽样：把 anchor 喂进去 → 它一次并行吐出后面 γ 个位置「下一个字」的概率表 → 各抽一个 → 草稿「今天天气真」。
* **验证** = 让 V4 干这件事再跟草稿对一下：把整句 `[D, 今, 天, 天, 气, 真]` 一次喂给 V4，它在**每个位置同时**吐出「它自己想要的字」，逐位置和草稿比对，取最长一致前缀。
* **修正** = 对不上的那个位置，V4 自己那张概率表里的字直接拿来当 bonus，不用另算。

> **V4 凭什么能「看着 D今天天气」预测下一个字？** 因为草稿「今天天气」是被**当成输入喂进去的**。V4 在「气」后面那个位置，看到的前文正好是「D今天天气」，自然给得出它想接的字——这就是「一趟车能核对一整串草稿」的根本原因。

一句话收束：**模型只会「看前文 → 吐下一个字」。小模型这么干 = 出草稿；大模型这么干、再跟草稿一对 = 验证；对不上时它表里的字 = 修正。三件事，同一个动作。**

### 第六件事：一轮接一轮，把整段写完

一轮结束会确定一串字（接受的那几个 + 1 个 bonus）。**最后那个 bonus，就当下一轮的起跑点（术语叫 anchor）。** 重复同样的动作，直到写完：

```
轮1  起跑 D  →  小模型蒙「今天天气真」 →  V4 验一趟 →  接受「今天天」+ 补「很」
                                                    └ 确定「今天天很」，「很」接力 ┐
轮2  起跑 很 →  小模型蒙「适合散步哦」 →  V4 验一趟 →  接受「适合」+ 补「X」      │←┘
                                                    └ 确定「适合X」，「X」接力 ┐
轮3  起跑 X  →  ……每轮都一样：从上一轮的 bonus 往后蒙、V4 验一趟、留一个新 bonus │←┘
```

要点三条：

* **唯一跨轮传递的，只有这个起跑点字（anchor）**；前面已确定的字，KV 都缓存好了，不重算。
* 小模型每轮的**输入** = 上一轮的 anchor 字 + 几个空位占位符；**输出** = anchor 后面的 γ 个草稿字 + 每个字的置信度（细节见 §1.1）。
* **每轮真正贵的只有 V4 那一趟**，但它一趟能敲定好几个字——省下的，正是这些字本来要逐个开的车。

> **一句话抓本质（懂 MTP 的看这条就够）**：DSpark 整体就是——**小模型像 MTP 那样一次「并行」蒙出后面好几个字（草稿）→ 大模型对这串草稿走一趟「并行解码」→ 只留下「和大模型自己一致的最长前缀」→ 在第一个分歧处至多再吐 1 个 bonus token**；然后以这个 bonus 当新起点，一轮一轮迭代下去。
>
> （顺带：老基线 **MTP-1 就是这套的退化版**——γ=1，小模型一次只蒙 1 个字、大模型最多确定 1 接受 + 1 bonus。DSpark 无非把「一次蒙几个」从 1 拉大，再想办法让大模型敢于、且划算地多验几个——这正好引出它的两个贡献。）

### 第七件事：那质量会不会变差？——不会，一个字都不会差

这是最让人安心的一点：**用小模型来猜，最后写出来的东西，跟大模型自己一个字一个字憋出来的，一模一样。** 不是「差不多」，是「完全一样」。

为什么能保证？因为「验证」那一步，**大模型是拿自己的标准在审，不是随便放水**：

* 小模型蒙的字，**只有大模型自己也认可、也会这么写，才会被采纳**；
* 只要有一个字大模型不满意，立刻从那里截断、把后面全扔掉，**由大模型亲自补一个它自己想写的字**。

所以小模型本质上只是个「加速器」：蒙得准 → 大模型频频点头 → 省很多趟车 → **快**；蒙得烂 → 大模型老摇头、几乎全靠自己补 → 退化成老办法那样慢，但**质量丝毫不受影响**。

> **一句话记住**：小模型负责「赌后面几个字」，大模型负责「守住质量底线」。赌中了省时间，赌错了大模型兜底——**只可能更快，绝不可能更差。**

### 第八件事：训练好的 V4 在这套里是什么角色？——被加速的「专家」，自己不重训

最容易搞反的一点：**这套投机解码里，V4 是「目标模型（target）」，就是那个把关的专家、你真正想要的大模型，它从头到尾冻结、不重新训练。真正被训练的是旁边那个小跟班 DSpark。** 换句话说，**不是「把 V4 训成会投机解码」，而是「给训练好的 V4 配一个小跟班 DSpark 来加速它」**——V4 是被加速的对象，不是被改造的对象，它的权重和输出质量全程不变（这也是上面无损的前提）。

训练好的 V4 在整套流程里干三件事：

| 时机 | V4（专家 / target）干的事 |
| --- | --- |
| **推理时——验证 + 兜底** | DSpark 蒙出一串字，V4 开一趟车并行核对、接受最长前缀；对不上时亲自补 bonus。**最终输出严格等于 V4 自己逐字写的结果，所以质量不掉。** |
| **推理时——给起点** | 每轮第一个 anchor 字由 V4 给出（上一轮的 bonus），DSpark 拿它当起点往后蒙。 |
| **训练时——当老师 + 借零件** | 训 DSpark 要拿 V4 的输出分布当「标准答案」；同时 DSpark **借用 V4 的 embedding 层和输出 head**（都冻结），自己只训骨干、串行头、confidence head 这几个小零件。 |

* **DSpark 必须训练**，但只训它自己那几个小零件（参数小、训得快），V4 一根毫毛不动。
* **一个 DSpark 绑一个 V4**：蒙得像谁就得跟谁配，论文给 V4-Flash 和 V4-Pro **各训了一个 DSpark**，不能混用。
* **DSpark 训得好坏只影响快慢、不影响质量**——质量始终由 V4 在验证那步守住。~

### 几个后面反复出现的词（看一眼就行）

| 词 | 大白话 |
| --- | --- |
| **草稿模型 / 目标模型**（draft / target） | 手快的实习生 / 把关的专家 |
| **一次前向**（forward pass） | 大模型「开一趟车」，很贵 |
| **block size $`\gamma`$** | 实习生一次蒙几个字（比如 5） |
| **接受长度 $`\tau`$** | 这一轮总共白赚 + 补出几个字，**越大越快**，是全文最重要的指标 |
| **接受率** | 实习生蒙中的命中率 |
| **bonus token** | 专家在草稿末尾亲手补的那一个字（也叫 anchor，是下一轮的起点） |
| **MTP** | DeepSeek 自带的一种最简单草稿方式；旧线上基线 **MTP-1 就是「一次只蒙 1 个字」**，DSpark 要取代的就是它 |

懂了这些，下面 §0 会进一步讲：**「实习生」这个小模型该怎么造？** 业界有两条路（一个准但慢、一个快但飘），各有死穴——而 DSpark 干的事，就是把这两条路的优点拼起来、缺点补掉。

---

## Take Home Message

> **一句话总结**：DSpark 是 DeepSeek-V4 在线服务真正使用的投机解码（Speculative Decoding）方案——用「**半自回归 draft**（并行骨干 + 轻量串行头）」解决并行 drafter 的后缀崩塌，用「**置信度调度的验证长度**（confidence head + 硬件感知调度器）」解决高并发下盲目验证浪费 batch 容量的问题。两个改进一个管 **draft 质量（提高接受长度 $`\tau`$）**，一个管 **系统效率（高并发下不把大模型验证算力浪费在必拒 token 上，省 $`T_{\text{verify}}`$）**，在 V4 线上流量里相对旧基线 MTP-1 把单用户生成速度提升 **60%–85%（Flash）/ 57%–78%（Pro）**。

* **背景三杠杆**：投机解码单 token 延迟 $`L = (T_{\text{draft}} + T_{\text{verify}}) / \tau`$。提速只有三条路——**draft 更快**（降 $`T_{\text{draft}}`$）、**draft 更准**（升接受长度 $`\tau`$）、**验证更聪明**（降有效 $`T_{\text{verify}}`$）。DSpark 同时压后两项。
* **两类 drafter 的死结**：自回归 drafter（Eagle3）$`\tau`$ 高但 $`T_{\text{draft}} \propto \gamma`$，被迫用浅网络小 block；并行 drafter（DFlash）$`T_{\text{draft}} \approx O(1)`$ 可以用深网络大 block，但每个位置独立预测 → 缺乏 token 间依赖（非自回归） → **后缀接受率快速衰减（suffix decay / multi-modal collision）**。
* **贡献一：半自回归生成（Semi-AR）**。并行骨干（DFlash）一次 forward 出全部 base logits——但**各位置是「同时、各算各的」，算第 2 个字时根本不知道第 1 个字会被抽成啥**，于是会拼出「当然问题」这种自相矛盾的后缀（这就是并行 drafter suffix decay 的根源）。所以再挂一个**极轻量串行头**：等前一个字定下来，回头微调后一个字的 logits、补回「位置之间的协调」。重活（读懂上下文）并行骨干已经干完，串行头只做「瞄一眼前一个字、顺手修正」的小事，所以**能轻**；又因为它是 draft 里唯一不能并行的部分，做重了就把并行省的时间赔回去，所以**必须轻**。两种实现：
    * **Markov head（默认）**：转移 bias 只依赖前一个 token，低秩分解 $`B = W_1 W_2`$（$`r = 256`$），$`W_1`$ 当 embedding 查表、$`W_2`$ 当 logit 投影。
    * **RNN head**：维护 block 内递归状态 $`s_k`$，能看到完整前缀历史，但收益边际、部署更复杂，故默认不用。
    * **关键发现**：「一点点自回归就够了」——2 层 DSpark 全面超过 5 层 DFlash，串行头加的延迟只有 **0.2%–1.3%**，却换来最高 30% 的接受长度提升。
* **贡献二：置信度调度的验证（Confidence-Scheduled Verification）**。
    * **Confidence head**：每个 draft 位置输出 $`c_k \in (0,1)`$，建模「**前缀全被接受的条件下，第 $`k`$ 个 token 通过验证的概率**」，用解析接受率 $`c_k^* = 1 - \frac{1}{2}\|p_k^d - p_k^t\|_1`$ 监督。
    * **Sequential Temperature Scaling（STS）**：神经网络置信度普遍 overconfident，调度需要的是**绝对概率值**（不只是排序），所以用逐位置 1D 网格搜索对累积乘积 $`\prod c_i`$ 做温度标定，ECE 从 3%–8% 压到 ~1%。
    * **Hardware-Aware Prefix Scheduler**：把「验证多长」formalize 成**全局吞吐最大化** $`\Theta = \tau \cdot \text{SPS}(B)`$，SPS（steps-per-second）capacity 曲线引擎启动时 profile 一次存成 cost table。利用前缀存活概率单调性做贪心 + early-stop，**lossless（严格保持 target 分布）**。
* **线上部署（V4 真实服务）**：drafter 与 V4-Flash/Pro preview 共部署，骨干 = 3 个 MoE 层 + mHC + SWA(128)，$`\gamma = 5`$。工程上解决了三大冲突：
    * **训练侧**：只通信 LM head 之前的 hidden state（$`O(d)`$ 而非 $`O(V)`$）+ anchor-bounded 序列打包。
    * **调度落地**：真实 SPS 是**锯齿离散**而非光滑、且与 ZOS（Zero-Overhead Scheduling）/ CUDA graph 冲突 → 改成**异步调度**（用前两步的置信度预估容量 $`K`$，转成动态 top-$`K`$），异步反而天然构成 causal barrier，可以去掉 early-stop 做无约束全局搜索。
    * **执行层**：变长 query batch → 所有 token flatten 成独立元素，依赖关系靠 sparse attention 里的 marker tensor 传，V4 上只需改 index-attention 和 compress 两个 kernel。
* **效果**：相对单 token 的 MTP-1（旧线上基线），DSpark 把 throughput–interactivity 的 Pareto frontier 整体外推。验证预算随并发**自适应**：轻载时从 MTP-1 的固定 2 token 扩到 4–6，重载时平滑收缩，避免低置信 token 抢占关键 batch 容量。
* **短板**：即使调度器把验证浪费压到最低，**draft 侧生成首个 $`\gamma`$-block 的并行骨干开销是固定且不可回收的**；对天生低接受率的复杂 query，这部分 upfront compute 白花。未来方向：draft 内做 difficulty-aware early exit。

---

## 0. 背景：投机解码与两类 drafter

### 0.1 投机解码（Speculative Decoding）回顾

自回归 LLM 一次 forward 只出一个 token，延迟正比于输出长度。投机解码用一个轻量 draft model $`M_d`$ 提议 $`\gamma`$ 个候选 token，再让 target model $`M_t`$ **一次 forward 并行验证**，按 rejection sampling 接受「与 target 分布一致的最长前缀」并补一个 bonus token。因为接受规则严格保持 target 分布，**投机解码无损加速**。

逐位置验证规则：位置 $`k`$ 上 target 算出自己的分布 $`p_k^t`$，token $`x_k`$ 以 $`\min(1, p_k^t(x_k)/p_k^d(x_k))`$ 的概率被接受。**第一个被拒的位置 $`k`$ 会丢弃其后所有 token**（无论质量多好）——这就是「前缀存活（prefix survival）」机制，也是后面所有设计的出发点。

设单 cycle 接受 token 数为 $`\tau`$，drafting / verification 的 wall-clock 为 $`T_{\text{draft}}`$ / $`T_{\text{verify}}`$，则平均每 token 延迟：

```math
L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}
```

**提速三杠杆**：降 $`T_{\text{draft}}`$（draft 更快）、升 $`\tau`$（draft 更准）、降有效 $`T_{\text{verify}}`$（验证更聪明）。DSpark 的两个贡献正好分别压 $`\tau`$ 和 $`T_{\text{verify}}`$。

### 0.2 两类 drafter 的根本权衡

```
                draft 速度 T_draft          接受长度 τ            block 大小 γ
                ----------------------------------------------------------------
 自回归 drafter   ∝ γ（逐 token 串行）      高（有 token 间依赖）   被迫小（浅网络）
 (Eagle3 等)     ↑ 慢
                ----------------------------------------------------------------
 并行 drafter     ≈ O(1)（一次 forward）    低（位置间独立）        可以大（深网络）
 (DFlash 等)     ↑ 快                      ↓ suffix decay
```

* **自回归 drafter**（Eagle3 基于 Training-Time Test）：每个位置 condition 在前面已采样 token 上，建模能力强；但 $`T_{\text{draft}} \propto \gamma`$，只能用小 $`\gamma`$ + 浅结构。为补偿短 block，常配 tree-based verification（树注意力验证多条路径），代价是验证 token 数膨胀、拖垮 serving 吞吐。
* **并行 drafter**（DFlash 是 SOTA）：所有 $`\gamma`$ 个位置一次 forward 出，$`T_{\text{draft}}`$ 几乎与 block 大小无关，能用深网络大 block。**DFlash 的核心是 KV injection**：prefill 时把 target 若干层的 hidden state concat 后投影进 draft 空间作为 context feature，注入每个 draft 层的 K/V：

```math
H_{\text{ctx}} = \text{RMSNorm}\left(W_c [H^{(l_1)}; \dots; H^{(l_m)}]\right), \quad K_i = [W_i^K H_{\text{ctx}}; W_i^K H_d], \; V_i = [W_i^V H_{\text{ctx}}; W_i^V H_d]
```

draft model 复用 target 的 embedding 和 LM head（都冻结），输入「anchor token + $`\gamma`$ 个 mask token」，一次出全部 mask 位置的 logits。

* **并行 drafter 的两个致命瓶颈**（DSpark 要解决的）：
    1. **质量瓶颈**：位置间独立 → 无法建模 token 间依赖 → multi-modal collision（多模态碰撞）。例如上下文允许「of course」和「no problem」两种续写，并行 drafter 各位置独立采样，可能拼出「of problem」「no course」这种不连贯组合 → **后缀接受率快速衰减**。
    2. **系统瓶颈**：并行很容易出长 block，但**盲目验证全部 token 会降低系统吞吐**。理想验证长度沿两个轴变化——**数据轴**（code 这种结构化请求接受率天然高于开放 chat）、**系统轴**（轻载时多验几个几乎免费，重载时每个无谓验证都在抢本可服务其他请求的 batch 容量）。

---

## 1. 架构

DSpark 的解码 cycle（对应论文 Figure 1），用 prompt `ABC` 走一遍：

```
 ┌─ 第 1 步：target 出 anchor ────────────────────────────────────────────┐
 │  Target(ABC) ──► D            D 既是上一轮 bonus token，也是 draft 起点  │
 └────────────────────────────────────────────────────────────────────────┘
                              │ 用 D 作 anchor 输入
                              ▼
 ┌─ 第 2 步：DSpark draft（半自回归）─────────────────────────────────────┐
 │  ① 并行骨干 Parallel Block：  D, Mask, Mask, Mask ──► base logits U1..U4 │
 │                                                       + hidden h1..h4    │
 │  ② 串行头 Sequential Block：  逐位置注入转移 bias，采样出 E F G H        │
 │  ③ Confidence Head：         同时输出每位置置信度 c1 c2 c3 c4            │
 └────────────────────────────────────────────────────────────────────────┘
                              │ c1..c4 交给调度器
                              ▼
 ┌─ 第 3 步：Hardware-Aware Prefix Scheduler ─────────────────────────────┐
 │  按累积存活概率排序，结合实时引擎负载 → 保留 EFG，丢弃低置信的 H        │
 └────────────────────────────────────────────────────────────────────────┘
                              │ 只把 EFG 送去验证
                              ▼
 ┌─ 第 4 步：target 并行验证 ─────────────────────────────────────────────┐
 │  Target(D,E,F,G) ──► E ✓ F ✓ G ✗     G 被拒 → target 顺势生成修正 G*    │
 │  下一轮 anchor = G*（或最后接受 token 的 bonus）                         │
 └────────────────────────────────────────────────────────────────────────┘
```

两个核心组件：**§1.1 半自回归生成**（draft 更准）、**§1.2 置信度调度验证**（验证更聪明）。

---

### 1.1 半自回归生成（Semi-Autoregressive Generation）

**动机**：纯并行 drafter 一次出全部 logit，每个位置无法 condition 在 block 内其他已采样 token 上，边界处会 multi-modal collision，接受率沿 block 快速衰减。DSpark 把 draft 拆成两个阶段——**重的并行骨干 + 轻的串行头**，在几乎不加延迟的前提下补回 token 间依赖。

> **「并行骨干都把字算出来了，为什么还要再挂个串行头？」** 难点其实只有一个：**「同时算出来」和「能互相看见」是两回事。**
>
> **把它想成填空题：一句话挖了 5 个空，找 5 个人来填。**
>
> **并行（骨干的做法）= 把 5 个人分别关进 5 个不通气的小房间，同时下笔。** 每个人手里只有题目（前文），看不到隔壁填了啥：
> - 这句话可以是「**当然可以**」，也可以是「**没问题**」。
> - 1 号房填了「当然」；
> - 2 号房**根本不知道** 1 号填了「当然」（同时在写、又不通气），他自己在「可以 / 问题」之间犹豫，填了「问题」；
> - 交上来一拼 →「**当然问题**」，前后对不上的鬼话。
>
> 这就是并行的死穴：**5 个人同时写，飞快（这是它的优点）；但因为互相看不见，越往后的空越容易跟前面打架** ——这就是「后缀崩塌」。
>
> **对照「自回归」（慢的老办法）= 只让 1 个人，从头一个空一个空地填。** 填完 1 号「当然」，**回头读一眼**，再填 2 号，自然填「可以」。不会矛盾，但慢（必须排队一个一个来）。
>
> **串行头 = 在 5 个人交卷之后，加一个手快的小编辑。** 他不重读整篇、不重新理解题意，只**扫一眼前一个空填了啥**，把后一个空顺手改通顺（看见 1 号是「当然」，就把 2 号从「问题」改成「可以」）。
> - **它能轻**：读题、理解意思这些重活，5 个填空的人（并行骨干）已经干完了；编辑只做「瞄一眼邻居、改个字」的小事，一个小矩阵就够。
> - **它必须轻**：编辑只能顺着一个一个空地改（这步天生没法并行）。要是编辑也磨蹭，就把并行刚省下的时间又赔回去了——那还不如直接用老办法。
>
> 这就是「**semi-autoregressive（半自回归）**」名字的由来：**5 个人并行填（保住快）+ 小编辑补一点「看前一个」的顺序依赖（补回准）。**

#### 1.1.1 并行阶段（Parallel Stage）

直接用 DFlash 作骨干，一次 forward 出整个 block 的 hidden state $`h_1, \dots, h_\gamma`$ 和 base logits $`U_1, \dots, U_\gamma`$。

* **唯一改动**：原 DFlash 喂「anchor + $`\gamma`$ 个 mask」、只预测 mask 位置；DSpark 把 **anchor 本身当作第一个预测位置**，即 $`\gamma`$ 个输入 token（anchor + $`\gamma-1`$ 个 mask）产出 $`\gamma`$ 个 draft logit。**省一次 draft 计算且 draft 质量不降。**

  > **为什么能这样改？** 以 $`\gamma=4`$ 为例：
  >
  > ```
  > DFlash：输入 [anchor][mask₁][mask₂][mask₃][mask₄]  ← γ+1 = 5 个 token
  >                          ↓       ↓       ↓       ↓
  >          draft logit：  L₁      L₂      L₃      L₄   （anchor 输出被丢弃）
  >
  > DSpark：输入 [anchor][mask₁][mask₂][mask₃]          ← γ = 4 个 token
  >                ↓       ↓       ↓       ↓
  >          draft logit：L₁      L₂      L₃      L₄   （同样 4 个预测）
  > ```
  >
  > 在 autoregressive transformer 里，anchor 位置的 hidden state 输出的本来就是"anchor 之后下一个 token 的概率分布"——和专门放一个 mask₁ 让模型预测那个位置**完全等价**。DFlash 原本白白把这个输出丢掉，转而多喂一个 mask₁ 得到同样的东西。DSpark 把冗余去掉：anchor 输出直接用作 L₁，输入从 $`\gamma+1`$ 缩到 $`\gamma`$，少处理一个 token，质量不变。

#### 1.1.2 串行阶段（Sequential Stage）

在 base logit $`U_k`$ 上叠一个**前缀依赖的转移 bias** $`B_k(x_0, x_{<k}, x_k)`$，让每个位置能 condition 在 block 内已采样 token 上。注意：不是定义一个全局归一化的 energy model，而是用**自回归因子分解**诱导一个 causal block 分布（这样每个 token 的概率仍是精确 softmax，满足 rejection sampling 对「精确 per-token 概率」的硬要求）：

```math
P(X \mid x_0) = \prod_{k=1}^{\gamma} p_k(x_k \mid x_0, x_{<k}), \quad p_k(v \mid x_0, x_{<k}) = \frac{\exp(U_k(v) + B_k(x_0, x_{<k}, v))}{\sum_{u \in \mathcal{V}} \exp(U_k(u) + B_k(x_0, x_{<k}, u))}
```

> **一句话读懂**：并行骨干已经算好了每个位置的 logit $`U_k`$，串行头只是**在上面加一个 bias $`B_k`$、重新 softmax**。关键在于 $`B_k`$ 依赖 $`x_{<k}`$（block 内前面已采样的 token），所以必须等位置 $`k-1`$ 采完才能算位置 $`k`$ 的 bias——这就是"串行"的全部含义。重活（读懂上下文）并行骨干已经干完，串行头只做「瞄一眼前一个采样结果、顺手调整 logit」的小事。

* $`x_0`$：上一轮验证的 anchor token；$`U_k`$：并行骨干在位置 $`k`$ 的 base logit；$`\mathcal{V}`$：词表。
* 推理时串行头左到右按 $`p_k(\cdot \mid x_0, x_{<k})`$ 采样。**因为这步天然串行，串行头必须极轻**（$`T_{\text{sequential}} \ll T_{\text{parallel}}`$），保证整体 draft 延迟仍由并行阶段主导。

两种串行头实现：

##### Markov head（默认）

最简实现：$`B_k`$ 只依赖前一个 token，退化为一阶转移 $`B(x_{k-1}, x_k)`$。原则上是 $`V \times V`$ 的大矩阵，用低秩分解近似：

```math
B(x_{k-1}, \cdot) = W_1[x_{k-1}] \, W_2 \in \mathbb{R}^V, \quad W_1 \in \mathbb{R}^{V \times r}, \; W_2 \in \mathbb{R}^{r \times V}
```

* $`W_1`$ 是 embedding 查表（按前一 token 取一行 $`r`$ 维向量），$`W_2`$ 是 logit 投影。默认 $`r = 256`$，存储和单步计算都很小，大词表也能跑得快。
* **直观**：位置 1 采到「of」后，Markov head 在位置 2 抬高「course」、压低「problem」，缓解跨模碰撞。

##### RNN head

Markov head 只记一步。RNN head 维护一个 **$`r`$ 维递归状态 $`s_k \in \mathbb{R}^r`$**（$`r=256`$），把 block 内整个已采样前缀的历史压进去。

每步的输入拼接：$`z_k = [s_{k-1};\; W_1[x_{k-1}];\; h_k] \in \mathbb{R}^{2r+d}`$（RNN 状态 + 前一 token embedding + 骨干 hidden），做 GRU 风格门控更新，再投影出 bias：

```math
s_k = \sigma(W_g z_k) \odot s_{k-1} + (1 - \sigma(W_g z_k)) \odot \tanh(W_c z_k), \quad B_k = W_2^\top \tanh(W_o z_k)
```

> **$`s_k`$ 怎么影响下一个 token 的分布**：$`s_k`$ 本身不是概率，它在下一步被拼进 $`z_{k+1}`$，经过 $`W_o`$ 投影 + tanh，再由 $`W_2^\top`$ 展开到词表大小，得到 $`B_{k+1} \in \mathbb{R}^{|V|}`$，叠到骨干 logit $`U_{k+1}`$ 上重新 softmax。整条链路：$`s_k \xrightarrow{z_{k+1}} W_o \xrightarrow{\tanh} W_2^\top \rightarrow B_{k+1} \rightarrow p_{k+1}`$。

* $`W_g, W_c, W_o \in \mathbb{R}^{(2r+d) \times r}`$ 由一个 linear 投影切成 gate / candidate / output 三份，$`s_0 = 0`$。
* **vs Markov head**：Markov head 的 bias 只查前一个 token 的 embedding（$`W_1[x_{k-1}]W_2`$，一步记忆）；RNN head 通过 $`s_k`$ 额外压入更早的历史，但实验提升有限。
* **实验结论**：RNN head 只在更长 proposal 上比 Markov head 略好，但实现更复杂、部署性质更差，所以**默认用 Markov head**。

> **「一点点自回归就够了」（A Little Autoregression Goes a Long Way）** —— 这是全文最重要的工程结论。后缀依赖建模不需要堆深的串行网络，一个一阶 Markov 头就把并行 drafter 的 suffix decay 补回来了，代价小到可以忽略。

---

### 1.2 置信度调度的验证（Confidence-Scheduled Verification）

半自回归让 DSpark 能高效出大 block，但**「多 draft」不等于「多加速」**。盲目验证整个 block 在高并发下反而降吞吐——因为接受率有**数据方差**（code 高、chat 低）和**系统方差**（轻载验证近乎免费、重载抢占 batch 容量）。DSpark 用 **confidence head（预测前缀存活概率）+ hardware-aware scheduler（按实时负载定验证长度）** 把 target 算力只投给「期望正收益」的 token。

> **「系统效率」说人话就是一句：验证多几个字，「服务器闲」时几乎免费，「服务器忙」时却要命——所以忙的时候别去验那些「反正会被拒」的字。**
>
> 为什么忙闲不一样？回忆前面 memory-bound 那段：
> - **服务器闲**（用户少）：大模型一次前向顺手多验几个字，几乎不花额外的钱（就搬一次权重的事）。所以草稿蒙 5 个字、全验，无所谓。
> - **服务器忙**（几百个用户同时在线）：大模型一批要**同时伺候所有人**，而一批能塞下的字数是有上限的（显存、算力就那么多）。这时候你草稿里那几个「八成会被拒」的字，每挤进去一个，就从别的用户那儿**抢走一个名额**——你白验，别人还得排队等。
>
> 所以「提升系统效率」做的事很简单：**看服务器忙不忙，动态决定每个用户只验最有把握的前几个字**。闲时多验几个（反正免费），忙时砍到只验有把握的、把名额让给更多用户。它**一个字都不改草稿质量**（那是半自回归的活），只决定「这次验到第几个字为止」。也正因如此，它的价值只有在 V4 真实高并发流量里才显出来（离线评测时干脆把它关掉）。

#### 1.2.1 Confidence Head

每个 draft 位置输出一个标量 $`c_k \in (0,1)`$，**建模的是「前缀 $`1..k-1`$ 全被接受的条件下，位置 $`k`$ 的 token 通过验证的条件概率」**（注意是条件概率，不是边际）：

```math
c_k = \sigma\left(w^\top [h_k; W_1[x_{k-1}]]\right)
```

* $`h_k`$ 是骨干 hidden，$`W_1[x_{k-1}]`$ 是前一 draft token 的 Markov embedding（复用 §1.1.2 的 $`W_1`$）。一个轻量 linear + sigmoid。
* 监督信号用**解析的逐步接受率** $`c_k^*`$，由 draft 分布 $`p_k^d`$ 与 target 分布 $`p_k^t`$ 的 total variation 距离决定：

```math
c_k^* = 1 - \tfrac{1}{2} \|p_k^d - p_k^t\|_1
```

（这正是 rejection sampling 的逐步接受概率，Leviathan et al. 2023。）

#### 1.2.2 Post-hoc 校准：Sequential Temperature Scaling（STS）

* **为什么需要校准？** 阈值类启发式只需要置信度能**正确排序** token 质量；但 DSpark 的硬件感知调度要用**累积接受概率的绝对量级**去算期望接受长度 $`\tau`$。神经网络置信度普遍 **overconfident**，直接用 raw 分数会扭曲吞吐估计、导致调度次优。
* **STS 做法**：因为每个 $`c_i`$ 是条件概率，按链式法则，前缀被接受的联合概率 = 累积乘积 $`\prod_{i \le k} c_i`$。在 held-out 验证集上**从左到右逐位置标定**：对每个位置 $`k`$ 做一次 1D 网格搜索找最优温度标量，**最小化累积乘积的 ECE**（Expected Calibration Error），且固定住前面已标定的分数。
* **关键性质**：温度缩放是**保序变换**——只把预测概率校准到经验接受率，不打乱 confidence head 学到的相对排序。效果：ECE 从 3%–8% → ~1%，ROC-AUC 0.81–0.90。

> **通俗解读**：
>
> **为什么要校准**：confidence head 输出的是 0–1 之间的分数，但神经网络天然倾向于"自我感觉良好"——实际只有 60% 把握的 token，它可能报出 90%。如果调度器直接拿这个虚高的分数去估算"这次能验几个 token"，就会系统性地高估，导致验太多、浪费 target 算力。
>
> **温度缩放在做什么**：把原始分数 $`c_k`$ 除以一个温度 $`T_k > 1`$（再过 sigmoid），让过高的置信度往下压，使"模型说 70% 的地方，经验接受率也真的在 70% 附近"。这是一个纯后处理步骤，**不重新训练模型，只在验证集上搜一个最优的缩放系数**。
>
> **为什么要逐位置从左到右标定**：第 $k$ 个 token 被接受的前提是前 $k-1$ 个都被接受（前缀规则），所以联合概率是 $`c_1 \times c_2 \times \cdots \times c_k`$ 的连乘。越靠后的位置，误差累积越严重。STS 的做法是：先标定位置 1 的温度，固定住，再标定位置 2，依次往后——保证每一步校准时，前面的乘积已经是准的，不会相互干扰。
>
> 每个位置各搜一个独立的温度标量 $`T_k`$，校准公式为 $`c_k^{\text{cal}} = \sigma(\text{logit}(c_k) / T_k)`$（$`T_k > 1`$ 时往下压）。以 $`\gamma=3`$ 为例，搜索过程如下：
> ```
> 步骤1：搜 T₁
>   枚举 T₁ 的候选值（如 1.0, 1.2, 1.5, ...）
>   对验证集每条样本，看 c₁_cal 和「位置1实际是否被接受」的 ECE
>   找最小 ECE 对应的 T₁，固定住
>
> 步骤2：搜 T₂（T₁ 已固定）
>   枚举 T₂ 的候选值
>   对验证集每条样本，计算 c₁_cal × c₂_cal 和「前2个都被接受」的 ECE
>   找最小 ECE 对应的 T₂，固定住
>
> 步骤3：搜 T₃（T₁、T₂ 已固定）
>   枚举 T₃ 的候选值
>   计算 c₁_cal × c₂_cal × c₃_cal 和「前3个都被接受」的 ECE
>   找最小 ECE 对应的 T₃，固定住
> ```
> 最终得到 $`\gamma`$ 个常数 $`T_1, T_2, \dots, T_\gamma`$。**这个搜索只在部署前跑一次**，不是每次推理都搜——枚举候选值、算 ECE 这些事在离线阶段一次性完成，几秒钟跑完。推理时 $`T_k`$ 已经是固定常数，每个位置只做一次除法：$`c_k^{\text{cal}} = \sigma(\text{logit}(c_k) / T_k)`$，**推理路径上零额外开销**。
>
> **保序**：温度缩放只是整体往下压，不改变哪个 token 比哪个 token 更可信的相对顺序（排序不变），所以 confidence head 学到的"相对质量判断"完全保留。



#### 1.2.3 Hardware-Aware Prefix Scheduler

**核心问题**：同时有 $`R`$ 个用户请求在跑，每个请求的 draft 质量不同，该给每个请求验证几个 token？静态阈值不够用——它不知道服务器现在有多忙。

**直觉**：多验一个 token 有收益（期望多接受几个字），也有代价（batch 变大、引擎变慢）。调度器的活就是**找那个"再多验一个开始亏"的临界点**，而且要跨所有请求全局最优。

**关键量**：每个请求 $`r`$ 在位置 $`j`$ 的**存活概率** $`a_{r,j}`$——前 $`j`$ 个 draft token 全被接受的概率，等于逐位置置信度连乘：

```math
a_{r,j} = c_{r,1} \times c_{r,2} \times \cdots \times c_{r,j}
```

越靠后 $`a_{r,j}`$ 越小（必须前面都对才能活到这里）。多验第 $`j`$ 个 token 的**边际期望收益就是 $`a_{r,j}`$**。

**优化目标**：最大化整体吞吐 $`\Theta`$ = 期望接受 token 数 × 引擎每秒能跑几步：

```math
\Theta = \underbrace{\tau}_{\text{期望接受量}} \times \underbrace{\text{SPS}(B)}_{\text{引擎速度（batch越大越慢）}}
```

$`\text{SPS}(B)`$ 在引擎启动时 profile 一次存成 cost table，推理时 O(1) 查。

**贪心算法**（论文 Algorithm 1）：因为存活概率越靠后越低，最优策略天然是"先验最有把握的，不划算就停"：

```
① 算出所有请求、所有位置的存活概率 a_{r,j}
② 把这些候选 token 按 a_{r,j} 从高到低排成一个全局队列
③ 从队列头开始，逐个试着"多验这一个"：
     查表算新的吞吐 Θ = (τ + a_{r,j}) × SPS(B+1)
     吞吐还在涨 → 收下，继续
     吞吐掉头   → 停（后面的 a_{r,j} 只会更小，不可能再涨）
④ 返回各请求的最优验证长度
```

**为什么不能提前把所有 token 都算好再全局搜**：存活概率 $`a_{r,j}`$ 依赖前一个已采样 token 的 Markov 特征，如果提前看到后面的 token 再做第 $`j`$ 步的决策，等于"开卷考试"——引入了 selection bias，破坏无损保证。early-stop 的 break 天然隔离了未来 token，保证决策只依赖已有信息。

**前提**：吞吐 $`\Theta`$ 关于 batch size 需要单峰（SPS 曲线平滑衰减）才能保证 early-stop 给出全局最优。真实锯齿 SPS 和异步流水线的适配见 §4.3。

---

## 2. 训练

训练时从每条 target 序列**随机采多个 anchor 位置**组成 $`\gamma`$-token block 作训练数据。target model 全程冻结；draft model 复用并冻结 target 的 embedding 和 LM head，**只更新骨干 drafter、串行块、confidence head**。

三项损失，全部按 $`w_k = \exp(-(k-1)/\gamma)`$ 加权（强调靠前位置——前缀验证下靠前 token 对期望接受长度贡献更大）：

```math
\mathcal{L}_{\text{ce}} = -\sum_{k=1}^{\gamma} w_k \log p_k^d(x_k^*) \qquad\qquad \text{（交叉熵，预测正确 next token）}
```

```math
\mathcal{L}_{\text{tv}} = \sum_{k=1}^{\gamma} w_k \|p_k^d - p_k^t\|_1 \qquad\qquad \text{（分布匹配，TV 距离）}
```

```math
\mathcal{L}_{\text{conf}} = -\sum_{k=1}^{\gamma} w_k \left[c_k^* \log c_k + (1 - c_k^*)\log(1 - c_k)\right] \quad \text{（confidence BCE）}
```

* $`\mathcal{L}_{\text{tv}}`$ 直接最大化接受率：逐步接受概率 $`= 1 - \frac{1}{2}\|p^d - p^t\|_1`$，minimize TV 就是 maximize 接受率。
* $`\mathcal{L}_{\text{conf}}`$ 把 confidence head 训成预测 §1.2.1 的软接受标签 $`c_k^*`$。
* **关于 $`p_k^d`$ 的类型与来源**：三个 loss 里的 $`p_k^d`$ 都是**过了串行头之后的最终 draft 分布**，即 $`p_k^d(v) = \text{softmax}(U_k(v) + B_k(v))`$（骨干 logit $`U_k`$ 加 Markov/RNN head 的 bias $`B_k`$ 再 softmax），不是骨干的 raw logit。$`p_k^d`$ 本身是词表大小的概率向量（$`\in \mathbb{R}^{|V|}`$）；$`\mathcal{L}_{\text{ce}}`$ 里的 $`p_k^d(x_k^*)`$ 是取其中第 $`x_k^*`$ 维——即"正确 token 被分配到的概率"——是一个标量。$`\mathcal{L}_{\text{tv}}`$ 里的 $`\|p_k^d - p_k^t\|_1`$ 才是完整向量做差后求 L1。

总目标（默认权重 $`\alpha_{\text{ce}} = 0.1, \alpha_{\text{tv}} = 0.9, \alpha_{\text{conf}} = 1.0`$）：

```math
\mathcal{L} = \alpha_{\text{ce}} \mathcal{L}_{\text{ce}} + \alpha_{\text{tv}} \mathcal{L}_{\text{tv}} + \alpha_{\text{conf}} \mathcal{L}_{\text{conf}}
```

---

## 3. 离线实验

### 3.1 实验设置

* **Target models**：Qwen3-{4B, 8B, 14B}、Gemma4-12B（跨规模、跨模型族）。
* **对比 drafter**：DFlash（并行 SOTA）、Eagle3（自回归，基于 Training-Time Test）。**公平起见所有 drafter 用同一框架、同一数据重训**；Eagle3 的 TTT horizon 对齐 block size 7、用相同 target feature 层；层数 Eagle3 = 1、DSpark / DFlash = 5。默认 DSpark = Markov head 变体。
* **训练数据**：Open-PerfectBlend（1.3M 样本：chat 17.6% / math 39.4% / code 38.9% / IF 4.1%）。只用 prompt，response 由各 target model 重新生成。每个 drafter 训 10 epoch。data 生成和评测都用 **non-thinking 模式**。
* **指标**：每解码 round 的**接受长度 $`\tau`$**（含 bonus token），采样温度 1.0，chain-based drafting。**离线评测关掉 confidence scheduler**，所有 drafter 固定 block 长度，以隔离纯 draft 质量。

### 3.2 主结果（Table 1，接受长度 $`\tau`$）

| Target | Drafter | Math 均值 | Code 均值 | Chat 均值 |
| --- | --- | --- | --- | --- |
| **Qwen3-4B** | Eagle3 | 4.56 | 3.87 | 2.40 |
|  | DFlash | 4.80 | 4.44 | 2.95 |
|  | **DSpark** | **5.57** | **5.12** | **3.49** |
| **Qwen3-8B** | Eagle3 | 4.66 | 4.15 | 2.58 |
|  | DFlash | 4.77 | 4.46 | 2.97 |
|  | **DSpark** | **5.65** | **5.28** | **3.50** |
| **Qwen3-14B** | Eagle3 | 4.52 | 4.00 | 2.52 |
|  | DFlash | 4.74 | 4.45 | 2.92 |
|  | **DSpark** | **5.63** | **5.24** | **3.47** |
| **Gemma4-12B** | Eagle3 | 5.39 | 4.75 | 2.99 |
|  | DFlash | 4.90 | 4.35 | 2.80 |
|  | **DSpark** | **5.65** | **5.09** | **3.25** |

*（注：上表为论文 Table 1 各域内基准的均值汇总，原表给出 GSM8K / MATH / AIME25 / MBPP / HumanEval / LCB / MT-Bench / Alpaca / Arena-Hard 9 个细分基准。）*

> **这张表与 confidence head 无关**：$`\tau`$（接受长度）纯粹衡量 draft 质量，只取决于半自回归生成（并行骨干 + 串行头）的好坏。Confidence head 的作用是"决定每次验证几个"，影响的是系统吞吐（后面 Table 2/3 的延迟和 tokens/s），不影响 $`\tau`$ 本身。两个贡献完全解耦：
>
> | 贡献 | 解决的问题 | 衡量指标 |
> |---|---|---|
> | 半自回归（串行头） | draft 质量差、suffix decay | $`\tau`$（接受长度）↑ |
> | Confidence head + 调度器 | 高并发下盲目验证浪费算力 | 系统吞吐、端到端延迟 |

**结论**：

* **macro-average 接受长度**：DSpark 相对 Eagle3 在 Qwen3-4B/8B/14B 上提升 **30.9% / 26.7% / 30.0%**；相对 DFlash 提升 **16.3% / 18.4% / 18.3%**。Gemma4-12B 上同样稳定领先 → **跨模型族泛化**。
* **强烈的域效应**：结构化任务（math 5.57 / code 5.12）接受长度天然高于开放 chat（3.49）。这种数据可预测性的方差，正是「静态验证长度会在尾部 token 上浪费算力」的直接动因 → 引出 confidence scheduler。

### 3.3 分析一：为什么并行能赢过自回归？（§4.3.1）

反直觉现象：并行 DFlash 和半自回归 DSpark 的接受长度常常**超过**全自回归 Eagle3，这与「逐步自回归 > 并行」的常识相悖。论文用 **position-wise conditional acceptance**（逐位置条件接受率，分母只数「前 $`k-1`$ 个 token 全被接受」的实例，剥离前缀错误的惩罚）拆解：

```
逐位置条件接受率（Qwen3-4B，示意趋势）
位置:        1      2      3      4      5      6      7
Eagle3:    0.81 ──────────────── 平稳/上扬 ────────────► 0.74   (Chat: 0.53→0.74)
DFlash:    0.88 ──────────────── 快速衰减 ─────────────► 0.63   (Code: 0.87→0.78)
DSpark:    0.93 ──────────────── 高且平稳 ─────────────►        (继承高起点 + 抑制衰减)
```

* **位置 1 的容量优势**：第一个 token 只 condition 在 target context 上，差距纯来自**架构容量**——自回归因 $`O(\gamma)`$ 延迟只能用浅网络，$`O(1)`$ 并行 drafter 能用深网络。DFlash 起点明显高于 Eagle3（Math 0.88 vs 0.81，Chat 0.72 vs 0.53）。而**前缀存活机制让第一个 token 杠杆最大**（位置 1 一拒整个 block 作废），所以这个初始优势被不成比例放大。
* **后段的独立性局限**：位置 2–7，前面 token 锁定语义路径后后续本应更可预测；自回归 Eagle3 能利用这种条件确定性维持甚至抬升接受率，DFlash 却因 multi-modal collision 快速衰减。
* **DSpark 的解法**：用深并行骨干吃下位置 1 的高容量（Math 起点 0.93），用轻串行头抑制后段衰减 → **全 block 高且平稳**。

### 3.4 分析二：一点点自回归就够了（§4.3.2）

* **drafter 深度**：固定 block=7，DSpark 层数 1→5 性能单调上升，**1→2 层边际增益最陡**。**2 层 DSpark 全域超过 5 层 DFlash** → 注入局部自回归比单纯堆深并行层的「精度-参数」性价比高得多。
* **proposal 长度**：固定 5 层，draft 长度扫 {4,8,12,16}。DSpark 在每个长度都超 DFlash，且**差距随 $`\gamma`$ 增大稳定拉开**（纯并行 DFlash 越长 suffix decay 越严重，边际效用递减；DSpark 抑制衰减所以相对增益变大）。$`\gamma=7`$ 时 math/code/chat 提升 16%/15%/18%，$`\gamma=15`$ 时扩到 30%/26%/22%。
* **延迟开销**：batch=128 下 target 验证主导计算，串行块开销可忽略。draft 长度 4→16 只给整 round 延迟加 **0.2%–1.3%**，却换来最高 30% 接受长度提升。

### 3.5 分析三：验证要更聪明而非更长（§4.3.3）

* **静态阈值 sweep**（隔离验证 confidence head）：随阈值升高，整体接受率稳步上升（estimator 过滤掉终将被拒的 token）。**chat 上 pruning 最显著**（高熵分布让定长验证最低效）：接受率 45.7% → 95.7%；结构化任务 pruning 温和（Math 76.9%→92.5%，Code 67.6%→92.0%）。
* **从静态阈值到校准调度**：静态阈值忽略系统负载（轻载验证低置信 token 机会成本小、重载抢 batch 容量），故需 hardware-aware scheduler。最大化系统吞吐要求 confidence model **既判别力强又校准准**。reliability diagram 显示 raw 模型判别力强（ROC-AUC 0.81–0.90）但 overconfident（ECE 3%–8%），STS 校准后平均 ECE 降到 ~1%。

---

## 4. 线上部署（Real-World Deployment）

离线只证明了算法增益，真正部署到 V4 这种大模型旁边，训练和推理都有额外系统挑战。

### 4.1 部署配置

DSpark drafter 与 **DeepSeek-V4-Flash / V4-Pro preview 共部署**。并行骨干 = **3 个 MoE 层 + mHC（Manifold-Constrained Hyper-Connections）+ SWA(128)**，最大 block $`\gamma = 5`$，Markov 串行头。confidence head 与 draft model 端到端联合训练，再 STS 校准。

### 4.2 可扩展训练（§5.1）

训 draft 需要 target 的输出分布做监督，但全文档上下文里同时跑两个模型的显存和跨 worker 通信开销巨大。两个系统优化（在内部 HAI-LLM 框架里）：

* **Hidden state communication**：传 target 全词表 logits（$`V \approx 10^5`$）会成带宽瓶颈。改为**只缓存 LM head 之前的 hidden state，跨 worker 只通信 hidden state**，LM head 投影在 draft worker 上**只对采样到的 target 位置**本地执行 → 每 token 通信复杂度降到 $`O(d)`$（$`d`$ 是 hidden 维）。（思路与 MLA 缓存 latent、V4 OPD 只 cache last-layer hidden state 异曲同工。）
* **Anchor-bounded sequence packing**：为了把 draft 计算成本与 target 上下文长度**解耦**，从训练序列里采固定数量的 draft anchor，把这些孤立的预测 block 打包成 dense batch。用 **token-level attention indices**（而非标准 2D mask）管理打包，跨多条独立序列/anchor 维持精确 causal mask，避开 padding 开销。

### 4.3 调度器落地（§5.2）：Algorithm 1 的两个现实冲突

Algorithm 1 理论无损，但直接上生产撞两堵墙：

1. **SPS 不是光滑单峰，而是锯齿离散**（step-wise 退化）→ early-stop 会被锯齿 cliff 困在局部最优。
2. **需要每步动态调度 draft token 数**，与连续 CUDA graph replay 和 **ZOS（Zero-Overhead Scheduling）冲突**——ZOS 要求下一步 batch size 在当前步完成前就已知，同步调度会 stall GPU 流水线。

**解法：异步调度（asynchronous）**

```
同步（理论，会 stall）：  当前步置信度 ──► 算容量K ──► 等结果 ──► 下一步   ✗ GPU 空转
                          └─────────── 串行依赖 ───────────┘

异步（落地）：           step t-2 的置信度 ──预估──► 容量上限 K（动态 top-K 截断长度）
                        step t   的候选 token ──按最新累积置信度严格排序──► admit
                          └ 历史预测只决定"截多长"，当前 token 只决定"排序" ┘
                          └────────── 二者解耦 = 天然 causal barrier ──────┘
```

* 用**前两步**的 confidence head 输出近似下一步验证容量 $`K`$，当前步候选仍按**最新累积置信度严格排序**，历史预测只用来定动态截断长度 → admission 变成**动态 top-$`K`$ 选择**。引入轻微时间偏移，但选择机制保序（最自信 token 永远优先验证），调度延迟被完全隐藏，无缝接 ZOS。
* **意外好处**：异步天然形成 causal barrier（admission 只依赖前两步信息、与当前 token $`x_{r,k}`$ 的实现隔离），所以可以**去掉 early-stop break、做无约束全局搜索**，跨过锯齿 SPS cliff 榨满物理吞吐，同时仍严格保持 target 分布。

> **通俗解读**：
>
> **问题1（GPU 空转）**：同步调度的困境是——"验几个"这个决定依赖当前步的 confidence 分数，但当前步还没算完，GPU 只能等。等的这段时间 GPU 白白空转。
>
> **解法**：把"验几个"和"验哪些"这两件事拆开：
> - **验几个（K）**：用**两步前**的 confidence 分数来估，在当前步开始前就算好，GPU 不用等
> - **验哪些**：用**当前步**的 confidence 分数排序，最有把握的先验
>
> 信息稍微滞后了两步，但 confidence 分数不会突变，近似误差极小，而 GPU 流水线完全不间断。
>
> **问题2（锯齿曲线困局）**：理论上的 early-stop 要在"当前步置信度突然下跌"时立刻刹车，但实际的 SPS 曲线是锯齿状的，一刹车就卡在局部低谷出不来。
>
> **意外收获**：异步方案里，"验几个"已经和当前 token 的实际内容解耦（用的是两步前的历史），所以 admission 决策天然与当前 token 无关。这意味着可以直接做**全局无约束搜索**——不用 early-stop，不怕锯齿，直接找全局最优的验证长度，反而比同步方案搜得更好。

### 4.4 高吞吐低延迟推理（§5.3）

* **部署 regime**：受 KV-cache 容量、可用流量（如 RL 长尾负载）限制，有效 batch 常远低于 GPU 算力饱和阈值。这个 regime 下传统的「单用户延迟 vs 总吞吐」权衡简化了——固定并发上限时，**最大化每 GPU token 吞吐 与 最大化单用户生成速度（tok/s/user）高度正相关**而非互斥。
* **变长 query 执行**：动态路由要求一个 batch 内支持变长 query，标准 decode kernel 为定长优化，naive 处理变长验证前缀会因 padding 和负载不均严重 underutilize GPU。**解法：物理执行与逻辑序列追踪解耦**——所有 request 的 token flatten 成独立元素同等处理，序列内依赖**靠 sparse attention 里的 marker tensor 传递**。具体到 V4 架构，**只需改 index-attention 和 compress 两个 kernel** 就能支持变长路由，调度器零额外底层开销。

### 4.5 线上效果（§5.4）

对比基线 **MTP-1**（单 token draft，V4-preview 发布两周后被 DSpark 取代的旧线上配置）。MTP-1 历史上一直用单 token，是因为静态多 token drafter（MTP-3/5）在高并发下因验证开销过大**严格降总吞吐**——所以 DSpark 直接证明了它能在动态服务里**安全解锁大 block 的潜力**。

| 部署 | 中等 SLA | DSpark 吞吐增益 | 严格 SLA | 现象 |
| --- | --- | --- | --- | --- |
| **V4-Flash** | 80 tok/s/user | **+51% throughput** | 120 tok/s/user | MTP-1 触底，DSpark 名义 +661%（解读为「解锁原本不可达的交互档」而非真实倍率） |
| **V4-Pro** | 35 tok/s/user | **+52% throughput** | 50 tok/s/user | MTP-1 进入低并发 regime，DSpark 名义 +406% |

* **匹配吞吐下的单用户加速**：V4-Flash **60%–85%**，V4-Pro **57%–78%**。
* **Pareto frontier 外推**：DSpark 在中等 SLA 提吞吐、在严格 SLA 下保住非退化的服务容量（跨过 MTP-1 的「性能悬崖」），把 throughput–interactivity 前沿整体往外推。
* **验证预算自适应负载**（Figure 8 机制）：
    * 中等并发（Flash <200、Pro <150 并发）→ 调度器利用空闲 target 算力，把验证预算从 MTP-1 静态 2 token **扩到 ~4–6 token**，每次 forward 接受更多 token。
    * 并发升高、target 容量饱和 → 调度器**平滑收缩**验证长度，低置信 token 在抢占关键 batch 容量前被剪掉。→ **轻载榨干空闲算力、重载守住 batch 容量**。

### 4.6 局限

prefix scheduler 把 target 侧验证浪费压到最低，但 **draft 侧用并行骨干生成首个 $`\gamma`$-block 的开销是固定且不可回收的**。对天生低接受率的复杂 query，这部分 upfront drafting compute 白花。未来方向：draft model 内做 **difficulty-aware early exiting**，让这类请求跳过完整 block 生成。

---

## 5. 与相关工作 / DeepSeek 体系的关系

* **vs DFlash**：DSpark 的并行骨干直接用 DFlash，唯一改 anchor 当首预测位；核心增量是**串行头（补依赖）+ confidence 调度（省验证）**。Concurrent 工作里 Domino 的 CausalEncoder 概念上类似 DSpark 的 RNN head，DFlare 用 layer-wise fusion 解 conditioning 瓶颈。
* **vs CRF-NAT / CTC-drafter**：这些也在并行 hidden 上叠串行模块，但 CRF 的全局归一化配分函数、CTC 的对齐路径 latent 边际化，都**算不出精确 per-token 概率**，无法满足 rejection sampling。DSpark 把串行修正保持**局部**（causal 因子分解），per-token 概率仍是精确 softmax → **可严格无损验证**，这是它能上生产的根本原因。
* **vs DeepSeek-V4 主报告**：DSpark 是 V4 serving 栈的一块（drafter 复用 V4 的 mHC / SWA / sparse attention，调度器适配 V4 的 index-attention / compress kernel）。V4 主报告讲模型架构与训练（Hybrid Attention、mHC、Muon、FP4、OPD），DSpark 讲**推理加速这一层**，二者配套——可对照阅读 📄 *DeepSeek V4 解读*（`HaiyangReading/TechReport/DeepSeekV4.md`）。
* **开源**：DeepSpec 训练库（含 Eagle3 / DFlash / DSpark）+ DSpark checkpoints（V4-Flash preview / V4-Pro preview）。
