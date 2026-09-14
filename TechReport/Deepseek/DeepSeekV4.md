# DeepSeek V4 解读

---

## Take Home Message

> **一句话总结**：DeepSeek V4 是一份在「**长上下文 × 低成本 × 大规模 MoE**」三个方向同时压上极致工程的技术报告——**原生** 1M 上下文、**FP4 + FP8 混合精度**训练、**1.6T 总参 / 49B 激活**的 Pro 版本，是当前开源体系里最完整的一次系统性升级。

* **架构四大核心改动**：
    * **Hybrid Attention（CSA + HCA 交错）**：CSA 先把 KV 沿序列方向 $`m=4`$ 压缩，再用 Lightning Indexer 在压缩候选集中做 top-K 稀疏选择；HCA 用 $`m'=128`$ 极高压缩率直接 dense MQA。**1M 上下文下 KV cache 与 FLOPs 都能压下来的核心技术**，相对 V3.2 的 DSA 是结构级重构。
    * **mHC（Manifold-Constrained Hyper-Connections）**：残差矩阵 $`B_l`$ 约束到双随机矩阵流形（Birkhoff polytope），同时解决标准残差表达力受限与原版 Hyper-Connections 数值不稳定。
    * **MoE Gate 改造**：激活换成 $`\sqrt{\text{softplus}(\cdot)}`$，前 3 个 MoE 层用 **Hash routing**（按 token ID 定专家）。
    * **Muon 优化器**：10 步 hybrid Newton-Schulz 正交化降低优化器开销，Embedding / Head / mHC bias / RMSNorm 仍走 AdamW。
* **精度工程**：MoE expert 权重 + CSA indexer QK 路径**全程 FP4**，其余 FP8，是开源体系目前最激进的一次精度落地，工程价值与架构改动并列。
* **Pre-training**：32T / 33T tokens 训出 Flash / Pro，**原生 1M 上下文，不依赖 YaRN**。三个关键 trick：Sparse Attention 两阶段冷启动（dense → indexer warmup → full sparse）、Anti-Spike（Anticipatory Routing + SwiGLU Clamping）、Muon update RMS rescale。
* **Post-training**：用 **On-Policy Distillation (OPD)** 替换 V3.2 的 Mixed RL，走 **Specialist Training + OPD Merged** 两阶段。三个独立贡献：
    * **Specialist Training**：math / code / agent / IF 各自独立 SFT + RL (GRPO) 出 expert；**Non-think / Think High / Think Max 三档思考模式 = 三个不同 RL 配置的 checkpoint，不是开关 / 提示词**。
    * **Self-Generative Reward Model (GRM)**：actor 自己当 GRM，生成与评判联合优化，省掉独立 scalar RM 训练，专攻难验证任务。
    * **Full-Vocabulary OPD**：对齐 teacher **完整 logit 分布**（非 token-level estimate），gradient variance 更低；工程上**只 cache teacher last-layer hidden state**、训练时临时重建 logits，扛住 10+ teachers 显存压力。
* **Post-training Infra**：**Million-Token Context RL**（rollout 拆 metadata + per-token field，token 粒度 WAL 防 length bias）+ **DSec 沙箱平台**（Rust 实现，Function Call / Container / microVM / fullVM）是 agentic RL 能 scale 的核心基础设施。
* **开源规格**：Flash 284B / 13B 激活，Pro 1.6T / 49B 激活；Base 走 FP8 Mixed，正式版走 FP4 + FP8 Mixed。
* **能力侧亮点**：Agentic Search vs RAG 胜率 **61.7% vs 18.3%**；中文白领任务 vs Opus-4.6-Max **53% win / 37% lose**；Code Agent 内部 30 题 **67% pass rate**（>Sonnet 4.5，≈Opus 4.5，<Opus 4.6 Thinking 80%）。
* **短板**：Instruction Following 偶尔忽略 formatting、长文本→精炼摘要能力欠缺、Code Agent 存在 trivial mistakes / 模糊 prompt 误解 / over-thinking。

---

## 架构核心改进

* **混合稀疏注意力**：CSA (Compressed Sparse Attention) + HCA (Heavily Compressed Attention) 交错分布（除了前两层连续 HCA）。CSA 压缩率设置为 4 并且使用 Lightning Indexer 做 top-K 稀疏 KV 选择。HCA 设置高压缩率 128、不做 KV 稀疏直接 Dense MQA。**这是 DS V4 实现 1M 上下文下低 FLOPs / KV cache 的核心技术。**
* **mHC 残差**：把 Hyper-Connections 的残差矩阵 $`B_l`$ 约束到双随机矩阵流形（Birkhoff polytope），相比原版 Residual Connection、Hyper-Connections，解决了表达能力受限、数值不稳定的问题。
* **MoE 侧改进**：Gate 激活从 sigmoid 改成 $`\sqrt{\text{softplus}(\cdot)}`$、前 3 个 MoE 层用 **Hash routing**（按 token ID 决定专家）。
* **使用 Muon 优化器**：使用 Muon 10 步 hybrid Newton-Schulz 正交化降低优化器开销，Embedding / Head / mHC static bias / RMSNorm 维持 AdamW。MoE expert 权重全程 FP4，CSA indexer 的 QK 路径也 FP4，其余部分保持 FP8。

目前 DeepSeek V4 开源的 4 个 Checkpoint 参数 / 激活量 / 上下文长度 / 精度参考：

| Model | #Total Params | #Activated Params | Context Length | Precision |
| --- | --- | --- | --- | --- |
| **DeepSeek-V4-Flash-Base** | 284B | 13B | 1M | FP8 Mixed |
| **DeepSeek-V4-Flash** | 284B | 13B | 1M | FP4 + FP8 Mixed\* |
| **DeepSeek-V4-Pro-Base** | 1.6T | 49B | 1M | FP8 Mixed |
| **DeepSeek-V4-Pro** | 1.6T | 49B | 1M | FP4 + FP8 Mixed\* |

> \* *FP4 + FP8 Mixed: MoE expert parameters use FP4 precision; most other parameters use FP8.*

---

## 1. 模型架构

DeepSeek V4 相比 DeepSeek V3 / V3.2 的改进集中在残差机制、注意力层结构、MoE gate 激活函数、优化器四个维度：

| 维度 | V3 / V3.2 | V4 |
| --- | --- | --- |
| **FFN** | DeepSeekMoE (fine-grained routed + shared) | **Gate 改为 $`\text{Sqrt}(\text{Softplus}(\cdot))`$ + 前 3 层 Hash routing** |
| **Attention** | MLA (V3) / MLA + DSA Lightning Indexer (V3.2) | Hybrid Attention：**CSA + HCA 交错** |
| **Residual** | 标准 Residual | **mHC：Manifold-Constrained Hyper-Connections** |
| **Optimizer** | AdamW | **Muon 优化器**，embedding / head / mHC static bias / RMSNorm 保留 AdamW |

Pre-training 这次直接用 32T / 33T token 语料训练 V4-Flash / V4-Pro，**原生支持 1M 上下文**（DS V4 最重要的点，相比 V3 系列利用 YaRN 进行的上下文拓展）。

---

### 1.1 混合注意力（Hybrid Attention）：CSA + HCA 的长文本方案

对原生 KV 做两种不同粒度的压缩，把每层注意力做成 SWA + 压缩 KV 的两路加和；压缩后若 KV 数目仍大（$`m=4`$ 压缩率下是 $`n/4`$），用 Lightning Indexer 做 top-k 稀疏选择；压缩率极高（$`m'=128`$）时直接 dense MQA。

#### 1.1.1 CSA（Compressed Sparse Attention）：压缩 + 稀疏选择

* **动机**：V3.2 的 DSA 直接在原始 KV 上用 Lightning Indexer 做 top-K Selection，每个 query 的 KV 候选大小仍是 $`O(n)`$。V4 先进行 KV Sparse 压缩到 $`O(n/m)`$ 再进行 Selection，因此 indexer 排序候选项集变为之前的 $`1/m`$。
* **实现**：分两步实现，先 KV Compress 把 $`O(n)`$ 序列长度压缩到 $`O(n/m)`$，后 Lightning Indexer 进行 Top-K 选择。

##### KV Compress

输入 $`H \in \mathbb{R}^{n \times d}`$，首先用两个可学习的 KV projection $`W^{aKV}, W^{bKV} \in \mathbb{R}^{d \times c}`$ 映射到 Latent（这里和 MLA 的逻辑是一样的），但是需要注意这里映射了 2 个 Latent 表示是为了压缩的时候进行窗口平滑（Sliding Window）：

```math
C^a = H \cdot W^{aKV}, \quad C^b = H \cdot W^{bKV}
```

此外，需要对应的另外两个可学习的 KV projection $`W^{aZ}, W^{bZ} \in \mathbb{R}^{d \times c}`$ 映射另外一组 Gate Latent，映射维度和上面一样，用来计算压缩时的加权率（Gating）：

```math
Z^a = H \cdot W^{aZ}, \quad Z^b = H \cdot W^{bZ}
```

然后 $`m`$ 个连续 token 计算两个压缩加权比率（论文里面取 $`m = 4`$）：

```math
[S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}] = \text{Softmax}_{\text{row}} \left( [Z^a_{mi:m(i+1)-1} + B^a; Z^b_{m(i-1):mi-1} + B^b] \right)
```

用上面的加权比率就可以将 $`m`$ 个连续 token 合成一个 compressed entry $`C^{\text{Comp}}_i`$，注意 $`C^a`$ 用于当前窗口 $`[mi, m(i+1)-1]`$，$`C^b`$ 用于前一个窗口 $`[m(i-1), mi-1]`$，两个窗口 overlap 一半，**这是 CSA 平滑的关键**：

```math
C^{\text{Comp}}_i = \sum_{j=mi}^{m(i+1)-1} S^a_j \odot C^a_j + \sum_{j=m(i-1)}^{mi-1} S^b_j \odot C^b_j
```

* $`B^a, B^b \in \mathbb{R}^{m \times c}`$：可学习的位置偏置（**相当于滑动窗口的 Absolute Position Embedding**）。
* $`\odot`$：Hadamard 积，即对应位置相乘，结果 shape 不变。

> **为什么要分两个投影权重，一个不行吗？**
>
> 单个投影 + 重叠窗口其实也能"平滑"——只要相邻 compressed entry 共享一部分 token，边界就不再被硬切开。所以平滑本身不是两个权重的真正理由。两个权重的价值在于：**同一段 token 在相邻两个 compressed entry 里扮演的位置角色是不同的**。
>
> 考虑某一段窗口 token $`[mk, m(k+1)-1]`$，它会被用到两次：
> - 作为 entry $`k`$ 的「当前窗口」（后半段），走 $`C^a / W^{aKV}`$；
> - 作为 entry $`k+1`$ 的「前一窗口」（前半段），走 $`C^b / W^{bKV}`$。
>
> 如果两次都用同一个 $`W^{KV}`$，那么这个 token 喂给两个相邻 entry 的 latent **是同一个向量**（只差 gate 的缩放和位置 bias）。结果是两个重叠 entry 高度相关、内容冗余，在固定 latent 维度 $`c`$ 的压缩预算下白白浪费容量。换成两个不同投影后，token 可以根据「我在这个 entry 里是前半还是后半」呈现**不同方向的表示**，使一对重叠 entry 携带的是**互补**而非重复的信息——同样的压缩率下保留更多原始信息。
>
> 一个类比：这相当于 overlap-add 加窗变换里，给前后两个重叠段各配一套（可学习的）分析窗，而不是共用一个窗。共用一个窗能平滑，但两套窗能在平滑的同时让每段各自编码最优的内容。位置 bias $`B^a, B^b`$ 本身就取了两组，也正是为了配合这种「前半/后半」角色的区分。

对应到代码里 `Compressor.forward`：

```python
# 记号：B=batch, n=seq_len, m=ratio (CSA=4 / HCA=128), d=head_dim
kv = self.wkv(x)               # [B, n, 2d]   C^a / C^b stacked, wkv 输出 coff * head_dim 即 2d
score = self.wgate(x)          # [B, n, 2d]   Z^a / Z^b stacked
# ... (kv 也 unflatten 到 [B, n/m, m, 2d])
score = score.unflatten(1, (-1, ratio)) + self.ape  # [B, n/m, m, 2d] + [m, 2d] (broadcast) = [B, n/m, m, 2d]，+ B^a / B^b
if overlap:
    kv = self.overlap_transform(kv, 0)               # [B, n/m, 2m, d]  错位 concat 前后窗口（a/b 拆开沿 m 轴拼成 2m）
    score = self.overlap_transform(score, float("-inf"))  # [B, n/m, 2m, d]  pad 处填 -inf，softmax 后等价于 0
kv = (kv * score.softmax(dim=2)).sum(dim=2)          # softmax over 2m → [B, n/m, d]，即 C_i^Comp
```

* $`W^{aKV}; W^{bKV}`$ 对应 `self.wkv`（`Linear(dim, 2 * head_dim)`）
* $`W^{aZ}; W^{bZ}`$ 对应 `self.wgate`
* $`B^a; B^b`$ 对应 `self.ape`（`Parameter(ratio, 2 * head_dim)`）
* `overlap_transform` 对应 $`C^a_{\text{当前}}`$、$`C^b_{\text{前}}`$ 的错位 Concat

压缩得到 $`C^{\text{Comp}} \in \mathbb{R}^{n/m \times c}`$，**序列长度是原本的 $`1/m`$**。

##### Lightning Indexer

这里的 Indexer 风格直接参考的 DeepSeek V3.2（可以参考 [GLM-5 完整技术报告解读] 里面的详细代码解析）。

对 query token $`t`$，先生成低秩 Latent 再生成 indexer queries（本质上仍然是 MLA）：

```math
\mathbf{c}_t^Q = \mathbf{h}_t W^{DQ} \quad \text{（注意这里是行向量右乘风格，本质上和 [GLM-5 完整技术报告解读] 的列向量左乘风格一样）}
```

*shape：$`\mathbf{h}_t`$ `[d]` $`\times`$ $`W^{DQ}`$ `[d, d_c^Q]` $`\to`$ $`\mathbf{c}_t^Q`$ `[d_c^Q]`（把 hidden 维 $`d`$ 压到 latent 维 $`d_c^Q`$）*

```math
[\mathbf{q}_{t,1}^I; \dots; \mathbf{q}_{t,n_h^I}^I] = \mathbf{q}_t^I = \mathbf{c}_t^Q W^{IUQ}
```

*shape：$`\mathbf{c}_t^Q`$ `[d_c^Q]` $`\times`$ $`W^{IUQ}`$ `[d_c^Q, n_h^I × d_h^I]` $`\to`$ $`\mathbf{q}_t^I`$ `[n_h^I × d_h^I]`，再切成 $`n_h^I`$ 个 `[d_h^I]` 的 head query*

同时需要生成 query head 之间的加权权重（$`n_h^I`$ 表示 Indexer 的 query 的 head 数量）：

```math
[w_{t,1}^I; w_{t,2}^I; \dots; w_{t,n_h^I}^I] = \mathbf{w}_t^I = \mathbf{h}_t \cdot W^w
```

*shape：$`\mathbf{h}_t`$ `[d]` $`\times`$ $`W^w`$ `[d, n_h^I]` $`\to`$ $`\mathbf{w}_t^I`$ `[n_h^I]`，即每个 head 一个标量权重*

然后应用 ReLU 风格的 index score（需要在多 head 之间求和）：

```math
I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \text{ReLU}(\mathbf{q}_{t,h}^I \cdot K_s^{\text{IComp}})
```

*shape：$`\mathbf{q}_{t,h}^I`$ `[d_h^I]` $`\cdot`$ $`K_s^{\text{IComp}}`$ `[d_h^I]` $`\to`$ 标量（两向量点积，维度必须相同），`ReLU` 和乘 $`w_{t,h}^I`$ 仍是标量，再 $`\sum_h`$ → $`I_{t,s}`$ 是**一个标量**（query $`t`$ 对 entry $`s`$ 的一个分）。一次对全部 $`S`$ 个候选 entry 算就升一维：$`\mathbf{q}_t^I`$ `[n_h^I, d_h^I]` $`\times`$ $`K^{\text{IComp}}`$ `[S, d_h^I]` → 每 head 每 entry 点积 `[n_h^I, S]` → ReLU、乘权重、按 head 求和 → $`I_{t,\cdot}`$ `[S]`，对应代码 `einsum("bshd,btd->bsht")`*

> **$`K_s^{\text{IComp}}`$ 这个 key 从哪来？** 它不是上面 CSA 主路那套 $`C^{\text{Comp}}`$，而是 **Indexer 自带一个独立的 Compressor**（原文代码里的 `self.compressor`）把输入按压缩率 $`m`$ 压出来的 indexer 专属 key，写进 `self.kv_cache`。上标 `IComp` = **I**ndexer **Comp**ressed。它和 query 一样全程走 FP4，所以打分这一步又轻又快。注意：Indexer 用的是它自己压出来的 key，与后面真 attention 用的 KV 是两套东西。

> **用大白话理解这一段**
>
> Indexer 是个**便宜的粗筛器**：给每个历史 entry 打个分 $`I_{t,s}`$，告诉真 attention「该看谁」，再把昂贵的 full attention 只算在 top-k 上——这就是 DSA 省算力的核心。
>
> 从 $`\mathbf{h}_t`$ 出发，兵分两路备料，最后汇合打分：
>
> - **A 路（备 query）**：`h_t → 压成 latent → 升维 → query q^I`
> - **B 路（备权重）**：`h_t → 每个 head 的权重 w^I`
> - **汇合打分**：`I_{t,s} = 按 head 求和( w^I · ReLU(q^I · key) ) → 选 top-k`
>
> 每个 $`W`$ 都只是「乘个矩阵、把向量换成另一种表示」的线性层。所谓「压小/撑大」，就是用不同 **shape**（张量的形状，即每个维度多长）的矩阵改变向量长度——例如 $`\mathbf{h}_t W^{DQ}`$ 就是 `[7168] × [7168, 1536] → [1536]`（相邻的 7168 必须相等才能乘）。前三个公式只是**备料**——A 路备好 query、B 路备好权重，等第四个公式汇合打分：
>
> - **① $`\mathbf{c}_t^Q`$（压小）**：把又长又贵的 $`\mathbf{h}_t`$ 降维成小 latent，shape `[7168] → [1536]`。
> - **② $`\mathbf{q}_t^I`$（撑大）**：再把 latent 升维成 $`n_h^I`$ 个 head 的 query，shape `[1536] → [n_h × head_dim]`。先压后升是为了**省参数**（两个瘦矩阵 ≈ 一个胖矩阵，但便宜得多），这就是 MLA 套路。
> - **③ $`\mathbf{w}_t^I`$（权重）**：直接从 $`\mathbf{h}_t`$ 算出每个 head 一个标量，决定打分时「谁的意见更重要」。它只输出几个数，太轻了不值得走「先压后升」。
> - **④ 打分**：query 和 key 点积 → `ReLU` 只留正相关 → 按 head 权重 $`w`$ 求和。
>
> 类比：先扫书脊（便宜、扫全部）挑出几本，再抽出来精读（贵、只读 top-k）。扫书脊之所以便宜，是因为它 **head 少、QK 走 FP4、还和主 attention 共享 query latent**。

Top-k 选择：

```math
C_t^{\text{SprsComp}} = \{ C_s^{\text{Comp}} \mid I_{t,s} \in \text{Top-}k(I_{t,\cdot}) \}
```

对应到代码里 `Indexer.forward`：

```python
q = self.wq_b(qr)                        # c^Q_t * W^IUQ
apply_rotary_emb(q[..., -rd:], ...)
q = rotate_activation(q)                # Hadamard rotation 为 FP4 做准备
fp4_act_quant(q, fp4_block_size, True)  # q 走 FP4
self.compressor(x, start_pos)           # 压缩 indexer keys, 内部处理 k 到 FP4
weights = self.weights_proj(x) * self.softmax_scale  # w^I_{t,h}
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[...])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)  # I_{t,s}
topk_idxs = index_score.topk(min(self.index_topk, ...))[1]
```

* $`W^{IUQ}`$ 对应 `self.wq_b`（`ColumnParallelLinear(q_lora_rank, index_n_heads * index_head_dim)`）
* $`W^w`$ 对应 `self.weights_proj`
* Indexer 的 compressed KV 对应 `self.compressor`，写入 `self.kv_cache[...]`
* $`\text{ReLU}(\mathbf{q} \cdot K)`$ 对应 `index_score.relu_()`
* **Indexer 的 QK 全走 FP4**：`fp4_act_quant(q, ...)`，减少 indexer 本身的 memory traffic 和算力。
* **top-K 规模比 V3.2 小**：V4-Pro 的 `index_topk=512`。
* **query 压缩 $`\mathbf{c}_t^Q`$ 是 indexer 和 attention 共享的**：省去一次 down-proj 计算。

##### Shared KV MQA

选出 top-k compressed entries 之后做 MQA：

```math
[\mathbf{q}_{t,1}; \dots; \mathbf{q}_{t,n_h}] = \mathbf{q}_t = \mathbf{c}_t^Q W^{UQ}
```

```math
\mathbf{o}_{t,i} = \text{CoreAttn}(\text{query} = \mathbf{q}_{t,i},\; \text{key} = C_t^{\text{SprsComp}},\; \text{value} = C_t^{\text{SprsComp}})
```

最后把多个 head 的 $`\mathbf{o}_{t,i}`$ 拼起来得到 $`\mathbf{o}_t = [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \dots; \mathbf{o}_{t,n_h}] \in \mathbb{R}^{c n_h}`$。

##### Grouped Output Projection

* **动机**：上一步合并的 $`\mathbf{o}_t`$ 的 hidden dim $`c n_h`$ 非常大，如果直接进行 $`W^o`$ 投影，投影矩阵是 $`c n_h \times d`$，参数量和计算量都很夸张。
* **实现**：先分组降维，再合并投影。
* **分组**：把 $`n_h`$ 个头平均分成 $`g`$ 组，每组有 $`n_h / g`$ 个头。每组拼起来的向量是 $`\mathbf{o}_t^{G_i} \in \mathbb{R}^{c \cdot n_h / g}`$。
* **组内降维**：每一组各自用一个小的投影矩阵，把 $`c \cdot n_h / g`$ 维压到 $`d_g`$ 维：$`\mathbf{o}_t^{G_i'} = \mathbf{o}_t^{G_i} W_i^{o\_a} \in \mathbb{R}^{d_g}`$。
* **合并后再投影到 $`d`$**：把 $`g`$ 个降维后的小向量拼起来（总共 $`d_g \cdot g`$ 维），再用一个投影矩阵送到最终的 $`d`$ 维：$`\hat{\mathbf{o}}_t = [\mathbf{o}_t^{G_1'}; \dots; \mathbf{o}_t^{G_g'}] \, W^{o\_b} \in \mathbb{R}^d`$。

对应到代码里 `Attention.__init__` 和 `forward`：

```python
self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups,
                                 self.n_groups * args.o_lora_rank,
                                 dtype=torch.bfloat16)
self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
# ...
o = o.view(bsz, seqlen, self.n_local_groups, -1)
wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
o = torch.einsum("bsgd,grd->bsgr", o, wo_a)  # 每组独立降维
x = self.wo_b(o.flatten(2))                  # 再拼起来投到 dim
```

* $`W_i^{o\_a}`$ 对应 `self.wo_a.weight.view(n_groups, o_lora_rank, -1)` 按组切片
* $`W^{o\_b}`$ 对应 `self.wo_b`

---

#### 1.1.2 HCA（Heavily Compressed Attention）：极端压缩 + dense MQA

##### KV Compress 与 CSA 共用实现

* **动机**：CSA 设置 $`m = 4`$ 的 KV 压缩率不够，在极长文本（如目标的 1M 上下文）下仍有巨大开销。DeepSeek V4 希望有一层**全局的、非稀疏 KV 通道**，补充 CSA 丢掉的全局信号。
* **实现**：HCA（Heavily Compressed Attention）设置更激进的 $`m' = 128`$，不做稀疏选择，直接 dense MQA。

由于没有 CSA 的滑动窗口策略（overlap），因此只有一路 KV：

```math
C = H \cdot W^{KV}, \quad Z = H \cdot W^Z
```

计算压缩重要性权重：

```math
S_{m'i:m'(i+1)-1} = \text{Softmax}_{\text{row}} (Z_{m'i:m'(i+1)-1} + B)
```

加权获得压缩 KV：

```math
C^{\text{Comp}}_i = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j
```

对应到代码里直接复用 `Compressor` 类，设置 `compress_ratio=128`、`overlap=False`：

```python
class Compressor:
    def __init__(self, ..., compress_ratio=4):
        ...
        self.overlap = compress_ratio == 4  # 只有 ratio=4 时才 overlap
```

##### Shared KV MQA / Grouped Output Projection 与 CSA 共用实现

对应到代码里 `Attention` 类（`compress_ratio != 4` 时 `self.indexer = None`）：

```python
if self.compress_ratio:
    self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
    if self.compress_ratio == 4:
        self.indexer = Indexer(args, self.compress_ratio)
    else:
        self.indexer = None
```

Grouped Output Projection 参考 CSA 实现。

---

#### 1.1.3 CSA / HCA 排布设置（Interleaved 布层）

可以注意到 V4-Pro 的 `config.json` 文件内有一个 `compress_ratios` 参数，是一个长度 62 的列表。这个就是决定每一层是 CSA / HCA 的关键参数（61 层 + 1 MTP 层）：

```json
"compress_ratios": [128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, ..., 4, 128, 4, 0]
```

* **Layer 0-1**：HCA only（$`m = 128`$）。最底层（非抽象特征）需要全局粗粒度视野，不需要 Indexer。
* **Layer 2-60**：CSA 和 HCA 交替排列。
* **Layer 61 (MTP block)**：$`m = 0`$，退化为纯 sliding window attention。

**直观理解 CSA / HCA 排布设置**：底层保持全局视野，之后交替 fine-grained 局部细节（CSA 的 top-k 稀疏 + 4x 压缩）和 coarse 全局汇总（HCA 的 128x dense）。

---

#### 1.1.4 Hybrid Attention 中的其他重要 Tricks

##### Query / KV RMSNorm

每个 head 的 Q、共享的 KV 都单独做 RMSNorm，目的是防止 attention logits 爆炸：

```python
qr = q = self.q_norm(self.wq_a(x))  # latent RMSNorm
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)  # 真正的 q norm
# ...
kv = self.kv_norm(kv)  # kv norm
```

##### Partial RoPE

只对 last 64 dims（`rope_head_dim=64`）做 RoPE，其余 448 维走 FP8 / FP4 量化：

```python
apply_rotary_emb(q[..., -rd:], freqs_cis)
apply_rotary_emb(kv[..., -rd:], freqs_cis)
# rope 维度保留 BF16，非 rope 维度走 FP8 simulate
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
```

##### Attention Sink

给每个 head 加一个可学习 logit $`z_{h}^{\prime}`$ 到 softmax 分母：

```math
s_{h,i,j} = \frac{\exp(z_{h,i,j})}{\sum_{k} \exp(z_{h,i,k}) + \exp(z_{h}^{\prime})}
```

让逐 head 的 attention 可以选择"不对任何 token 施加注意力"，实现上由 `sparse_attn` 核融合进 softmax 分母：

```python
self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
...
o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
```

##### Sliding Window Branch

每层保留 `n_win=128` 个 uncompressed KV，跟压缩 KV 拼在一起做 attention：

```python
self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim))
# kv_cache_size = args.window_size + (max_seq_len // compress_ratio if compress_ratio else ...)
# 前 window_size 行放 SWA KV，后面 max_seq_len/m 行放 compressed KV
```

**这其实是对 CSA / HCA 的漏洞修复，有两个动机：**

1. **维持严格因果性带来的块内盲区**：CSA / HCA 的 query 只能 select "已经完成的压缩块"，跟 query 同属一个压缩块的 $`m-1`$ 个邻居彼此看不见（CSA 的 $`m=4`$、HCA 的 $`m'=128`$ 都有这个问题）。
2. **最近邻优先**：最近 token 相关性最强，留未压缩的精确版本比压缩后的更精确。

---

### 1.2 mHC：替换 Residual 为稳定、自适应表达的受约束连接结构

详细介绍可以参考 **mHC: Manifold-Constrained Hyper-Connections**。

* **动机**：Residual Connections 表达能力受限、且存在 Seesaw 效应。作为 Residual Connections 的改进 Hyper-Connections (HC) 也存在数值不稳定、系统开销大的问题。

#### 1.2.1 HC 的形式化表示

HC（Hyper-Connections）的思路是把残差流从 $`C`$ 维扩宽到 $`nC`$ 维，$`n`$ 是 expansion rate，相当于有 $`n`$ 条并行残差流。单层 HC 的公式：

```math
\mathbf{x}_{l+1} = \mathcal{H}_l^{res} \mathbf{x}_l + \mathcal{H}_l^{post\top} \mathcal{F}(\mathcal{H}_l^{pre} \mathbf{x}_l, \mathcal{W}_l)
```

其中：

* $`\mathbf{x}_l \in \mathbb{R}^{n \times C}`$ 是 $`n`$ 流拓展的残差状态（相比原始残差连接多了 $`n`$ 倍宽度）
* $`\mathcal{H}_l^{res} \in \mathbb{R}^{n \times n}`$ 是残差流内的可学习特征混合矩阵
* $`\mathcal{H}_l^{pre} \in \mathbb{R}^{1 \times n}`$ 把 $`nC`$ 维残差流聚合盘成 $`C`$ 维送进 Layer（保证原始变换 $`\mathcal{F}`$ 不变）
* $`\mathcal{H}_l^{post} \in \mathbb{R}^{1 \times n}`$ 把 Layer 的 $`C`$ 维输出映射回 $`nC`$ 维残差流

参考上图，可以看到 HC 在 Pre Mapping 和 Post Mapping 上引入了**可学习矩阵（橙色）**，mHC 在此基础上对 Res Mapping 和 Pre / Post Mapping 都增加了**流形投影（绿色）**。HC 的三个 Mapping 可学习矩阵计算方法：

```math
\begin{cases}
\tilde{\mathbf{x}}_l = \text{RMSNorm}(\mathbf{x}_l) \\
\mathcal{H}_l^{pre} = \alpha_l^{pre} \cdot \tanh(\theta_l^{pre} \tilde{\mathbf{x}}_l^\top) + \mathbf{b}_l^{pre} \\
\mathcal{H}_l^{post} = \alpha_l^{post} \cdot \tanh(\theta_l^{post} \tilde{\mathbf{x}}_l^\top) + \mathbf{b}_l^{post} \\
\mathcal{H}_l^{res} = \alpha_l^{res} \cdot \tanh(\theta_l^{res} \tilde{\mathbf{x}}_l^\top) + \mathbf{b}_l^{res}
\end{cases}
```

<div align="right">(1)</div>

其中：

* $`\alpha_l^{pre}, \alpha_l^{post}, \alpha_l^{res} \in \mathbb{R}`$ 是可学习的 gating 因子，初始化为小值（论文中 $`\alpha = 0.01`$）
* $`\theta_l^{pre}, \theta_l^{post} \in \mathbb{R}^{1 \times C}, \;\theta_l^{res} \in \mathbb{R}^{n \times C}`$ 是动态映射（dynamic mapping）的线性投影参数
* $`\mathbf{b}_l^{pre}, \mathbf{b}_l^{post} \in \mathbb{R}^{1 \times n}, \;\mathbf{b}_l^{res} \in \mathbb{R}^{n \times n}`$ 是静态偏置（static mapping）

> 这里有一个细节：HC 的 mapping 由两部分组成：动态部分（依赖当前输入 $`\mathbf{x}_l`$）和静态部分（全局可学习偏置）。初始化时 $`\alpha`$ 接近 0，所以主要是静态偏置在起作用，接近标准残差网络的 identity mapping。

---

#### 1.2.2 mHC 核心约束 / 代码实现

还是最上面的对比图，mHC 在 HC 的基础上做出的改进：**限制 $`\mathcal{H}_l^{res}`$ 是双随机矩阵（doubly stochastic matrix）**。

双随机矩阵流形的定义（Birkhoff polytope $`\mathcal{M}^{res}`$）：

```math
\mathcal{P}_{\mathcal{M}^{res}}(\tilde{\mathcal{H}}_l^{res}) := \left\lbrace \mathcal{H}_l^{res} \in \mathbb{R}^{n \times n} \mid \mathcal{H}_l^{res} \mathbf{1}_n = \mathbf{1}_n, \; \mathbf{1}_n^\top \mathcal{H}_l^{res} = \mathbf{1}_n^\top, \; \mathcal{H}_l^{res} \ge 0 \right\rbrace
```

<div align="right">(2)</div>

其中，$`\mathbf{1}_n`$ 是 $`n`$ 维全 1 向量；条件 $`\mathcal{H}_l^{res} \mathbf{1}_n = \mathbf{1}_n`$ 要求每行之和为 1；条件 $`\mathbf{1}_n^\top \mathcal{H}_l^{res} = \mathbf{1}_n^\top`$ 要求每列之和为 1；条件 $`\mathcal{H}_l^{res} \ge 0`$ 要求元素非负。$`n = 1`$ 时，双随机矩阵退化为标量 1，整个 HC 退化为标准残差连接 identity mapping。

双随机矩阵流形约束通过以下三个性质解决 HC 的问题：

1. **Norm Preservation（范数保持）**：双随机矩阵的谱范数 $`\|\mathcal{H}_l^{res}\|_2 \le 1`$。这意味着学习到的映射是 non-expansive 的，梯度爆炸问题得到缓解。
2. **Compositional Closure（乘法封闭）**：双随机矩阵对矩阵乘法封闭。也就是说，复合映射 $`\prod_{i=1}^{L-l} \mathcal{H}_{L-i}^{res}`$ 仍然是双随机矩阵，整个深度方向的稳定性都得到了保证。
3. **Birkhoff Polytope 几何解释**：$`\mathcal{M}^{res}`$ 是所有置换矩阵的凸包。因此 $`\mathcal{H}_l^{res} \mathbf{x}_l`$ 本质上是各 stream 特征的**凸组合（convex combination）**，均值守恒、方差有界，很自然地防止了信号漂移。

**简单例子**：设 $`n = 2`$，下述 $`\mathcal{H}_l^{res}`$ 就是一个合法的双随机矩阵：

```math
\mathcal{H}_l^{res} = \begin{pmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{pmatrix}
```

它的优势：每条 stream 输出 = 自身的 0.7 倍 + 另一条流的 0.3 倍。信息有了混合，但每条流的总强度没有放大，就是"稳定且有信息交换"。

另外，论文还对 $`\mathcal{H}_l^{pre}`$ 和 $`\mathcal{H}_l^{post}`$ 加了非负性约束，防止正负系数合成时信号相消。

mHC 中三个 mapping 的计算分两步：

##### Step 1: 计算未约束的 $`\tilde{\mathcal{H}}`$

给定第 $`l`$ 层的输入隐层矩阵 $`\mathbf{x}_l \in \mathbb{R}^{n \times C}`$，先 flatten 成 $`\tilde{\mathbf{x}}_l = \text{vec}(\mathbf{x}_l) \in \mathbb{R}^{1 \times nC}`$，然后：

```math
\begin{cases}
\tilde{\mathbf{x}}_l' = \text{RMSNorm}(\tilde{\mathbf{x}}_l) \\
\tilde{\mathcal{H}}_l^{pre} = \alpha_l^{pre} \cdot (\tilde{\mathbf{x}}_l' \varphi_l^{pre}) + \mathbf{b}_l^{pre} \\
\tilde{\mathcal{H}}_l^{post} = \alpha_l^{post} \cdot (\tilde{\mathbf{x}}_l' \varphi_l^{post}) + \mathbf{b}_l^{post} \\
\tilde{\mathcal{H}}_l^{res} = \alpha_l^{res} \cdot \text{mat}(\tilde{\mathbf{x}}_l' \varphi_l^{res}) + \mathbf{b}_l^{res}
\end{cases}
```

<div align="right">(3)</div>

* $`\varphi_l^{pre}, \varphi_l^{post} \in \mathbb{R}^{nC \times n}, \; \varphi_l^{res} \in \mathbb{R}^{nC \times n^2}`$ 是线性投影参数
* $`\text{mat}(\cdot)`$ 是从 $`\mathbb{R}^{1 \times n^2}`$ 到 $`\mathbb{R}^{n \times n}`$ 的 reshape 函数

> **mHC 和 HC 在参数化上的关键区别**：HC 用的是各流独立的 $`n \times C`$ 状态（每条流单独 RMSNorm），mHC 先把所有流 **flatten 拼接**成 $`1 \times nC`$ 的向量，再做 RMSNorm 和投影。这样 mHC 在计算三个 mapping 时能利用**跨流的完整信息**。

##### Step 2: 将 $`\tilde{\mathcal{H}}`$ 投影到流形约束

```math
\begin{cases}
\mathcal{H}_l^{pre} = \sigma(\tilde{\mathcal{H}}_l^{pre}) \\
\mathcal{H}_l^{post} = 2\sigma(\tilde{\mathcal{H}}_l^{post}) \\
\mathcal{H}_l^{res} = \text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}_l^{res})
\end{cases}
```

<div align="right">(4)</div>

* $`\sigma(\cdot)`$ 是 Sigmoid 函数，保证 $`\mathcal{H}_l^{pre}`$ 元素非负，值域 $`(0, 1)`$
* $`\sigma(\tilde{\mathcal{H}}_l^{post})`$ 乘以 2 的两个目的：
    * 让 $`\mathcal{H}_l^{post}`$ 的期望值约为 1（$`2 \times \sigma(0) = 1`$），和 HC 初期静态偏置为 1 的设定对齐。
    * 让输出范围在 $`(0, 2)`$，保留更大表达空间。
* $`\text{Sinkhorn-Knopp}(\cdot)`$ 把任意正矩阵迭代投影为双随机矩阵：先对所有元素取 $`\exp`$ 保证正数，然后交替做行归一化和列归一化，下面详细介绍。

##### Sinkhorn-Knopp 算法

先做指数变换让所有元素为正：$`\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_l^{res})`$，然后交替做行归一化和列归一化：

```math
\mathbf{M}^{(t)} = \mathcal{T}_r \left( \mathcal{T}_c(\mathbf{M}^{(t-1)}) \right)
```

<div align="right">(5)</div>

* $`\mathcal{T}_r`$ 表示行归一化（每行除以行和）
* $`\mathcal{T}_c`$ 表示列归一化(每列除以列和)

当 $`t \to \infty`$ 时，$`\mathbf{M}^{(t)}`$ 收敛到双随机矩阵 $`\mathcal{H}_l^{res} = \mathbf{M}^{(t_{\max})}`$。实验中取 $`t_{\max} = 20`$。

> **直观理解**：每次行归一化让每行和为 1（满足行条件），但可能破坏列条件；每次列归一化修复列条件，但可能破坏行条件。交替迭代，两个条件同时收敛。收敛速度很快，20 步足以达到很高精度。

对应代码实现（`Block.hc_pre`）：

```python
class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains 'hc_mult' copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        # x: [b, s, hc, d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b, s, h...
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # x: [b, s, d], residual: [b, s, hc, d], post: [b, s, hc], comb: [b, s, hc, hc], y: [b, s, hc...
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual, dim=3)
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x
```

---

### 1.3 MoE：Anticipatory Routing + Hash Routing

#### 1.3.1 Hash Routing

DeepSeek V4 的前 3 个 MoE 层换成 Hash routing。Hash routing 具体来说就是：**每个 token ID 用一个固定 hash 函数映射到固定的 expert 集合，不需要任何学习。即 token ID $`\to`$ expert indices，训练前预计算。**

V4-Pro / V4-Flash 都设置 `n_hash_layers=3`，只用前 3 层的原因：

* **底层是词形特征**（底层特征，即基本语法属性），不需要语义级别的路由。
* **Hash 路由零 gating cost、零 load imbalance**，用在底层可以省掉学习开销，帮助后期层收敛。

参考代码 `Gate.__init__` + `forward` 实现：

```python
self.hash = layer_id < args.n_hash_layers  # 前 n_hash_layers 层为 hash
if self.hash:
    self.tid2eid = nn.Parameter(
        torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32),
        requires_grad=False,
    )
    self.bias = None
else:
    self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
# ...
if self.hash:
    indices = self.tid2eid[input_ids]
else:
    indices = scores.topk(self.topk, dim=-1)[1]
```

---

#### 1.3.2 Sqrt(Softplus(·))

V3 / V3.2 系列 gate 用 sigmoid 计算：

```math
s_i = \sigma(\mathbf{w}_i^\top \mathbf{x})
```

V4 改成：

```math
s_i = \sqrt{\text{softplus}(\mathbf{w}_i^\top \mathbf{x})} = \sqrt{\log(1 + e^{\mathbf{w}_i^\top \mathbf{x}})}
```

如何理解 $`\sqrt{\text{softplus}(x)}`$ 相比 $`\sigma(x)`$ 的优势，我们先看两个极限：

* 当 $`x \to -\infty`$ 时：$`\sigma(x) \to 0`$（exp 指数衰减），$`\sqrt{\text{softplus}(x)} \to \sqrt{e^x} = e^{x/2}`$（更平缓）
* 当 $`x \to +\infty`$ 时：$`\sigma(x) \to 1`$（饱和），$`\sqrt{\text{softplus}(x)} \to \sqrt{x}`$（发散）

所以 sqrt-softplus 的一大特性是**在高分区仍有梯度，区分度更强**。V3 的 sigmoid 在 logit 很大时 score 都被压到 1 附近，top-k 选择基本由 bias 主导。V4 改用 sqrt-softplus 后，score 本身就能表达高分优先级。

对应代码实现参考 `Gate.forward`：

```python
elif self.score_func == "sigmoid":
    scores = scores.sigmoid()
else:  # "sqrtsoftplus"
    scores = F.softplus(scores).sqrt()
```

---

#### 1.3.3 Anticipatory Routing

作者发现 loss spike 跟 MoE 路由的同步有关，路由机制会放大 outlier。做法是用不同步更新：在第 $`t`$ 步用当前参数 $`\theta_t`$ 做 feature 计算，但路由 index 用 $`\theta_{t-\Delta t}`$ 提前算好的版本，即路由滞后一步。但是这个只作为 rescue 策略，只有在检测到 loss spike 时短期激活，稳定后切回同步更新。

---

#### 1.3.4 SwiGLU Clamping

`config.json` 里 `swiglu_limit=10.0`，即对 SwiGLU 的线性分量 clamp 到 $`[-10, 10]`$、gate 分量限制上界 10：

```python
if self.swiglu_limit > 0:
    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
    gate = torch.clamp(gate, max=self.swiglu_limit)
```

---

#### 1.3.5 其他 MoE 配置

* **384 routed + 1 shared expert**，V4-Pro 每 token 激活 6 个 routed + 1 shared。
* **auxiliary-loss-free load balancing** 保留（bias 控制 top-k），加一个低权的 sequence-level balance loss。
* **V4 把 MoE expert 权重全程 FP4**（MXFP4 格式），CSA indexer 的 QK 路径也 FP4，其余部分保持 FP8。

---

### 1.4 Muon 优化器

详细解析参考 **Muon 优化器**。

#### 1.4.1 Muon：Newton-Schulz 正交化代替 AdamW 的二阶矩

我们直接来看伪代码，因为 Muon 的伪代码真的非常简单：

> **Algorithm 2** Muon
> **Require**: Learning rate $`\eta`$, momentum $`\mu`$
> 1: Initialize $`B_0 \leftarrow 0`$
> 2: **for** $`t = 1, \dots`$ **do**
> 3: &nbsp;&nbsp;Compute gradient $`G_t \leftarrow \nabla_\theta L_t(\theta_{t-1})`$
> 4: &nbsp;&nbsp;$`B_t \leftarrow \mu B_{t-1} + G_t`$
> 5: &nbsp;&nbsp;$`O_t \leftarrow \text{NewtonSchulz5}(B_t)`$
> 6: &nbsp;&nbsp;Update parameters $`\theta_t \leftarrow \theta_{t-1} - \eta O_t`$
> 7: **end for**
> 8: **return** $`\theta_t`$

除开 `NewtonSchulz5` 这个函数看不懂，整体逻辑非常简单：

* 计算 Gradient
* 滑动平均计算 Momentum
* `NewtonSchulz5` 计算一个梯度变化量 $`O_t`$
* 直接用 $`\eta O_t`$ 更新参数

而这里的 `NewtonSchulz5` 是指 Newton-Schulz (NS) 迭代，`NewtonSchulz5` 是指迭代了 5 次，NS 迭代的意义是：

```math
\text{Ortho}(B_t) = \arg\min_{O} \left\lbrace \|O - B_t\|_F : \text{either } O^\top O = I \text{ or } OO^\top = I \right\rbrace
```

即 NS 迭代会寻找一个与动量矩阵 $`B_t`$ 最接近的**半正交矩阵** $`O_t`$ 用来更新权重。

这其实是和 $`U V^\top`$ 等价的，其中 $`B_t`$ 的奇异值分解是 $`U S V^\top`$。但是因为奇异值分解的计算复杂，所以使用了其他求 $`O_t`$ 的方式，Newton-Schulz (NS) 迭代就是 Muon 使用的近似策略。

DeepSeek V4 给出的核心 Hybrid Newton-Schulz 迭代：

```math
M_k = a M_{k-1} + b (M_{k-1} M_{k-1}^\top) M_{k-1} + c (M_{k-1} M_{k-1}^\top)^2 M_{k-1}
```

* **前 8 步用激进系数** $`(a, b, c) = (3.4445, -4.7750, 2.0315)`$：快速把奇异值推向 1
* **后 2 步用稳态系数** $`(a, b, c) = (2, -1.5, 0.5)`$：把奇异值锁死在 1 附近

> **Hybrid Newton-Schulz 迭代分两阶段的原因？** 单一激进系数可能 overshoot（奇异值震荡），单一稳态系数又收敛太慢。
>
> **为什么要找半正交矩阵？** 因为 SGD-momentum 和 Adam 计算得到的参数变化量（$`\Delta W`$）一般是低秩矩阵，也就是说权重的更新仅由少数几个方向主导。而正交化有效地平衡了各个"方向"的更新幅度。有的更新方向在更新中幅度较小，但对整体的优化很重要。
>
> **为什么 V4 不再需要 QK-Clip？** V4 对 Q 和 KV 直接做 RMSNorm，attention logits 本身就不会爆炸，所以 Muon 不用 QK-Clip。

---

#### 1.4.2 Muon vs AdamW 的参数分配

需要注意的是，在实际训练中并非所有参数都走 Muon：

| 模块 | 优化器 |
| --- | --- |
| **Embedding** | AdamW |
| **Prediction head** | AdamW |
| **mHC 的静态 bias $`S_l`$、gating factor $`\alpha_l`$** | AdamW |
| **RMSNorm** | AdamW |
| **其他所有参数（attention, MoE expert, mHC 等）** | **Muon** |

* embedding / head 稀疏更新（每次只更新 batch 里出现的 token 行），走 AdamW。
* RMSNorm、mHC 是 1D 向量，Muon 对标量 / 向量没有意义。

---

## 2. Infra：KV Cache / FLOPs 推导

> 🤔 *In the one-million-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2.*

DeepSeek V4 支持 1M 长上下文、以及极低的推理成本，原论文给出「1M 上下文配置下，V4-Pro 单 token FLOPs 是 V3.2 的 27%、KV cache 是 V3.2 的 10%」，成为最近讨论里最容易被「转述但不验证」的两个数字。这一节我们用 V3.2 / V4-Pro 各自的 Inference 代码和论文中的全部架构参数，**拆解 DeepSeek V4-Pro 背后的 Infra 设计、KV Cache / FLOPs 计算**。

*（参考图：左图 Accumulated KV Cache (GB) vs Sequence Length (K)，V4-Pro 比 V3.2 缩小 13.7×、V4-Flash 缩小 9.5×；右图 Single-Token FLOPs (T) vs Token Position (K)，V4-Pro 比 V3.2 降低 9.8×、V4-Flash 降低 3.7×。）*

---

### 2.1 KV Cache 对比

#### 2.1.1 DeepSeek V3.2

推荐先阅读 📄 *DeepSeek-V3.2-EXP*、📄 *GLM-5（第一节）*。

*（参考图：DS V3.2 的 DSA 架构，关注红框内的 K/V 分支需要 cache、右侧绿色 Lightning Indexer 本质也是 MLA 同样需要 cache。）*

首先参考 V3.2 的 `config.json` 关键配置：

* `num_hidden_layers = 61` （隐藏层数）
* `kv_lora_rank = 512` （MLA 的 latent KV 维度）
* `qk_rope_head_dim = 64` （MLA 的 RoPE 部分维度）
* `index_topk = 2048`、`index_n_heads = 64`、`index_dim = 128` （DSA Lightning Indexer）
* `dtype = "fp8"`、`scale dtype = FP32` （精度、量化 scale 精度）

然后来看 **V3.2 的 MLA + DSA 架构**：

* **MLA 架构**：把 K / V 共享压成 latent 向量 $`\mathbf{c}_t^{KV}`$（不持久存原始 K / V），**并且由于 decoupled RoPE，需要同时缓存 $`\mathbf{c}_t^{KV}`$ 以及 $`\mathbf{k}_t^R`$，并且在多头间共用（参考上图）。**

DeepSeek V3.2 K / V 分支的公式可以表示为：


```math
\mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t,
```

```math
[\mathbf{k}_{t,1}^C; \mathbf{k}_{t,2}^C; \dots; \mathbf{k}_{t,n_h}^C] = \mathbf{k}_t^C = W^{UK}\mathbf{c}_t^{KV},
```

```math
\mathbf{k}_t^R = \text{RoPE}\left(W^{KR}\mathbf{h}_t\right),
```

```math
\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R],
```

```math
[\mathbf{v}_{t,1}^C; \mathbf{v}_{t,2}^C; \dots; \mathbf{v}_{t,n_h}^C] = \mathbf{v}_t^C = W^{UV}\mathbf{c}_t^{KV}
```

DeepSeek V3.2 K / V 分支本质上做的事情：先降维得到 $`\mathbf{c}_t^{KV}`$，再升维得到 $`\mathbf{k}_t^C`$ 和 $`\mathbf{v}_t^C`$，对另一分支的 $`W^{KR}\mathbf{h}_t`$ 做 RoPE，最后 $`\mathbf{k}`$ 拼接，分别输出分头后的 $`\mathbf{k}`$ 和 $`\mathbf{v}`$。

* **DSA 架构**：DSA 在 MLA 之上叠了一层 sparse：每次 attention 前先用 Lightning Indexer 在所有 KV 上算 index score，选出 top-k=2048 个 token 进 core attention。关键点：**Indexer 选出的 top-k 是「计算时只算这 2048 个」，但 KV cache 仍然要保留全部 token，因为下一个 query 选谁是动态的；在多头间共用。**

每 token 每层的 cache 由三部分组成（PE-Cache 用 FP16，其余 FP8）：

| 统计项 | 形状 | FP32 scale | 字节数 |
| --- | --- | --- | --- |
| **MLA KV-Cache ($`\mathbf{c}_t^{KV}`$)** | `[L, 61, 512]` FP8 | `(512/128) * 4 = 16` bytes | `512 + 16 = 528` bytes |
| **MLA PE-Cache ($`\mathbf{k}_t^R`$)** | `[L, 61, 64]` FP16 | None | `64 * 2 = 128` bytes |
| **DSA Indexer Cache** | `[L, 61, 128]` FP8 | `(128/128) * 4 = 4` bytes | `128 + 4 = 132` bytes |

单层单 token 合计 `528 + 128 + 132 = 788` bytes，61 层全模型每 token 共 `61 * 788 = 48068` bytes，由此得到：

```math
\text{KV}_{\text{V3.2}}(L) = 48068 \cdot L \text{ bytes}
```

##### 为什么 V3.2 的 MLA latent 512 维不需要再切 RoPE / NoPE？

V3.2 用的是 decoupled RoPE（DeepSeek-V2 提出）：`wkv_a` 输出 `[B, L, 512+64]`，预先把 RoPE 部分（64 维）拆到独立的 pe_cache，剩下的 512 维 latent 整段都是 NoPE，所以可以一次性 FP8 量化、不需要再切分。DeepSeek V4 把这个设计换了：V4 的 KV entry 是 `head_dim = 512` 的单一 vector，把 RoPE 和 NoPE 合并在一起（partial RoPE），最后 64 维做 RoPE、前 448 维做 NoPE / FP8 量化。下面 V4 的 Main entry 分析里可以看到「64 + 448 切分」就是这个设计差异的来源。

##### 关于源码 vs 设计目标的精度差异

上面 V3.2 的分析按 MLA latent 用 FP8 + FP32 scale 存储（paper 设计目标），与 V3.2 reference impl 中 Indexer 的精度一致。但 V3.2 reference impl 的 MLA latent 部分实际是 BF16（代码注释「we use fp8 kv cache in actual deployment, so here we simulate the precision by casting kv to fp8 and then back to bf16」），框架实现可能存在差异。

---

#### 2.1.2 DeepSeek V4-Pro

推荐先阅读 📄 *DeepSeek-V4 架构：技术报告 / 源码解析*。

*（参考图：DeepSeek-V4 KV 缓存布局示意。KV 缓存由两部分组成：用于 CSA / HCA 的传统 KV 缓存，以及用于 SWA 和 CSA / HCA 中尚未准备好压缩的 token 的 state cache。）*

必须先读完 📄 *DeepSeek-V4 架构：技术报告 / 源码解析* 了解 V4 架构，然后再思考在这个架构里需要缓存什么。回顾 CSA（Compressed Sparse Attention）/ HCA（Heavily Compressed Attention）压缩公式，**并对应上面的缓存布局**：

* **CSA 的 KV 压缩公式：**

```math
C^a = H \cdot W^{aKV}, \quad C^b = H \cdot W^{bKV}
```


```math
Z^a = H \cdot W^{aZ}, \quad Z^b = H \cdot W^{bZ}
```


```math
[S^a; S^b] = \text{Softmax}_{\text{row}}([Z^a + B^a; Z^b + B^b])
```


```math
C_i^{\text{Comp}} = \sum_{j=mi}^{m(i+1)-1} S_j^a \odot C_j^a + \sum_{j=m(i-1)}^{mi-1} S_j^b \odot C_j^b
```

* **HCA 的压缩公式复用了 CSA，去掉了 a / b 双通道：**

```math
C = H \cdot W^{KV}, \quad Z = H \cdot W^Z, \quad S = \text{Softmax}_{\text{row}}(Z + B), \quad C_i^{\text{Comp}} = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j
```



**读这几条公式可以梳理出除了 Indexer 之外「每层每 request 的 cache 应该存什么」：**

* Compressed entry $`C_i^{\text{Comp}}`$ 要进长期 KV cache。每 $`m`$（CSA）或 $`m'`$（HCA）个 token 算出一条压缩后的 `c = head_dim` 维的 entry，永久保留供后续 query attend。
* Scale Z / S 是压缩瞬间的中间量，不进 cache。每攒够 $`m`$ 个 token 触发一次公式压缩，原始的 `Z` / `S` 立即可以丢弃。
* State Cache 是「下一次压缩还没攒够输入」的临时 KV State。参考 V4 论文："all pending tokens and their associated hidden states must be retained in a buffer until the compression operation can be executed"。

参考 V4-Pro 的 `config.json` 获取关键参数：

* `num_hidden_layers = 61`、`head_dim = 512`、`n_heads = 128`、`num_key_value_heads = 1` （所有 query head 共享同一个 KV head，shared-KV MQA 是底层配置）
* `window_size = 128` （SWA 分支）
* `index_topk = 1024`、`index_n_heads = 64`、`index_head_dim = 128` （CSA 自己的 Lightning Indexer）
* `compress_ratios = [128, 128, 4, 128, 4, ..., 128, 4, 0]`，前 2 层是 HCA（128）、后续 CSA / HCA 交替（4 / 128），末尾的 `0` 是 MTP 层。61 个非 MTP 层中，HCA 层 = 2 + 29 = 31 层（前 2 层 + 后续交替里一半），CSA 层 = 30 层。

##### HCA 层 Cache 计算

*（参考图：Shared Key-Value Multi-Query Attention 架构下，Sliding Window KV Entries 与 Heavily Compressed KV Entries 通过 Token-Level Compressor 拼接并与 Queries 做 Attention。）*

看过 📄 *DeepSeek-V4 架构：技术报告 / 源码解析* 应该了解，HCA 其实是简化的非稀疏 CSA，甚至共用了一部分类实现。因此我们先计算 HCA 层的 Cache。首先看 HCA 压缩后的单条 Compressed entry $`C^{\text{Comp}}`$ 的存储占用，有两个细节需要关注：

* **Main KV partial RoPE 切分**：V4 使用了 partial RoPE，KV entry 是 `head_dim = 512` 的单一 vector，把 RoPE 和 NoPE 合并在一起，最后 64 维做 RoPE、前 448 维做 NoPE / FP8 量化。参考源码中的 `apply_rotary_emb(kv[..., -rd:], freqs_cis)`，即给最后 64 维做 RoPE 旋转、保持 BF16。把前 448 维做 FP8 量化（block 64）。
* **Main KV scale dtype = UE8M0（1 byte E8M0）**：V4-Pro 全局 scale_dtype 设成 UE8M0。

因此每条 Compressed Main KV entry（图中单个 Heavily Compressed KV Entries）大小（其实后面的 CSA 的一条 Entry 的大小也等于 HCA 的）：

```math
S_{\text{main entry}} = \underbrace{64 \times 2}_{\text{RoPE BF16}} + \underbrace{448 \times 1}_{\text{NoPE FP8}} + \underbrace{(448 / 64) \times 1}_{\text{UE8M0 scale (block 64)}} = 128 + 448 + 7 = 583 \text{ bytes}
```

每个 HCA 层有三类 cache：

* **SWA KV**（图中全部 Sliding Window KV Entries）：`window_size × head_dim = 128 × 512 = 65536` bytes（这部分是 sliding window 内的未压缩 KV，简化按 FP8 算且忽略 scale，如果是 BF16 reference impl 下应是乘 2 倍）。
* **HCA KV**（图中全部 Heavily Compressed KV Entries，随序列增长）：`L / m' × S_main_entry = L / 128 × 583` bytes。
* **HCA State Cache**（压缩 Buffer，FP32）：HCA 无 overlap，buffer 上界 = 正在累积、还没攒够 128 个 token 的中间状态（C, Z）各一份，形状 `[m', c] = [128, 512]` $`\text{FP32} \rightarrow 2 \times 128 \times 512 \times 4 = 512 \text{ KB}`$。

单层 HCA：`SWA + HCA KV Cache + State Cache = 0.5625 MB + 4.55 × L bytes`。

31 个 HCA 层总计：`31 × 0.5625 MB = 17.4375 MB` + `31 × 4.55L = 141.05L` bytes。

##### CSA 层 Cache 计算

*（参考图：CSA 层的架构，相比 HCA 额外增加了 Lightning Indexer、Index Scores、Multi-Query Attention 以及 Top-k Selector 选出 Selected Compressed KV Entries 的过程。）*

相比 HCA，CSA 会复杂一些：Compressed Entry 分为 Compressed KV Entry 和 Compressed Indexer K entry（Compressed Indexer Key）。Compressed KV Entry 每一条目的大小和 HCA 一致，即：

```math
S_{\text{main entry}} = \underbrace{64 \times 2}_{\text{RoPE BF16}} + \underbrace{448 \times 1}_{\text{NoPE FP8}} + \underbrace{(448 / 64) \times 1}_{\text{UE8M0 scale (block 64)}} = 128 + 448 + 7 = 583 \text{ bytes}
```

参考 V4 技术报告，每条 Compressed Indexer K entry 大小（MXFP4 + UE8M0）：

```math
\text{Indexer entry} = \underbrace{128 \times 0.5}_{\text{MXFP4}} + \underbrace{(128 / 32) \times 1}_{\text{UE8M0 (block 32)}} = 64 + 4 = 68 \text{ bytes}
```

CSA 层的五类 cache：

* **Compressed Main KV**：每 `m=4` token seq 压成 `583 bytes`，单层总大小 `583/4*L bytes`
* **Compressed Indexer K**：每 `m=4` token seq 压成 `68 bytes`，单层总大小 `68/4 = 17 bytes`
* **SWA KV**：`64 KB`（同 HCA）
* **CSA State Cache**（FP32 Buffer）：CSA 有 overlap，每条 entry 由 `2m` 个 token 派生（前 `m` 走 ($`\text{C}^{\text{a}}`$, $`\text{Z}^{\text{a}}`$)，后 `m` 走 ($`\text{C}^{\text{b}}`$, $`\text{Z}^{\text{b}}`$)）。buffer 形状 = `[2m, 2c] = [8, 1024]` FP32，KV / Score 各一份：

```math
2 \times (2\text{m} \times 2\text{c} \times 4 \text{ byte}) = 2 \times (8 \times 1024 \times 4) = 64 \text{ KB}
```


* **Indexer State Cache**（FP32 Buffer）：同样的 `[2m, 2c^I] = [8, 256]` FP32，KV / Score 各一份：

```math
2 \times (2\text{m} \times 2\text{c}^{\text{I}} \times 4) = 2 \times (8 \times 256 \times 4) = 16 \text{ KB}
```



单层 CSA：`(64 + 64 + 16) KB + (583/4 + 17) × L = 0.140625 MB + 162.75 × L bytes`。

30 个 CSA 层合计：`30 × 0.140625 = 4.21875 MB` + `30 × 162.75 = 4882.5 × L` bytes。

##### CSA + HCA 层 Cache 总计

**DeepSeek V4-Pro KV 总计**：

```math
\text{KV}_{\text{V4-Pro}}(L) = 21.65625 \text{ MB} + 5{,}023.55 \cdot L \text{ bytes}
```

L 系数里 97% 来自 CSA 的 Compressed KV + Indexer KV，HCA 在 KV 上的占比很小，目标只在于长程性能保底。

对比 §2.1.1 推得的 V3.2 结果：

```math
\text{KV}_{\text{V3.2}}(L) = 48068 \cdot L \text{ bytes}
```

---

#### 2.1.3 1M 上下文长度下 V3.2 vs V4-Pro Cache 效率对比

| 序列长度 | V3.2 | V4-Pro | V4-Pro / V3.2 |
| --- | --- | --- | --- |
| **8K** | 375.53 MB | 60.90 MB | 16.22% |
| **32K** | 1502.12 MB | 178.65 MB | 11.89% |
| **128K** | 6008.50 MB | 649.62 MB | 10.81% |
| **256K** | 12017.00 MB | 1277.58 MB | 10.63% |
| **512K** | 24034.00 MB | 2533.50 MB | 10.54% |
| **1M** | 48068.00 MB | 5045.35 MB | 10.50% |

可以得出几个结论：

* **1M 时 V4-Pro / V3.2 = 10.50%**，与原 V4 论文摘要「10%」匹配。
* **短序列（8K）V4-Pro 占比反而高到 16%**，因为 $`\sim 21.7 \text{ MiB}`$ 的常数项（State Cache）在短序列里占主导。**可见 CSA + HCA 的 State Cache 是固定开销，序列越短占比越大。**
* **32K 后 V4-Pro / V3.2 稳定在 10.5%~11.9% 之间**，256K 上下文之后基本就 $`\sim 10.5\%`$，可见 V4 架构确实在 KV 储存效率上远超 V3.2 的 DSA 架构。

*（参考图：右下角插图为 Accumulated KV Cache (GB) vs Sequence Length (K) 曲线，直观展示 DeepSeek-V3.2、V4-Pro、V4-Flash 在不同序列长度下的 KV Cache 增长趋势，其中 V4-Pro 比 V3.2 小 13.7×。）*

---

### 2.2 FLOPs 对比

#### 2.2.1 DeepSeek V3.2

DeepSeek V3.2 总参数量 685B，每 Token 激活参数量约 37B。

Weight 参数相关 FLOPs 为 `37 * 2 = 74B`。

DSA 有两块 attention 计算：

* **Lightning Indexer**：在全部 `L` 个 KV 上算分数。每层 `index_n_heads × index_head_dim × L × 2 = 64 × 128 × L × 2 = 16384 × L` FLOPs，61 层共 `999424 × L` FLOPs。
* **Top-k MQA core attention**：选出 `top-k = 2048` 个 token 后做 attention。每层 `n_heads × (k_dim + v_dim) × top_k × 2 = 128 × (512+64+512) × 2048 × 2`，61 层共约 `34.8B` FLOPs。

V3.2 总计算量（估计）为：

```math
F_{\text{V3.2}}(L) = 74\text{B} + 999424 \cdot L + 34.8\text{B} = 999424 \cdot L + 108.8\text{B}
```

---

#### 2.2.2 DeepSeek V4-Pro

CSA 和 HCA 都是 shared-KV MQA，主要计算量是 `n_heads × head_dim × num_kv_entries × 2`。

**HCA 层（31 层）**：每个 query attend 到 `n_win + L/m' = 128 + L/128` 个 KV entry。

* 单层：`2 × 128 × 512 × (128 + L/128) × 2 ≈ 33.55 M + 2048 L` FLOPs
* 31 层共 `1.04B + 63488 × L` FLOPs

**CSA 层（30 个）**：

* **Lightning Indexer**：在 `L/m = L/4` 个压缩 KV 上打分。每层 `64 × 128 × L/4 × 2 = 4096 × L`，30 层共 `122880 × L`。
* **Top-k MQA**：每个 query attend `n_win + top_k = 128 + 1024 = 1152` 个 entry。每层 `2 × 128 × 512 × 1152 × 2 ≈ 0.302B`，30 层共 `9.06B`。

V4-Pro 总计算量（估计）为：


```math
F_{\text{V4-Pro}}(L) = 98\text{B} + 1.04\text{B} + 63{,}488L + 122{,}880L + 9.06\text{B} = 186{,}368 \cdot L + 108.1\text{B}
```

---

#### 2.2.3 1M 上下文长度下 V3.2 vs V4-Pro FLOPs 对比

| 序列长度 | V3.2 | V4-Pro | V4-Pro / V3.2 |
| --- | --- | --- | --- |
| **8K** | 116.99 B | 109.63 B | 93.71% |
| **32K** | 141.55 B | 114.21 B | 80.68% |
| **128K** | 239.80 B | 132.53 B | 55.27% |
| **256K** | 370.79 B | 156.96 B | 42.33% |
| **512K** | 632.79 B | 205.81 B | 32.52% |
| **1M** | 1156.77 B | 303.52 B | 26.24% |

可以得出几个结论：

* **1M 时 V4-Pro / V3.2 ≈ 26.24%**，对应论文摘要的「27%」。
* **FLOPs 改进比低于 KV cache**：1M 时 KV cache 是 V3.2 的 10.50%，FLOPs 是 26.24%。因为 V4-Pro 用了更大激活参数（49B vs 37B），抵消了 attention 改进的收益。
* **CSA Indexer 是 V4 attention 节约的最大单一来源**：V3.2 Indexer 在原始 `L` 上算，V4 Indexer 在压缩后 `L/4` 上算，单这个改进就把 attention 计算的主导项压到了原来的 1/8。

*（参考图：右下角插图为 Single-Token FLOPs (T) vs Token Position (K) 曲线，展示 DeepSeek-V3.2、V4-Pro、V4-Flash 的 FLOPs 随 Token 位置的变化趋势，其中 V4-Pro 比 V3.2 降低 9.8×。）*

---

## 3. 训练策略

📌 **Paper**：《DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence》

**Model Repository**：deepseek-ai/DeepSeek-V4

**Key Knowledge Points**

* **Pre-training**：Flash / Pro 训练配置、超长上下文渐进式训练、Sparse Attention 两阶段冷启动 (Sparse Attention cold start)、Training Anti-Spike（Anticipatory Routing & SwiGLU Clamping）、Muon update RMS rescale。
* **Post-training**：用 **On-Policy Distillation (OPD)** 替换 Mixed RL，采用 **Specialist Training + OPD Merged** 两阶段。Specialist Training 使用 self-generative reward model，在 1M context 工程框架内优化 Tool Call Schema、Interleaved Thinking 格式与 Quick Instruction。

模型架构（§1）与 Infra（§2）部分已在前文完整分析，本节聚焦 V4 的训练流程。

---

### 3.1 Pre-Training

相比 V3 / V3.2 系列，V4 在预训练侧的改进集中在**数据、配置、稳定性**三个维度，下面分别展开。

#### 3.1.1 训练数据 (Training Data)

V4 的预训练语料规模为 **32T tokens**，重点优化数据质量。各维度的具体 trick 包括：

* **去模板化 / 移除机器生成数据**：通过专门的过滤器清理大量自动生成的模板化网页内容，缓解 *How to synthesize text data without model collapse* 中提到的 model collapse 风险。
* **保持 Math / Code 数据高占比**：作为 reasoning 能力的核心驱动数据，这两类语料在 V4 训练中仍占主导地位。
* **引入 agentic 数据**：用于增强 code agent 能力（作为传统 reasoning 数据之外的补充）。这部分数据放在 **mid-training** 阶段引入，而非早期 pre-training，避免污染早期 base model 表征。
* **多语言扩展**：增强长尾语言和文化的覆盖。
* **长文档专项**：把科学论文、技术报告、长文档作为高优先级数据，提升 1M context 下的任务表现。
* **Sample-level attention masking**：和 V3 不同，把多个样本 packing 到同一 sequence 时，V4 强制不同样本之间互不可见（V3 允许跨样本可见）。这一改动减少了样本间虚假相关性，对长文档的语义独立性更友好。

**沿用的 Tokenizer / 训练任务配置：**

* **Tokenizer 配置**：在 V3 词表（128K）基础上新增了少量 special tokens。
* **FIM 策略配置**：沿用 V3 的 token-splitting（支持 FIM 训练的 token 序列切分）和 Fill-in-the-Middle（FIM，即 `<PRE> Prefix <SUF> Suffix <MID> Middle <EOS>`）补全任务配置（参考：📄 *什么是大模型 FIM 预训练任务？*）。

> **例：原始 FIM 文本及其切分形式**
>
> 原始文本：
> `The quick brown fox jumps over a lazy dog`
>
> 切分示例：
> `<PRE>The quick brown fox <MID>jumps over<EOM><SUF> a lazy dog`
>
> 对应 PSM 格式：
> `<PRE>The quick brown fox <SUF> a lazy dog<MID>jumps over<EOM>`

---

#### 3.1.2 模型与训练配置

两个不同量级模型 Flash / Pro 的**结构配置 / 训练配置**：

| Hyperparameter | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
| --- | --- | --- |
| **Parameter Total / Activated** | 284B / 13B | 1.6T / 49B |
| **Layers / Hidden_Dim** | 43 / 4096 | 61 / 7168 |
| **Experts / Activated Experts** | 1 shared + 256 routed / 6 Activated | 1 shared + 384 routed / 6 Activated |
| **CSA attention top-k** | 512 | 1024 |
| **训练语料 token 数** | 32T | 33T |
| **最大 batch size (tokens)** | 75.5M | 94.4M |
| **Optimizer (AdamW)** | $`\beta_1 = 0.9, \beta_2 = 0.95, \varepsilon = 10^{-20}`$, weight_decay = 0.1 | $`\beta_1 = 0.9, \beta_2 = 0.95, \varepsilon = 10^{-20}`$, weight_decay = 0.1 |
| **Optimizer (Muon)** | momentum = 0.95, weight_decay = 0.1 | momentum = 0.95, weight_decay = 0.1 |
| **LR: Peak / End** | $`2.7 \times 10^{-4}`$ / $`2.7 \times 10^{-5}`$ | $`2.0 \times 10^{-4}`$ / $`2.0 \times 10^{-5}`$ |
| **LR Schedule** | linear warmup 2000 步 $`\rightarrow`$ peak LR 常数 $`\rightarrow`$ cosine 衰减到 end LR | linear warmup 2000 步 $`\rightarrow`$ peak LR 常数 $`\rightarrow`$ cosine 衰减到 end LR |
| **Sequence length 渐进阶段** | 4K $`\rightarrow`$ 16K $`\rightarrow`$ 64K $`\rightarrow`$ 1M | 4K $`\rightarrow`$ 16K $`\rightarrow`$ 64K $`\rightarrow`$ 1M（4K 阶段更长） |
| **Sparse attention 引入时机** | 前 1T token 全 dense attention $`\rightarrow`$ 64K 时引入 lightning indexer warmup $`\rightarrow`$ 切到 sparse | 相比 Flash，dense 阶段更长，再走两阶段导入 |
| **MTP loss weight** | 0.3 $`\rightarrow`$ 0.1（衰减期） | 0.3 $`\rightarrow`$ 0.1（衰减期） |
| **负载均衡：bias update / loss** | speed = 0.001 / weight = 0.0001 | speed = 0.001 / weight = 0.0001 |

**关键设计点**

* **Muon update RMS rescale = 0.18**：目标是让 Muon 输出的更新幅度匹配 AdamW 的量级，从而能直接复用 AdamW 的 LR 经验值。具体来看 V4 的 Muon 算法相比标准 Muon 增加了 Rescale factor $`\gamma`$。

  ```text
  Algorithm 1 Muon Optimizer for DeepSeek-V4
  Require: Learning rate η, momentum μ, weight decay λ, update rescaling factor γ
  1: for each training step t do
  2:   for each logically independent weight W ∈ R^{n×m} do
  3:     G_t = ∇_W L_t(W_{t-1})                          ▷ Compute gradients
  4:     M_t = μM_{t-1} + G_t                            ▷ Accumulate momentum buffer
  5:     O'_t = HybridNewtonSchulz(μM_t + G_t)           ▷ Nesterov trick and hybrid Newton-Schulz
  6:     O_t = O'_t · √max(n, m) · γ                     ▷ Rescale the update RMS
  7:     W_t = W_{t-1} · (1 - ηλ) - ηO_t                 ▷ Perform weight decay and update
  8:   end for
  9: end for
  ```

  > *（注：上述为 V4 的 Muon 算法；相比标准 Muon，第 6 步新增了 $`\cdot \gamma`$ 缩放因子，用以对齐更新幅度。）*

* **Sparse attention 两阶段切换**：直接从 0 开始训练 sparse attention 会让 indexer 崩溃，所以先用 dense attention 训到 64K（V4-Flash 是 1T token，V4-Pro 更长），再做 lightning indexer 的短期 warmup（dense 推理但只更新 indexer 参数），最后才切到全 sparse。
* **Routing 节点数取消限制**：V3 限制每个 token 最多被路由到固定数量的节点，V4 取消了这个约束，配合 anticipatory routing 让并行策略更灵活。

---

#### 3.1.3 训练稳定性 Trick

100B 以上的 MoE 模型训练会出现 **loss spike** 现象（loss 值突然急剧上升出现尖峰）。

DeepSeek V4 认为 spike 总是和 MoE 层的 outlier 相关，routing 机制的不稳定性会产生 outlier。使用 rollback（把训练状态退回到之前保存的某个 checkpoint）不能根本性解决这个问题，所以 V4 从两个方面进行 outlier 抑制：**routing 抑制恶性反馈循环 + 直接抑制 outlier 异常值**。

##### 技巧 1：Anticipatory Routing

* **动机：标准同步 routing 存在的问题。** Routing 用当前参数 $`\theta_t`$ 算专家分配，但 expert 的更新也用 $`\theta_t`$，二者高度耦合。一旦某个 token 在某个 expert 上出现 outlier 激活，routing 的 score 和 expert 自身的 weight 会同步朝着该 outlier 方向调整，形成正反馈，最终触发 loss spike。
* **实现：解耦 routing 和 backbone 的更新时刻。** 在第 $`t`$ 步 feature 计算仍用 $`\theta_t`$，但 routing index 用历史参数 $`\theta_{t-\Delta t}`$ 计算。实操上为了避开两次加载 model parameter 的开销，在 step $`t - \Delta t`$ 提前 fetch 数据预计算（Anticipatory）并 cache 好后续 step $`t`$ 要用的 routing index。

工程上还做了两层优化：

* **流水线 overlap**：把 anticipatory 计算和 EP 通信 overlap。
* **兜底触发**：默认走标准训练，只有检测到 loss spike 才触发短暂 rollback + 启用 anticipatory routing，无触发一段时间后再切回标准同步 Routing。

##### 技巧 2：SwiGLU Clamping

* **动机：SwiGLU 数值范围爆炸。** SwiGLU 的 linear 分量在长尾上偶尔会出现极大值，过大的激活是 routing outlier 的源头。
* **实现：直接把 SwiGLU 的 linear 部分 clamp 到 $`[-10, 10]`$，gate 部分上限 10。** `config.json` 里 `swiglu_limit=10.0`：

```python
if self.swiglu_limit > 0:
    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
    gate = torch.clamp(gate, max=self.swiglu_limit)
```

> **参考**：OpenAI 2025 在 gpt-oss 中的实践也使用了 SwiGLU Clamping，对最终性能没有可观测的损害。

---

#### 3.1.4 预训练评估结果

| Category | Benchmark (Metric) | # Shots | DeepSeek-V3.2 Base | DeepSeek-V4-Flash Base | DeepSeek-V4-Pro Base |
| --- | --- | --- | --- | --- | --- |
|  | Architecture | - | MoE | MoE | MoE |
|  | # Activated Params | - | 37B | 13B | 49B |
|  | # Total Params | - | 671B | 284B | 1.6T |
| **World Knowl.** | AGIEval (EM) | 0-shot | 80.1 | **82.6** | **83.1** |
|  | MMLU (EM) | 5-shot | 87.8 | **88.7** | **90.1** |
|  | MMLU-Redux (EM) | 5-shot | 87.5 | **89.4** | **90.8** |
|  | MMLU-Pro (EM) | 5-shot | 65.5 | **68.3** | **73.5** |
|  | MMMLU (EM) | 5-shot | 87.9 | **88.8** | **90.3** |
|  | C-Eval (EM) | 5-shot | 90.4 | **92.1** | **93.1** |
|  | CMMLU (EM) | 5-shot | 88.9 | **90.4** | **90.8** |
|  | MultiLoKo (EM) | 5-shot | 38.7 | **42.2** | **51.1** |
|  | Simple-QA verified (EM) | 25-shot | 28.3 | **30.1** | **55.2** |
|  | SuperGPQA (EM) | 5-shot | 45.0 | **46.5** | **53.9** |
|  | FACTS Parametric (EM) | 25-shot | 27.1 | **33.9** | **62.6** |
|  | TriviaQA (EM) | 5-shot | 83.3 | 82.8 | **85.6** |
| **Lang. & Reas.** | BBH (EM) | 3-shot | **87.6** | 86.9 | **87.5** |
|  | DROP (F1) | 1-shot | **88.2** | **88.6** | **88.7** |
|  | HellaSwag (EM) | 0-shot | **86.4** | 85.7 | **88.0** |
|  | WinoGrande (EM) | 0-shot | 78.9 | **79.5** | **81.5** |
|  | CLUEWSC (EM) | 5-shot | **83.5** | 82.2 | **85.2** |
| **Code & Math** | BigCodeBench (Pass@1) | 3-shot | **63.9** | 56.8 | **59.2** |
|  | HumanEval (Pass@1) | 0-shot | 62.8 | **69.5** | **76.8** |
|  | GSM8K (EM) | 8-shot | **91.1** | 90.8 | **92.6** |
|  | MATH (EM) | 4-shot | **60.5** | 57.4 | **64.5** |
|  | MGSM (EM) | 8-shot | 81.3 | **85.7** | **84.4** |
|  | CMath (EM) | 3-shot | **92.6** | **93.6** | 90.9 |
| **Long Context** | LongBench-V2 (EM) | 1-shot | 40.2 | **44.7** | **51.5** |

*（注：上表为统一 internal framework、严格一致设置下的 base model 对比。）*

**核心结论与分析**

重点看 **V4-Flash-Base（13B 激活）** 对比 **V3.2-Base（37B 激活）**，V4-Flash-Base 在绝大多数 benchmark 上反超 V3.2-Base，尤其世界知识和长上下文任务（V4-Pro-Base 再上一个台阶）：

* **参数效率大幅提升**：V4-Flash-Base 仅用 **13B 激活参数**就能在 MMLU、C-Eval、Simple-QA、HumanEval、LongBench-V2 等多个维度上超越 V3.2-Base（37B 激活）。这说明 V4 的**架构和数据改进对参数效率有实质性提升**，进步不仅来自于 token scaling（数据量缩放）。
* **纯 Reasoning 维度的劣势**：但 V4-Flash 在 BBH、GSM8K、MATH 这几项上略输 V3.2，说明在纯 reasoning 上 13B 激活仍是劣势（**需要后续训练增强**）。
* **知识储备发生质变**：V4-Pro-Base（49B 激活）比 V3.2-Base（37B 激活）只多了 32% 的激活参数，但在 Simple-QA verified 从 28.3 拉高到 55.2、FACTS Parametric 从 27.1 拉高到 62.6，即**世界知识维度上有 2 倍以上的飞跃**。这说明 V4 的多语言 / 长文档 / 科学论文数据收集对知识储备带来了质的突变。

---

### 3.2 Post-Training

V4 在 post-training 阶段的关键改进：**把 V3.2 时代的 mixed RL 阶段替换成两阶段 On-Policy Distillation (OPD)**：

* **Specialist Training**：针对 math / code / agent / instruction following 等每个领域，独立训练一个 expert 模型。
* **On-Policy Distillation**：学生模型在自己采样的 trajectory 上，从 10+ teacher 全词表 logits 分布拟合 reverse KL。

整体架构与流程如下：

```text
[Base Model] ──(各 domain 独立训练)──> [ [Math Specialist], [Code Specialist], ..., [Agent Specialist] ]
                                                       │
                                            (Multi-teacher On-Policy Distillation)
                                                       │
                                                       ▼
                                        [Final Unified Model (V4-Pro / V4-Flash)]
```

每个 specialist 训练仍然使用 SFT + RL (GRPO) 的传统路径，**关键改进在于合并阶段：OPD（对比传统的 weight averaging、mixed RL）**。

#### 3.2.1 Specialist Training

##### Reasoning Efforts：三档思考预算

V4-Pro 和 V4-Flash 都支持 Non-think / Think High / Think Max 三档思考模式。实现方式是在 specialist training 阶段训练 / 储存三个不同 RL 配置的 checkpoint（不是通过开关 / 提示词实现的）：

| 模式 | 特点 | 典型场景 | 格式 |
| --- | --- | --- | --- |
| **Non-think** | 直接基于习惯 / 简单规则响应 | 日常任务、应急、低风险决策 | `</think> summary` |
| **Think High** | 显式逻辑分析 | 复杂问题、规划、中风险决策 | `<think> thinking </think> summary` |
| **Think Max** | 把 reasoning 推到极限 | 探索模型 reasoning 边界 | system prompt 强制思考指令 + `<think> thinking </think> summary` |

「Think Max」会在 system prompt 前面注入**强制深思指令**：

> **Injected Instruction**
> Reasoning Effort: Absolute maximum with no shortcuts permitted.
> You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.
> Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.

在 RL 配置上三档思考之间的区别：

* 每档对应不同的 **length penalty** 和 **context window**（论文没给训练时的具体数字）。
* Evaluation 时为对应推理预算，三档分别用 **8K / 128K / 384K context window**。

##### Generative Reward Model (GRM)

* **动机：传统 scalar reward model 在 hard-to-verify 任务上遇到瓶颈。**
    * **易验证任务**（数学、代码）直接用 rule-based verifier 就足够。
    * **难验证任务**（写作、规划、推理）训练 scalar reward model（RLHF 路线）需要大量人类标注。
* **实现：让 actor 自己当 reward model，joint optimization。** DeepSeek V4 不再训练独立的 scalar reward model，**用 actor network 自己充当 Generative Reward Model (GRM)**：actor 一边生成 response，一边学评估自己的 trajectory，评估能力（judging proficiency）和生成能力（generative capability）联合优化。该设计的好处：
    * 模型 internal reasoning 直接进入评估流程，scoring 更鲁棒。
    * 只需要少量 diverse human annotation，利用模型泛化能力。
    * 统一了 actor / reward model 的角色，避免训练两个模型的资源开销和分布漂移。

##### Tool-Call Schema

V4 引入了一个新 tool-call schema，用 special token `<|DSML|>` 包裹 tool 调用，避开 JSON 输出的脆弱性：

> **Tool Call Schema 说明文档**
>
> ```text
> ## Tools
>
> You have access to a set of tools to help answer the user's question. You can invoke tools by writing a `<|DSML|tool_calls>` block like the following:
>
> <|DSML|tool_calls>
> <|DSML|invoke name="$TOOL_NAME">
> <|DSML|parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE
> </|DSML|parameter>
> ...
> </|DSML|invoke>
> <|DSML|invoke name="$TOOL_NAME2">
> ...
> </|DSML|invoke>
> </|DSML|tool_calls>
>
> String parameters should be specified as is and set string="true". For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set string="false".
>
> If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.
>
> Otherwise, output directly after </think> with tool calls or final response.
>
> ### Available Tool Schemas
>
> {Tool Definition...}
>
> You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
> ```

实验显示 **XML 比 JSON 在 tool-call 错误率上更低，escape 失败更少**。

##### Interleaved Thinking

V3.2 在 agent 场景下保留 reasoning trace 跨 tool result，但 user message 一来就清空所有 reasoning。V4 利用 1M context 把策略改成两套：

* **Tool-Calling Scenarios**：所有 reasoning content 完整保留，跨 user message 边界也保留，让模型在 long-horizon agent 任务上有连贯的 chain-of-thought。
* **General Conversation**：保持 V3.2 策略，新 user message 一来就清空 reasoning，保持上下文紧凑。

两类场景对应的 Thinking 管理示意（左列是 tool-calling 场景：每一轮的 thinking + tool call + tool result 都在 input 里持续累积；右列是普通对话场景：上一轮的 thinking 在新 user message 到来时被丢弃）：

| Tool-Calling Scenarios: Thinking with tools | General Conversation: Thinking without tools |
| --- | --- |
| 每一轮的思维链与工具交互持续保留在上下文中 | 新一轮对话开始时，上一轮的 `<think>` 过程被清除 |

##### Quick Instruction：避开辅助任务的额外 prefilling

Chatbot 场景下，模型经常要做一堆辅助任务（判断要不要触发 web search、识别用户意图、生成搜索 query 等）。传统做法是用一个独立的小模型，但小模型不能复用主模型的 KV cache，每次都要重新 prefilling。

**V4 的实现**：直接在主模型的输入序列后追加 special token 触发对应辅助任务，复用已有 KV cache，把 TTFT (time to first token) 压到最低。对应的 special token 包括：

```text
<|action|>
<|title|>
<|query|>
<|authority|>
<|domain|>
<|extracted_url|>
<|read_url|>
```

---

#### 3.2.2 On-Policy Distillation：把 10+ specialists 合成一个模型

推荐前置阅读：📄 *On-Policy Distillation 是什么？工作脉络梳理*

##### 替换 Mixed RL 到 OPD

* **动机：传统多专家融合方法存在的问题。**
    * **Weight averaging / Model soup**：直接平均参数，不同 specialist 之间的能力会相互稀释 / 矛盾。
    * **Mixed RL**：用一个统一的 reward 函数同时优化所有 domain，容易 reward hacking 且不同 domain 互相干扰。
* **实现：full-vocabulary On-Policy Distillation。** V4 沿用了 Thinking Machines 的 OPD 框架，目标函数：

```math
\mathcal{L}_{\text{OPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}\left(\pi_\theta \parallel \pi_{E_i}\right)
```

* $`\pi_\theta`$ 是 student，也就是最终统一模型。
* $`\pi_{E_i}`$ 是第 $`i`$ 个 domain expert (teacher)。
* $`w_i`$ 是分配给该 expert 的重要性权重。
* $`D_{\text{KL}}`$ 是 reverse KL（注意是 student 对 teacher 的 KL）。
* 计算 reverse KL 时，trajectory 来自 student 自己采样，这就是「on-policy」的来源。

两个关键点需要注意（在 📄 *On-Policy Distillation 是什么？工作脉络梳理* 有详细解析）：

* **为什么用 reverse KL 而不是 forward KL？** 参考 Thinking Machines 的 blog：reverse KL 是 mode-seeking 的，强制 student 收敛到 teacher 的某个具体行为模式而非分散在多个次优模式上；同时 reverse KL 是 unhackable 的，因为 teacher 的低 KL 总对应 desired behavior 的高概率。

  *（注：左图 Forward KL: Mean-Seeking Behaviour 呈现平缓的双峰包络分布；右图 Reverse KL: Mode-Seeking Behaviour 呈现精准拟合单峰的分布。）*
* **为什么用 full-vocabulary KL 而非 token-level estimate？** Full-vocabulary 把 teacher 在每个位置上的整条 logit 分布都对齐过来，gradient estimate variance 显著降低、训练更稳定。代价是计算 / 显存压力大（尤其 V4 是 10+ teachers 场景）。详细参考 📄 *On-Policy Distillation 是什么？工作脉络梳理*。

##### OPD 工程实现

上面说了由于 10+ teacher 使用 full-vocabulary KL，储存全 logit 显存压力大，因此 V4 的实现链路：

1. **Teacher weight centralized storage**：所有 teacher weight 卸到中心化分布式存储，按需 load + ZeRO-like sharding，缓解 I/O 和 DRAM 压力。
2. **只 cache teacher 的 last-layer hidden state**：只缓存中央 buffer 里的 last-layer hidden state，训练时 retrieve 这些 hidden state，临时跑对应的 prediction head 重建 logits（类似 MLA）。
3. **Order training samples by teacher index**：调度上把同一 teacher 的样本聚到一起 dispatch，保证每个 distinct teacher head 在一个 mini-batch 内只 load 一次，任意时刻 device memory 最多一个 teacher head。
4. **TileLang kernel 算 KL**：student / teacher logits 之间的精确 KL divergence 由优化的 TileLang kernel 完成，加速计算并避开动态显存分配。
5. **Background async I/O**：所有参数 / hidden state 的加载和卸载在 critical path 之外异步完成，不阻塞 training compute。

**本质优化思路是用调度换显存**：teacher 多但每次用才加载 head 计算完整 logits，hidden state 比 logits 小两个数量级所以可以缓存，prediction head 即用即 load。

---

#### 3.2.3 RL / OPD Infra

除了 OPD Scheduling，V4 在 post-training infra 还需要注意以下工程细节：

* **Rollout 集成 FP4 (MXFP4)**：rollout 和所有 inference-only forward pass（包括 teacher 和 reference model）都直接用 FP4，降低 rollout 阶段的显存和 latency。
* **Preemptible & Fault-Tolerant Rollout**：每个 generation request 维护 token-level Write-Ahead Log (WAL)，每生成一个 token 立刻 append。preemption 时 pause inference engine 保存 KV cache，resumption 时用 WAL + KV cache 继续解码生成。
* **Million-Token Context RL 拆分加载**：rollout data 被拆成 metadata + 重 per-token field，后者走 shared-memory loader 避免节点内重复、用完立刻释放。on-device mini-batch 数动态调节，平衡吞吐与 I/O。token 粒度 WAL 保证重启不 regeneration from scratch（否则有 length bias，短响应更容易存活让模型偏向短输出）。
* **DeepSeek Elastic Compute (DSec)**：DeepSeek Elastic Compute (DSec) 是一套 Rust 写的沙箱平台（Function Call / Container / microVM / fullVM），底层用 3FS + EROFS layered storage，是 V4 agentic RL 训练能 scale 的核心基础设施。实现大规模管理 RL sandbox，每个 sandbox 留存 trajectory log，可以做确定性 replay 和 fast-forwarding（preemption 后用 cached 结果加速恢复）。

---

### 3.3 实验

DeepSeek V4 的 Evaluation 主要分为两类：

* **标准 Benchmark**：公开数据集
* **Real-world Tasks**：
    * **中文写作 Chinese Writing**：覆盖功能性写作（报告、邮件、文案等）和创意写作（小说、散文、诗歌等），主要对标 Gemini-3.1-Pro，从指令遵循和写作质量两个维度评估。
    * **搜索 Search**：DeepSeek Chatbot 的联网问答能力，分为 RAG（非思考模式，一次性检索）和 Agentic Search（思考模式，多轮主动调用搜索工具）。评测涵盖客观问答（查事实、找实体）和主观问答（分析、对比、推荐、规划）。
    * **白领任务 White-Collar Task**：模拟企业办公场景的 30 个复杂中文专业任务，跨 13 个行业，涵盖信息分析、文档生成和文档编辑三类。由人工盲评，从任务完成度、指令遵循、内容质量、排版美观四个维度对比 Claude Opus 4.6-Max。
    * **代码智能体 Code Agent**：从 DeepSeek 内部研发工作中收集的 30 个编程任务，涵盖功能开发、bug 修复、重构、诊断，技术栈包括 PyTorch、CUDA、Rust、C++。在多轮工具调用环境下评测，对标 Claude 系列模型。

#### 3.3.1 标准 Benchmark：V4-Pro-Max 与闭源 / 开源旗舰对比

下表是 V4-Pro-Max 与当前 SoTA 开源 / 闭源模型的对比：

| Category | Benchmark (Metric) | Opus-4.6 Max | GPT-5.4 xHigh | Gemini-3.1-Pro High | K2.6 Thinking | GLM-5.1 Thinking | DS-V4-Pro Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knowledge & Reasoning** | MMLU-Pro (EM) | [89.1] | 87.5 | **91.0** | 87.1 | 86.0 | 87.5 |
|  | SimpleQA-Verified (Pass@1) | 46.2 | 45.3 | **75.6** | 36.9 | 38.1 | [57.9] |
|  | Chinese-SimpleQA (Pass@1) | 76.4 | 76.8 | **85.9** | 75.9 | 75.0 | [84.4] |
|  | GPQA Diamond (Pass@1) | 91.3 | [93.0] | **94.3** | 90.5 | 86.2 | 90.1 |
|  | HLE (Pass@1) | [40.0] | 39.8 | **44.4** | 36.4 | 34.7 | 37.7 |
|  | LiveCodeBench (Pass@1) | 88.8 | - | [91.7] | 89.6 | - | **93.5** |
|  | Codeforces (Rating) | - | [3168] | 3052 | - | - | **3206** |
|  | HMMT 2026 Feb (Pass@1) | [96.2] | **97.7** | 94.7 | 92.7 | 89.4 | 95.2 |
|  | IMOAnswerBench (Pass@1) | 75.3 | **91.4** | 81.0 | 86.0 | 83.8 | [89.8] |
|  | Apex (Pass@1) | 34.5 | [54.1] | **60.9** | 24.0 | 11.5 | 38.3 |
|  | Apex Shortlist (Pass@1) | [85.9] | 78.1 | [89.1] | 75.5 | 72.4 | **90.2** |
| **Long** | MRCR 1M (MMR) | **92.9** | - | 76.3 | - | - | [83.5] |
|  | CorpusQA 1M (ACC) | **71.7** | - | 53.8 | - | - | [62.0] |
| **Agentic** | Terminal Bench 2.0 (Acc) | 65.4 | **75.1** | [68.5] | 66.7 | 63.5 | 67.9 |
|  | SWE Verified (Resolved) | [80.8] | - | [80.6] | 80.2 | - | [80.6] |
|  | SWE Pro (Resolved) | 57.3 | [57.7] | 54.2 | **58.6** | [58.4] | 55.4 |
|  | SWE Multilingual (Resolved) | [77.5] | - | - | [76.7] | 73.3 | [76.2] |
|  | BrowseComp (Pass@1) | 83.7 | 82.7 | **85.9** | 83.2 | 79.3 | 83.4 |
|  | HLE w/ tools (Pass@1) | [53.1] | 52.0 | 51.6 | **54.0** | 50.4 | 48.2 |
|  | GDPval-AA (Elo) | 1619 | **1674** | 1314 | 1482 | 1535 | 1554 |
|  | MCPAtlas Public (Pass@1) | [73.8] | 67.2 | 69.2 | 66.6 | 71.8 | 73.6 |
|  | Toolathlon (Pass@1) | 47.2 | **54.6** | 48.8 | 50.0 | 40.7 | [51.8] |

*（注：表中加粗代表全场最高，方括号 `[...]` 标记的数字代表在特定分类下的领先表现。）*

**实验结论要点分析**

* **Reasoning 维度 Coding 竞赛水平领先**：LiveCodeBench 与 Codeforces Rating、Apex Shortlist 均为 SOTA 水平。论文提到 Codeforces leaderboard 上 V4-Pro-Max 排到人类选手第 23 位。
* **Agentic 任务落后**：技术报告认为 Terminal Bench 的环境本身有问题，其次就是在 SWE Pro 上水平低于 Kimi K2.6 和 GLM 5.1。
* **数学能力接近 SOTA**：HMMT 2026 Feb 为 95.2，落后最佳的 GPT-5.4 xHigh (97.7) 和 Opus-4.6 Max (96.2) 但都在 2~3 分之内，IMOAnswerBench 89.8 也接近 GPT-5.4 的 91.4。
* **知识维度落后闭源模型、优于开源模型**：SimpleQA-Verified 57.9 vs Gemini-3.1-Pro 75.6，差距明显。但相对 K2.6 Thinking (36.9) 和 GLM-5.1 Thinking (38.1) 这两个开源模型 V4-Pro-Max 仍然有很大优势。这意味着 V4 在开源界已经把 SimpleQA 拉到一个新台阶，但和闭源 frontier 仍有差距。
* **Long-Context 表现中等**：MRCR 1M 上 V4-Pro-Max 83.5 落后 Opus-4.6 Max 的 92.9，但比 Gemini-3.1-Pro 的 76.3 高出约 7 分。说明虽然 DeepSeek V4 支持 1M 上下文，但是**召回能力距离 SOTA 仍有距离**。
* **HLE w/ tools 表现不佳**：DeepSeek V4 在「工具 + 长链推理」组合上 V4 仍有提升空间。

---

#### 3.3.2 Reasoning Effort 的边际收益

DeepSeek V4 对三档思考预算（8K、128K、384K）进行消融实验，比对不同思考预算能带来多少提升：

| Category | Benchmark (Metric) | DeepSeek-V4-Flash Non-Think | DeepSeek-V4-Flash High | DeepSeek-V4-Flash Max | DeepSeek-V4-Pro Non-Think | DeepSeek-V4-Pro High | DeepSeek-V4-Pro Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knowledge & Reasoning** | MMLU-Pro (EM) | 83.0 | 86.4 | 86.2 | 82.9 | 87.1 | 87.5 |
|  | SimpleQA-Verified (Pass@1) | 23.1 | 28.9 | 34.1 | 45.0 | 46.2 | 57.9 |
|  | Chinese-SimpleQA (Pass@1) | 71.5 | 73.2 | 78.9 | 75.8 | 77.7 | 84.4 |
|  | GPQA Diamond (Pass@1) | 71.2 | 87.4 | 88.1 | 72.9 | 89.1 | 90.1 |
|  | HLE (Pass@1) | 8.1 | 29.4 | 34.8 | 7.7 | 34.5 | 37.7 |
|  | LiveCodeBench (Pass@1-COT) | 55.2 | 88.4 | 91.6 | 56.8 | 89.8 | 93.5 |
|  | Codeforces (Rating) | - | 2816 | 3052 | - | 2919 | 3206 |
|  | HMMT 2026 Feb (Pass@1) | 40.8 | 91.9 | 94.8 | 31.7 | 94.0 | 95.2 |
|  | IMOAnswerBench (Pass@1) | 41.9 | 85.1 | 88.4 | 35.3 | 88.0 | 89.8 |
|  | Apex (Pass@1) | 1.0 | 19.1 | 33.0 | 0.4 | 27.4 | 38.3 |
|  | Apex Shortlist (Pass@1) | 9.3 | 72.1 | 85.7 | 9.2 | 85.5 | 90.2 |
| **Long** | MRCR 1M (MMR) | 37.5 | 76.9 | 78.7 | 44.7 | 83.3 | 83.5 |
|  | CorpusQA 1M (ACC) | 15.5 | 59.3 | 60.5 | 35.6 | 56.5 | 62.0 |
| **Agentic** | Terminal Bench 2.0 (Acc) | 49.1 | 56.6 | 56.9 | 59.1 | 63.3 | 67.9 |
|  | SWE Verified (Resolved) | 73.7 | 78.6 | 79.0 | 73.6 | 79.4 | 80.6 |
|  | SWE Pro (Resolved) | 49.1 | 52.3 | 52.6 | 52.1 | 54.4 | 55.4 |
|  | SWE Multilingual (Resolved) | 69.7 | 70.2 | 73.3 | 69.8 | 74.1 | 76.2 |
|  | BrowseComp (Pass@1) | - | 53.5 | 73.2 | - | 80.4 | 83.4 |
|  | HLE w/ tools (Pass@1) | - | 40.3 | 45.1 | - | 44.7 | 48.2 |
|  | MCPAtlas Public (Pass@1) | 64.0 | 67.4 | 69.0 | 69.4 | 74.2 | 73.6 |
|  | GDPval-AA (Elo) | - | - | 1395 | - | - | 1554 |
|  | Toolathlon (Pass@1) | 40.7 | 43.5 | 47.8 | 46.3 | 49.0 | 51.8 |

**结论要点分析**

* **Reasoning 任务对思考预算极度敏感**：与 OpenAI o1 结论一致，即 reasoning 性能在很大程度上由 test-time compute 决定。HLE 从 Non-Think 到 High 直接从 7.7 涨到 34.5（约 4.5x），LiveCodeBench 从 56.8 到 89.8。
* **Knowledge 任务对思考预算不敏感**：MMLU-Pro Non-Think 已经 82.9，High 增加到 87.1，再到 Max 只到 87.5。这是因为 knowledge tasks 主要靠 parametric memory（参数记忆），思考预算影响有限。
* **High $`\rightarrow`$ Max 的边际收益小**：Codeforces 从 High 到 Max 是 $`2919 \rightarrow 3206`$、HLE 是 $`34.5 \rightarrow 37.7`$。Max 的代价是 384K context，边际收益逐步降低。
* **V4-Flash-Max 在 reasoning 任务上接近 V4-Pro-Max**：LiveCodeBench 91.6 vs 93.5、Codeforces 3052 vs 3206、HLE 34.8 vs 37.7。这意味着对 reasoning 主导的任务，13B 激活的 Flash 配长 thinking 已经够用，主要差距来自于 knowledge 任务（SimpleQA、MMLU-Pro）上，这部分是参数容量决定的。

---

#### 3.3.3 Long-Context：1M token 上的真实表现

V4 系列原生 1M context，但 1M 上的实际 retrieval / understanding 能力测评看 MRCR 8-needle 的曲线：

| 输入长度 | V4-Pro-Max | V4-Flash-Max |
| --- | --- | --- |
| **8K** | 0.90 | 0.91 |
| **16K** | 0.85 | 0.84 |
| **32K** | **0.94** | 0.87 |
| **64K** | 0.90 | 0.85 |
| **128K** | 0.92 | 0.87 |
| **256K** | 0.82 | 0.76 |
| **512K** | 0.66 | 0.60 |
| **1024K** | 0.59 | 0.49 |

*（注：MRCR 8-needle 评测曲线随上下文增长召回率下降；横轴为 Input Tokens，纵轴为 Average MMR。）*

**核心结论**：**128K 以内 retrieval 表现很好（均在 0.85 以上），256K 开始可见衰减，1M 退化到约 0.59**。这和 MRCR 1M 上 V4-Pro-Max 83.5 落后 Opus-4.6 Max 的 92.9 结论一致：**V4 的 1M 上下文绝不是 perfect retrieval**。

---

#### 3.3.4 Real-World Tasks：真实工作场景测评

为了解决"标准 benchmark 和真实使用差距"的问题，V4 在 4 个维度自建数据上测试真实场景的 pairwise 评测（win vs. lose）：

* **中文写作 Chinese Writing**：覆盖功能性写作（报告、邮件、文案等）和创意写作（小说、散文、诗歌等），主要对标 Gemini-3.1-Pro，从指令遵循和写作质量两个维度评估。
* **搜索 Search**：DeepSeek Chatbot 的联网问答能力，分为 RAG（非思考模式，一次性检索）和 Agentic Search（思考模式，多轮主动调用搜索工具）。评测涵盖客观问答（查事实、找实体）和主观问答（分析、对比、推荐、规划）。
* **白领任务 White-Collar Task**：模拟企业办公场景的 30 个复杂中文专业任务，跨 13 个行业，涵盖信息分析、文档生成和文档编辑三类。由人工盲评，从任务完成度、指令遵循、内容质量、排版美观四个维度对比 Claude Opus 4.6-Max。
* **代码智能体 Code Agent**：从 DeepSeek 内部研发工作中收集的 30 个编程任务，涵盖功能开发、bug 修复、重构、诊断，技术栈包括 PyTorch、CUDA、Rust、C++。在多轮工具调用环境下评测，对标 Claude 系列模型。

##### 中文写作 Chinese Writing

* **功能写作 (Functional writing)**：V4-Pro vs Gemini-3.1-Pro，胜率 **62.7% vs 34.1%**。论文给出的解释是 Gemini 偶尔会用自己的 stylistic preference 覆盖用户显式要求，V4 在 instruction-following 上指令遵循更好。
* **创意写作 (Creative writing)**：V4-Pro 在 instruction following 上 60.0% 胜率，writing quality 上 77.5% 胜率。但在「最具挑战的 prompt」的多约束 / 多轮场景，**Claude Opus 4.5 的胜率 52.0% vs V4-Pro 的 45.9%**，仍是 Claude 能力更强。

##### 搜索 Search

DeepSeek 的 web / app 上 non-think 用 RAG，think 用 agentic search。

* **RAG 模式 (V4-Pro vs V3.2)**：内部综合评估 V4 win 28.1%、V3.2 win 10.4%、tie 61.5%。**V4 在 single-value search 和 planning & strategy 上提升最多**，但在 comparison 和 recommendation 类任务上 V3.2 仍有竞争力。

> **Comparative Evaluation of DeepSeek-V4-Pro and DeepSeek-V3.2 on Search Q&A Tasks.**
>
> | Category | Subcategory | # | V4 win | V3.2 win | tie | V4% | V3.2% | tie% |
> | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
> | **Objective Q&A (客观问答)** | Single-value Search (单值信息查找) | 95 | 36 | 10 | 49 | 37.9 | 10.5 | 51.6 |
> | | Entity Search (实体信息查找) | 99 | 24 | 7 | 68 | 24.2 | 7.1 | 68.7 |
> | | Enumerative Search (枚举型信息查找) | 95 | 19 | 8 | 68 | 20.0 | 8.4 | 71.6 |
> | | **Subtotal (小计)** | **289** | **79** | **25** | **185** | **27.3** | **8.7** | **64.0** |
> | **Subjective Q&A (主观问答)** | Causal Analysis (原因分析) | 100 | 28 | 5 | 67 | 28.0 | 5.0 | 67.0 |
> | | Comparison (对比) | 96 | 28 | 20 | 48 | 29.2 | 20.8 | 50.0 |
> | | Advice Seeking (寻求建议) | 92 | 23 | 8 | 61 | 25.0 | 8.7 | 66.3 |
> | | Recommendation (推荐) | 95 | 26 | 19 | 50 | 27.4 | 20.0 | 52.3 |
> | | Planning & Strategy (攻略计划) | 92 | 32 | 11 | 49 | 34.8 | 12.0 | 53.3 |
> | | Opinion & Evaluation (评价看法) | 96 | 30 | 8 | 58 | 31.2 | 8.3 | 60.4 |
> | | Trend Analysis (趋势分析) | 96 | 23 | 3 | 70 | 24.0 | 3.1 | 72.9 |
> | | **Subtotal (小计)** | **667** | **190** | **74** | **403** | **28.5** | **11.1** | **60.4** |
> | **TOTAL (总计)** | | **956** | **269** | **99** | **588** | **28.1** | **10.4** | **61.5** |

* **Agentic Search vs RAG**：agentic search 总胜率 61.7%、RAG 18.3%、tie 20.0%，agentic search 全方位领先。代价是 agentic 平均 16.2 个 tool call、prefill 13649 tokens、output 1526 tokens；RAG 是 prefill 10453 tokens、output 1308 tokens。**Agentic 成本略高但效果显著更好**。

> **Agentic Search vs. Retrieval Augmented Search for DeepSeek-V4-Pro.**
>
> | Difficulty | Category | # | Agent Win | RAG Win | Tie | Agent% | RAG% | Tie% |
> | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
> | **Easy** | Objective Q&A (客观问答) | 196 | 110 | 43 | 43 | 56.1 | 21.9 | 21.9 |
> | | Subjective Q&A (主观问答) | 321 | 198 | 56 | 67 | 61.7 | 17.4 | 20.9 |
> | **Hard** | Objective Q&A (客观问答) | 168 | 102 | 33 | 33 | 60.7 | 19.6 | 19.6 |
> | | Subjective Q&A (主观问答) | 184 | 126 | 27 | 31 | 68.5 | 14.7 | 16.8 |
> | **Total (总计)** | | **869** | **536** | **159** | **174** | **61.7** | **18.3** | **20.0** |

##### 中文白领任务 White-Collar Task

V4-Pro-Max vs Opus-4.6-Max，跨 13 个行业、30 个复杂中文专业（公司）任务：

* **总体胜率**：53% win / 10% tie / 37% lose。
* **分项胜率**：分析 55%、生成 52%、编辑 47%。
* **分维度评分**：
    * Task Completion（任务完成度）：98.32 vs 96.68 (**V4 略优**)
    * Content Quality（内容质量）：83.32 vs 78.00 (**V4 优**)
    * Instruction Following（指令遵循）：87.78 vs 88.76 (**V4 略输**)
    * Formatting Aesthetics（格式排版美观）：76.68 vs 72.68 (**V4 优**)
    * Overall（整体评价）：86.52 vs 84.06 (**V4 优**)

> ⚠️ **主要痛点与短板：** V4 的弱项在 **Instruction Following**。偶尔会忽略具体 formatting（格式）约束，且在**长文本压缩成精炼摘要**的能力上有所欠缺。此外，PPT 类视觉设计也仍有进步空间。

##### 代码智能体 Code Agent

DeepSeek 自己用 ~200 个真实内部研发任务做 benchmark，筛选出 30 个用作 evaluation set（评估集）：

| Model | Haiku 4.5 | Sonnet 4.5 | DeepSeek-V4-Pro-Max | Opus 4.5 | Opus 4.5 Thinking | Opus 4.6 Thinking |
| --- | --- | --- | --- | --- | --- | --- |
| **Pass Rate (%)** | 13 | 47 | **67** | 70 | 73 | **80** |

* **表现分析**：V4-Pro-Max 取得 **67% 的通过率**，显著超过 Sonnet 4.5 (47%)，非常接近 Opus 4.5 (70%)，但仍然落后于最顶尖的 Opus 4.6 Thinking (80%)。

**开发者真实调研反馈**：DeepSeek 内部对 85 个开发者做了调研，结果表明：

* **52%** 完全愿意把 V4-Pro 作为日常代码主力。
* **39%** 倾向于愿意。
* **不到 9%** 说不。

> **开发者普遍反馈**：V4 在大多数任务上的结果令人满意，但在处理 **trivial mistakes（低级错误）**、**模糊 prompt 误解** 以及 **occasional over-thinking（偶尔过度思考）** 上仍有改进空间。

