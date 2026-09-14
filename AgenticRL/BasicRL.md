# 强化学习基础（面向 LLM / Agentic RL 的视角）

本文围绕 **PPO** 展开，把会反复用到的基础概念（策略梯度、Advantage、重要性采样、Clip、GAE、KL、Actor-Critic、以及 GRPO / RLHF 视角下的变体）系统串起来。重点在「为什么这样设计」，而不仅仅是公式。

---

## 0. 记号约定

- 状态 $s_t$ ，动作 $a_t$ ，奖励 $r_t$ ，折扣 $\gamma \in [0,1]$
- 策略 $\pi_\theta(a|s)$ ：参数为 $\theta$ ，输出动作分布
- 轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, \dots)$
- 回报 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$
- 状态值函数 $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t=s]$
- 动作值函数 $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]$
- 优势函数 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$

在 LLM 场景下： $s_t$ 是已生成的 prompt + tokens， $a_t$ 是下一个 token， $r_t$ 通常只在序列末尾由 Reward Model 给出。

---

## 1. 策略梯度（Policy Gradient）：一切的起点

目标：最大化期望回报

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]
```

**策略梯度定理**：

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \Psi_t \right]
```

### 1.1 推导：从期望回报到策略梯度（小白友好版）

我们要证明的目标公式：

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
```

> 用人话说：「在很多条采样到的轨迹上，把每个时刻的 $\nabla \log \pi$ 用整条轨迹的回报加权平均，就是 $\nabla J$ 。」

为什么有用？因为右边是一个 **期望**，可以用采样近似（跑游戏 $N$ 次取平均）；而左边是一个抽象的梯度，看不出怎么算。

下面分 6 小步推导。**每一步都只用了高中/大一的微积分**。

---

#### Step 1：把期望写成"概率 × 数值"的求和

期望就是「概率加权平均」。把 $J$ 展开：

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_{\tau} p_\theta(\tau) \, R(\tau)
```

> 如果你看到的是 $\int p_\theta(\tau) R(\tau) d\tau$ ，那只是连续版本，意思一样。下面我都用 $\sum$ 写，更直观。

记号解释：
- $p_\theta(\tau)$ ：在策略 $\pi_\theta$ 下，**采样到轨迹 $\tau$ 的概率**
- $R(\tau)$ ：这条轨迹的总回报（一个数）
- $J(\theta)$ ：所有可能轨迹的"概率 × 回报"加起来

「期望回报」= 「每条轨迹的发生概率 × 它的得分」之和。

#### Step 2：对 $\theta$ 求梯度，把 $\nabla$ 挪进求和号里

```math
\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} p_\theta(\tau) R(\tau) = \sum_{\tau} \nabla_\theta p_\theta(\tau) \cdot R(\tau)
```

为什么 $R(\tau)$ 不动？因为 **奖励是环境给的，不依赖参数 $\theta$**。 $\theta$ 改变的只是「采样到 $\tau$ 的概率」，而不是「这条 $\tau$ 值多少分」。

> **为什么我们一定要用「采样取平均」来算这个梯度？**
> 因为轨迹空间是天文数字级别的——每一步动作都有多种选择，组合起来 $\tau$ 的数量指数爆炸，**穷举求和 $\sum_\tau$ 根本不可能**。唯一实际可行的办法是**蒙特卡洛估计（Monte Carlo estimation）**：如果求和能写成期望 $\mathbb{E}_{x \sim p}[f(x)]$ 的形式，我们只需按 $p$ 采样 $N$ 条轨迹，对 $f(x_i)$ 取算术平均即可近似：
>
> ```math
> \mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i), \quad x_i \sim p
> ```
>
> 采样频率本身就自动替代了概率权重 $p(x)$，不需要显式算出每条轨迹的概率值。所以整个 REINFORCE 推导的动机就是：**把梯度变成期望，好让我们能采样估计**。

**遇到的麻烦**： $\nabla_\theta p_\theta(\tau)$ 不再是一个「概率 × 数值」的形式（前面那个因子不是概率了）。我们无法用「采样取平均」来估计它。

> **为什么"不是概率"就不能采样？**
> 蒙特卡洛估计的标准长相是 $\mathbb{E}_{x\sim p}[f(x)] = \sum_x p(x) f(x)$ ，要求前面的加权因子 $p(x)$ 是**合法概率**（非负、和为 1）。这样按 $p$ 采样 $N$ 个 $x_i$ ，**采样频率本身就替你完成了"乘以 $p(x)$ "**，对 $f(x_i)$ 取平均即可。
>
> 而 $\nabla_\theta p_\theta(\tau)$ 是个**梯度**：可以为负，且 $\sum_\tau \nabla_\theta p_\theta(\tau) = \nabla_\theta \sum_\tau p_\theta(\tau) = \nabla_\theta 1 = 0$ ——根本不是分布，没法"按它采样"。
>
> 所以我们要想办法把它**变回期望形式**。

#### Step 3：log-derivative trick（一个小代数恒等式）

这是整个推导**唯一**的小技巧。用的是高中链式法则：

```math
\frac{d}{dx} \log f(x) = \frac{f'(x)}{f(x)} \quad \Longrightarrow \quad f'(x) = f(x) \cdot \frac{d}{dx}\log f(x)
```

套到我们的式子上：

```math
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau)
```

> 看一眼就明白：左边一个梯度，右边等价地写成「**概率 × log 的梯度**」。变魔术之处在于：**右边把概率 $p_\theta$ 又请回来了**。

代回 Step 2：

```math
\nabla_\theta J(\theta) = \sum_{\tau} p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau)
```

🎉 现在 $p_\theta(\tau)$ 又是一个概率了！这正好是期望的形式：

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log p_\theta(\tau) \cdot R(\tau)\right]
```

**这一步的意义**：我们已经成功把"梯度"变成"期望"。**采样几条轨迹就能近似它**。

#### Step 4：把 $\log p_\theta(\tau)$ 拆开

现在唯一的疑问是： $\log p_\theta(\tau)$ 到底是个啥？我们要把它具体写出来。

一条轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots, s_T)$ 是按这样生成的：

1. 环境给个初始状态 $s_0$ （概率 $\rho_0(s_0)$ ）
2. 策略选动作 $a_0$ （概率 $\pi_\theta(a_0|s_0)$ ）
3. 环境转移到 $s_1$ （概率 $P(s_1|s_0,a_0)$ ）
4. 策略选动作 $a_1$ （概率 $\pi_\theta(a_1|s_1)$ ）
5. ……

所有事件**独立**地连起来（马尔可夫性），所以概率是连乘：

```math
p_\theta(\tau) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)
```

取 log，连乘变连加：

```math
\log p_\theta(\tau) = \underbrace{\log \rho_0(s_0)}_{\text{与 }\theta\text{ 无关}} + \sum_t \log \pi_\theta(a_t|s_t) + \underbrace{\sum_t \log P(s_{t+1}|s_t,a_t)}_{\text{与 }\theta\text{ 无关}}
```

对 $\theta$ 求梯度时，常数项直接没了：

```math
\boxed{\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)}
```

🔑 **这步极其重要**：环境的转移概率 $P$ 完全消失！策略梯度**不需要知道环境怎么工作**——这是 RL 算法能 model-free（不学环境模型）的根本原因，也是它能用在 LLM 上的关键（语言"环境"几乎不可建模）。

#### Step 5：拼起来，得到 REINFORCE

代回 Step 3：

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[ \underbrace{\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)}_{\text{方向：让这些动作更可能发生}} \cdot \underbrace{R(\tau)}_{\text{打分：决定走多远、朝哪边}} \Big]
```

✅ 这就是策略梯度的**最朴素形式**（ $\Psi_t = R(\tau)$ ），叫做 **REINFORCE**。

> **🐕 直观理解：试错 + 强化**
>
> - $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 是参数空间里的**指南针**——"想让这个动作更常出现， $\theta$ 该往哪挪"，与好坏无关。
> - $R(\tau)$ 是环境给的**标量打分**： $R>0$ 顺着指南针走（强化）， $R<0$ 逆着走（抑制）， $|R|$ 越大步子越大。
> - 期望 $`\mathbb{E}_{\tau\sim\pi_\theta}`$ 表示**用当前策略真的跑几把**，再按得分调整自己。
>
> 一句话：**试着做 → 看分数 → 好的多复制、坏的少复制**（训狗式学习）。
>
> 也可看作**加权 MLE**：普通监督学习 $\nabla \log p(\text{data})$ 是无差别模仿；策略梯度是**自己生成数据、按 reward 加权地模仿自己**——这正是 RLHF 能 work 的本质。
>
> **MLE（Maximum Likelihood Estimation，最大似然估计）的非梯度形式**：
>
> ```math
> \theta^* = \arg\max_\theta \prod_{i=1}^{N} p_\theta(x_i) = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta(x_i)
> ```
>
> 一句话：找一组参数 $\theta$，使得观测到的数据在模型下的联合概率最大。取 log 把连乘变求和（不影响 argmax），对应到策略/语言模型场景就是 $\arg\max_\theta \sum_t \log \pi_\theta(a_t|s_t)$。表格里的梯度形式正是对这个目标求导后的结果。
>
> **💡 Insight：为什么概率要取 log？**
>
> 1. **数学等价（前提）**：log 单调递增，$\arg\max \prod p = \arg\max \sum \log p$，最优解不变。
> 2. **数值稳定（工程上必须）**：概率 $\leq 1$，几千项连乘直接下溢到 0；取 log 后变成负数求和，浮点数完全hold住。
> 3. **求导方便（锦上添花）**：连乘的导数要用乘积法则，每项梯度牵涉所有其他项；求和的导数逐项独立，天然适配 mini-batch SGD。
>
> 把两者公式并排看就更清楚了：
>
> | | 梯度形式 | 数据来源 | 权重 |
> |---|---|---|---|
> | 监督学习 (MLE) | $\nabla_\theta \sum_t \log \pi_\theta(a_t\mid s_t)$ | 外部标注数据 | 所有样本**权重相等** (=1) |
> | 策略梯度 (REINFORCE) | $\nabla_\theta \sum_t \log \pi_\theta(a_t\mid s_t) \cdot R(\tau)$ | **自己采样**的轨迹 | 按 reward **加权** |
> | RLHF | $\nabla_\theta \sum_t \log \pi_\theta(a_t\mid s_t) \cdot r_\phi(\text{response})$ | 模型自己生成的回答 | 按 **reward model 打分**加权 |
>
> 关键区别只有两处：
> - **数据从哪来**：MLE 模仿人类标注；策略梯度/RLHF 模仿自己（on-policy 生成）
> - **要不要加权**：MLE 无差别全抄；策略梯度/RLHF 只强化"得分高"的那些输出
>
> 所以 RLHF 的本质就是：让模型**自己说一堆回答 → reward model 打分 → 高分的做更多、低分的做更少**。形式上和 REINFORCE 一模一样，只是 $R(\tau)$ 换成了一个学出来的 reward model $r_\phi$。这也解释了为什么 RLHF 能让模型"超越"监督数据——它不被标注上限卡住，而是在自己的输出空间里做**择优放大**。

**怎么用？** 采样 $N$ 条轨迹 $`\{\tau^{(i)}\}_{i=1}^N`$ ，估计：

```math
\widehat{\nabla J}(\theta) \approx \frac{1}{N}\sum_{i=1}^N \sum_t \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot R(\tau^{(i)})
```

这就能拿去做梯度上升了： $\theta \leftarrow \theta + \alpha \cdot \widehat{\nabla J}(\theta)$ （ $\widehat{\nabla J}$ 本身就是和 $\theta$ 同形状的向量）。

> **🔧 工程上怎么落地？** 实际不会手算 $\nabla \log \pi$ ，而是构造一个**代理 loss** 让 autograd 自动反传：
>
> $L(\theta) = -\dfrac{1}{N}\sum_i \sum_t \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot \underbrace{R(\tau^{(i)})}_{\text{detach 当常数}}$
>
> - 加**负号**：把"梯度上升 $J$ "翻译成"loss 下降"，适配标准优化器
> - $R(\tau^{(i)})$ 必须 **detach / stop\_gradient**，它只是样本权重，不参与反传
>
> ```python
> log_prob = policy.log_prob(a_t, s_t)     # ∇θ 沿这里反传
> loss = -(log_prob * R.detach()).mean()    # R 是 detach 的标量
> loss.backward(); optimizer.step()         # θ ← θ + α∇̂J  ✅
> ```
>
> **本质**：如果 $R\equiv 1$ ，这就是普通最大似然（交叉熵）；现在每条轨迹有自己的得分 $R$ → **按 reward 加权的监督学习**。框架、优化器、反传全不用改。

#### Step 6：用「因果性」降方差，得到 reward-to-go

REINFORCE 能用，但**方差太大**：每个 $\nabla \log \pi(a_t|s_t)$ 都乘了同一个 $R(\tau)$ ，包括 $a_t$ 之前发生的奖励。直觉上这不对：

> $t=5$ 时刻选的动作，**不应该**为 $t=0,1,2,3,4$ 已经发生的奖励"负责"。

我们能不能把"早于 $t$ 的奖励"从 $R(\tau)$ 里扔掉？答案是 **可以，且不改期望**。

**关键引理**：对任意 $t' < t$ ，

```math
\mathbb{E}_{\tau \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot r_{t'} \right] = 0
```

**为什么？** 用条件期望（"先固定历史，再看动作"）：

设历史 $h_t = (s_0, a_0, \dots, s_t)$ 。当 $t' < t$ 时， $r_{t'}$ 完全由 $h_t$ 决定（一个常数），可以提出来：

```math
\mathbb{E}[\nabla \log \pi(a_t|s_t) \cdot r_{t'}] = \mathbb{E}_{h_t}\Big[\; r_{t'} \cdot \underbrace{\mathbb{E}_{a_t \sim \pi(\cdot|s_t)}[\nabla \log \pi(a_t|s_t)]}_{(\star)} \;\Big]
```

而 $(\star)$ 恒等于 0，因为：

```math
\sum_{a} \pi(a|s) \nabla \log \pi(a|s) = \sum_{a} \nabla \pi(a|s) = \nabla \underbrace{\sum_a \pi(a|s)}_{=1} = \nabla 1 = 0
```

> 这一步用了和 Step 3 一样的恒等式 $\pi \nabla \log \pi = \nabla \pi$ ，再加上「概率求和为 1」。**这是所有 baseline 类技巧的根基**，请记住。

> **🎯 物理直觉：因果律 + 概率守恒**
>
> **① 因果律**： $r_{t'}$ （ $t' \lt t$ ）在选 $a_t$ **之前就已成事实**，调整 $a_t$ 的策略**改变不了过去的奖励**。两件事统计独立，相乘期望就拆成两个期望的乘积。
>
> **② 概率守恒（score function 零均值）**： $\nabla \log \pi(a|s)$ 是"让动作 $a$ 更可能"的方向，但所有动作概率之和恒等于 1——**增加某个动作必须减少别的**，所有"指南针方向"互相抵消，平均必为零向量： $\mathbb{E}_{a\sim\pi}[\nabla \log \pi(a|s)] = 0$ 。
>
> 🐕 **训狗类比**：你**不能**因为狗"上周乖"就奖励它"今天某个具体动作"——今天选啥改变不了过去，这种"旧账信号"平均下来是 0，对学习无信息量。只有"动作之后"发生的事（reward-to-go）才真正承载了"这个动作好不好"的信息。

所以我们可以**安心地**只保留 $t' \ge t$ 的奖励，得到 **reward-to-go**：

```math
\hat G_t = \sum_{t' = t}^{T} \gamma^{t'-t} r_{t'}
```

> **💡 这里为什么突然出现了折扣因子 $\gamma$？**
>
> 前面推导全程没有 $\gamma$，reward-to-go 的"无折扣"版本应该是 $\hat G_t = \sum_{t'=t}^{T} r_{t'}$。折扣因子 $\gamma \in [0,1)$ 是一个**额外引入的设计选择**，动机有三层：
>
> **① 数学收敛（必须）**：如果轨迹长度 $T \to \infty$（无限步任务），不加折扣的总回报 $\sum r_{t'}$ 可能发散到无穷大。乘上 $\gamma^{t'-t}$ 后级数收敛，$G_t$ 有界。
>
> **② 远期奖励不确定性大（实际考量）**：越远的 $r_{t'}$ 中间经过的随机步越多，对"当前动作好不好"的信号越模糊。$\gamma^{t'-t}$ 指数衰减 = **让近期奖励权重大、远期奖励权重小**，降低方差。
>
> **③ 经济学直觉（偏好建模）**：$\gamma$ 相当于"时间偏好率"——今天的 1 块钱比明天的 1 块钱值钱。agent 更看重即时回报，这在很多任务里是合理的归纳偏置。
>
> **注意**：加了 $\gamma < 1$ 后梯度估计**严格来说引入了偏差**（因为我们改变了优化目标，从最大化总回报变成最大化折扣回报）。但实践中这个偏差远小于它带来的方差收益，所以几乎所有实现都用 $\gamma \in [0.99, 0.999]$。

策略梯度变成：

```math
\nabla_\theta J = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat G_t \right]
```

**这一步神奇之处**：方差变小、估计更稳定，但**期望不变**（"无偏"）。这是后面所有 advantage 估计的灵魂。

> **💡 为什么方差会变小？**
>
> 用全部回报时，梯度的每个样本是 $\nabla \log \pi(a_t|s_t) \cdot \sum_{t'=0}^{T} r_{t'}$；用 reward-to-go 后变成 $\nabla \log \pi(a_t|s_t) \cdot \sum_{t'=t}^{T} r_{t'}$，区别就是砍掉了 $t' < t$ 的过去奖励。
>
> 我们已经证明被砍掉的部分**期望为零**（因果律 + score function 零均值），但"期望为零" ≠ "每次采样为零"！每次采到的 $r_{t'} \cdot \nabla \log \pi(a_t|s_t)$ 可以是很大的正数或负数——长期平均消掉，但**每一次都在剧烈波动**。
>
> 由于"未来奖励"和"过去奖励"关于 $a_t$ 条件独立，方差可以拆开：
>
> ```math
> \text{Var}\big[\nabla \log \pi \cdot R(\tau)\big] = \text{Var}\big[\nabla \log \pi \cdot \hat G_t\big] + \underbrace{\text{Var}\big[\nabla \log \pi \cdot \textstyle\sum_{t'<t} r_{t'}\big]}_{\geq\, 0,\;\text{纯噪声}}
> ```
>
> 被砍掉的那项：对期望的贡献 = **0**（有用信号量为零），对方差的贡献 = **正数**（纯噪声）。砍掉它 = **信号不变，噪声减少** → 方差严格变小。
>
> 一句话：过去的奖励 $r_{t'<t}$ 跟当前动作 $a_t$ 统计独立，乘出来期望为零——**它不携带任何关于"$a_t$ 好不好"的信息，却每次采样都在注入随机波动**。扔掉它就是扔掉纯噪声，信噪比自然提高。

#### Step 7：把 $\hat G_t$ 推广为通用权重 $\Psi_t$

沿着同样的「无偏 + 减方差」逻辑，可以把 $\hat G_t$ 换成各种东西：

1. 减 baseline： $\Psi_t = \hat G_t - b(s_t)$ （任何只依赖 $s_t$ 的函数都不引入偏差，原因和 Step 6 那个 $(\star) = 0$ 一模一样）
2. 用 $Q$ 函数： $\Psi_t = Q^\pi(s_t,a_t)$ （因为 $\mathbb{E}[\hat G_t | s_t, a_t] = Q^\pi$ ，期望守恒）
3. 用 advantage： $\Psi_t = A^\pi = Q^\pi - V^\pi$ （baseline 取 $b(s_t)=V^\pi(s_t)$ ）
4. 用 TD 残差： $\Psi_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ （用真 $V$ 时也是 $A^\pi$ 的无偏估计）
5. 用 GAE：在 $\lambda$ 上对多种步数插值

每一种都是「方差—偏差」轴上的不同权衡。最终通用形式：

```math
\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \Psi_t \right]}
```

---

### 1.2 整个推导回头看：只有两个核心 idea

如果让你只记两件事，记这两个：

1. **log-derivative trick**： $\nabla p = p \cdot \nabla \log p$
   作用：把"梯度"包装成"期望"，使采样估计成为可能。

2. **$\mathbb{E}[\nabla \log \pi(a|s) \cdot c(s)] = 0$**（baseline 引理）
   作用：可以任意减掉一个只依赖状态的项，**只降方差，不引入偏差**。这是 advantage、baseline、reward-to-go 全部的数学基础。

> **💡 Baseline 引理为什么成立 & 为什么能降方差？**
>
> **为什么等于零：** $c(s)$ 对于动作 $a$ 的采样来说是常数，提出来后剩下的正是 score function 零均值：
>
> ```math
> \mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s) \cdot c(s)] = c(s) \cdot \underbrace{\sum_a \pi(a|s) \nabla \log \pi(a|s)}_{= \nabla \sum_a \pi(a|s) = \nabla 1 = 0} = 0
> ```
>
> 所以**不管 $c(s)$ 是什么**，减掉它都不改变梯度期望（无偏）。
>
> **为什么能降方差：** $\hat G_t$ 本身可能是很大的正数（比如 +200），波动剧烈。如果 baseline $b(s_t) \approx \mathbb{E}[\hat G_t | s_t] = V^\pi(s_t)$，那 $\hat G_t - b(s_t)$ 就变成围绕零波动的小数（比如 ±20），数值尺度小一个量级，方差自然小。
>
> **直觉：** 原来的信号是"这条轨迹总共拿了 200 分"→ 所有动作都被鼓励。减掉 baseline 后变成"这条轨迹**比平均多拿了 20 分**"→ 只有真正好于平均的动作被鼓励，差于平均的被抑制。信号从"绝对好坏"变成"相对好坏"，更精准、波动更小。这就是为什么 advantage $A^\pi = Q^\pi - V^\pi$ 是最常用的形式——$V^\pi$ 就是那个最优的 baseline。

其它步骤都只是代数操作。

---

### 1.3 一个超小例子（彻底体会"采样估计"）

假设状态只有 1 个，动作只有 2 个： $a_1, a_2$ 。

策略 $\pi_\theta(a_1) = \sigma(\theta)$ （sigmoid）， $\pi_\theta(a_2) = 1 - \sigma(\theta)$ 。奖励：选 $a_1$ 得 +1，选 $a_2$ 得 0。

**真实梯度**（解析算）：

```math
J(\theta) = \sigma(\theta) \cdot 1 + (1-\sigma(\theta)) \cdot 0 = \sigma(\theta) \quad \Rightarrow \quad \nabla J = \sigma(\theta)(1-\sigma(\theta))
```

**用策略梯度估计**：采样 100 次，每次记下 $(a_i, r_i)$ ，估计：

```math
\widehat{\nabla J} = \frac{1}{100}\sum_i \nabla_\theta \log \pi_\theta(a_i) \cdot r_i
```

只有 $a_i = a_1$ 的样本贡献非零项（因为 $r_i=0$ 时整项为 0），近似为 $\sigma(\theta) \cdot \nabla \log \sigma(\theta) = \sigma(\theta)(1-\sigma(\theta))$ ，和真实梯度一致。✅

这就是策略梯度的全部魔法：**采样 + 加权 log 概率 = 无偏的梯度估计**。

---

### 1.4 和监督学习的类比（一句话理解策略梯度）

**监督学习交叉熵**（要最小化的 loss）：

```math
\mathcal{L}_{CE}(\theta) = -\log \pi_\theta(a^* \mid s) \quad \Rightarrow \quad \nabla \mathcal{L}_{CE} = -\nabla \log \pi_\theta(a^* \mid s)
```

（ $a^*$ 是数据集里的正确标签。最小化 loss = 最大化正确标签的对数概率。）

> **💡 CE loss 里的 log 是怎么来的？——两条等价的推导路线**
>
> **路线 1：从 MLE 来（概率视角）**
>
> 目标：最大化正确标签的概率 $`\max_\theta \pi_\theta(a^*|s)`$
> → 取 log（单调，不改变最优解）：$`\max_\theta \log \pi_\theta(a^*|s)`$
> → 加负号变最小化（优化器默认 minimize）：$\min_\theta -\log \pi_\theta(a^*|s)$
> → 这就是 CE loss。
>
> **路线 2：从信息论来（分布距离视角）**
>
> Cross-Entropy 定义：$H(p,q) = -\sum_a p(a) \log q(a)$，衡量真实分布 $p$ 与模型分布 $q$ 的距离。监督学习中 $p$ 是 one-hot（$p(a^*)=1$，其余为 0），代入求和只剩一项：
>
> ```math
> H(p,q) = -1 \cdot \log \pi_\theta(a^*|s) = -\log \pi_\theta(a^*|s)
> ```
>
> **两条路殊途同归：CE loss = 负对数似然 = MLE 的最小化形式。**
>
> **CE 与 MLE 的关系：**
>
> | | MLE | CE loss |
> |---|---|---|
> | 形式 | $\max_\theta \sum_i \log p_\theta(x_i)$ | $\min_\theta -\frac{1}{N}\sum_i \log p_\theta(x_i)$ |
> | 本质 | 同一件事 | 同一件事 |
> | 区别 | 概率/统计的叫法，做 max | 深度学习的叫法，加负号做 min |
>
> log 不是"额外加的"，而是**概率→损失函数**的标准桥梁：把 $[0,1]$ 的概率映射到 $[0, +\infty)$ 的 loss，概率越大 loss 越小，概率趋近 0 时 loss → $+\infty$（惩罚极重）。

**策略梯度**（要最大化的目标）：

```math
\nabla_\theta J = \mathbb{E}_{a_t \sim \pi_\theta}\left[ \Psi_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
```

写成"loss"形式（实际代码里都是这么写的，前面加负号）：

```math
\mathcal{L}_{PG}(\theta) = -\mathbb{E}\left[ \Psi_t \cdot \log \pi_\theta(a_t \mid s_t) \right]
```

**对比看**：

| 维度        | 监督学习（CE）                  | 策略梯度（PG）                                  |
| --------- | ------------------------- | ----------------------------------------- |
| 学谁？       | 正确标签 $a^*$ （人工给）           | 自己采样的 $a_t \sim \pi_\theta$                |
| 每个样本的"权重" | 1（每个标签都同等重要）              | $\Psi_t$ （advantage：好就正、坏就负）               |
| 效果        | 把 $\pi(a^* \mid s)$ 拉高             | $\Psi_t>0$ 把 $\pi(a_t \mid s)$ 拉高， $\Psi_t<0$ 拉低 |
| 数据来源      | 静态数据集                     | 模型自己 rollout 出来的                          |

**一句话**：策略梯度 = **带权重的、自己生成 label 的交叉熵**。
- 模型自己采样动作
- 环境（或 reward model）打分
- 正分鼓励、负分抑制

这也是为什么 LLM RLHF 代码上和 SFT 几乎一样——把 label 换成自采样、把 loss 乘个 advantage 就完事了。

其中 $\Psi_t$ 可以取多种形式（这就是各种算法的差异来源）：

| $\Psi_t$ 的选择                       | 算法 / 含义              |
| ----------------------------------- | -------------------- |
| $G_t$ （总回报）                          | REINFORCE（高方差）       |
| $G_t - b(s_t)$                      | REINFORCE + baseline |
| $Q^\pi(s_t,a_t)$                    | Q Actor-Critic       |
| $A^\pi(s_t,a_t)$                    | Advantage AC（A2C）    |
| $r_t + \gamma V(s_{t+1}) - V(s_t)$  | TD residual          |
| GAE                                 | PPO / 现代 RLHF        |

**直觉**： $\Psi_t > 0$ 时提高 $\log \pi(a_t|s_t)$ → 提升该动作概率； $\Psi_t < 0$ 时反之。

### 为什么要减 baseline？

减一个**只与 $s$ 有关**的 baseline $b(s)$ ，**不引入偏差**：

```math
\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s) \cdot b(s)] = b(s) \nabla \underbrace{\sum_a \pi(a|s)}_{=1} = 0
```

但**显著降低方差**。最常用的 baseline 就是 $V(s)$ ，所以才有了 Advantage $A = Q - V$ 。

---

## 2. Advantage：动作"比平均好多少"

```math
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
```

- $A > 0$ ：这个动作优于该状态下的平均水平 → 鼓励
- $A < 0$ ：劣于平均 → 抑制
- $A = 0$ ：和平均一样，不更新

### 2.1 朴素估计

- 蒙特卡洛： $\hat A_t = G_t - V(s_t)$ ，**无偏但高方差**
- 一步 TD： $\hat A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ ，**低方差但有偏**

> **🎯 直观理解：两种估"动作好不好"的方式**
>
> 本质上都是在估 $Q^\pi(s_t, a_t)$ （这个动作之后总共能拿多少回报），然后减掉基准 $V(s_t)$ 。**区别在用什么估 $Q$**：
>
> **① 蒙特卡洛 = "实测到底"**：用真实走完整条轨迹的总回报 $G_t$ 当 $Q$ 。
> - ✅ **无偏**：来自真实数据，环境怎么走就怎么记。
> - ❌ **高方差**： $G_t$ 包含**之后每一步**动作选择、环境转移、奖励的随机性——一条轨迹运气好 $+1000$ ，运气差 $-1000$ ，但其实开局动作差不多好。
>
> **② 一步 TD = "走一步看一步 + 信任 critic 预测剩下的"**：只用真实跑一步 $r_t$ ，剩下的未来全靠 critic 的 $V(s_{t+1})$ 估。
> - ✅ **低方差**：只引入 1 步的随机性，后面用平滑的预测代替。
> - ❌ **有偏**： $V$ 是学出来的近似函数，**$V$ 不准 → 估出来的 advantage 就跟着歪**。
>
> **🚗 导航类比**：
> - **MC** = 真开车跑一遍,看实际花了多久 → 准但堵车/红绿灯让每次结果差很多
> - **TD** = 开 1 分钟看实际路况 + 导航软件估剩下还要多久 → 稳定但导航不准就跟着错
>
> 一个**完全信现实**(嘈杂)、一个**完全信预测**(可能歪)——这就是 GAE 用 $\lambda$ 在两者之间插值的根本动机。

### 2.2 GAE（Generalized Advantage Estimation）

把 1 步、2 步、… 多步 TD 残差按几何权重组合：

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
```

```math
\hat A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}
```

- $\lambda = 0$ ：退化为单步 TD（低方差、高偏差）
- $\lambda = 1$ ：退化为 MC（无偏、高方差）
- 实际常用 $\lambda \in [0.9, 0.99]$

> **💡 推导：$\lambda=1$ 时 GAE 为什么退化为 MC？**
>
> 令 $\lambda=1$，GAE 变为 $\hat A_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}$。把 $\delta_{t+l} = r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})$ 代入并拆成三个求和：
>
> ```math
> \hat A_t = \underbrace{\sum_{l=0}^{\infty} \gamma^l r_{t+l}}_{(1)} + \underbrace{\sum_{l=0}^{\infty} \gamma^{l+1} V(s_{t+l+1})}_{(2)} - \underbrace{\sum_{l=0}^{\infty} \gamma^l V(s_{t+l})}_{(3)}
> ```
>
> 对 (2) 换元 $m = l+1$：$\sum_{m=1}^{\infty} \gamma^m V(s_{t+m})$
>
> 对 (3) 拆出首项：$V(s_t) + \sum_{m=1}^{\infty} \gamma^m V(s_{t+m})$
>
> (2) − (3) 中 $\sum_{m=1}^{\infty}$ 的部分**完全抵消**（telescoping / 望远镜消去），只剩 $-V(s_t)$：
>
> ```math
> \hat A_t = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)
> ```
>
> 这正是**蒙特卡洛 advantage 估计**：用真实回报 $G_t$ 减去 baseline。所有中间的 $V$ 项都望远镜式地消掉了——$\lambda=1$ 意味着"完全不截断，信任真实回报到底"，critic 只作为 baseline 出现一次。

**为什么需要 GAE**：在偏差和方差之间提供一个**连续调节旋钮**。这是现代 PPO 的标配。

> **🎯 直观理解：把 $\delta_t$ 看作"惊讶值"**
>
> $`\delta_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{实际拿到 + 之后估值}} - \underbrace{V(s_t)}_{\text{原本预期}}`$
>
> 就是 **"实际比预期好/差多少"** ——一个**瞬时惊讶（prediction error）**：
> - $\delta_t > 0$ ：这步比预想的好 → 该动作值得鼓励
> - $\delta_t < 0$ ：比预想的差 → 该动作要抑制
>
> **但 $V$ 自己也不准！** $V(s_{t+1})$ 也可能高估/低估，要等到再往后走才暴露——所以 $\delta_{t+1}$ 是"再过一步才浮现出来的惊讶"， $\delta_{t+2}$ 是"再过两步浮现的惊讶"……
>
> **GAE 就是把这些迟到的惊讶按几何权重 $(\gamma\lambda)^l$ 累加**：当下立刻显现的算满分，越往后权重越小。
>
> **$\lambda$ 是"对 critic 的信任旋钮"**：
> - $\lambda = 0$ ：完全信 $V$ ，只看 1 步惊讶 → 平滑稳定，但 $V$ 错了就跟着错（**有偏**）
> - $\lambda = 1$ ：完全不信 $V$ ，把所有未来惊讶都算上 → 等价于 $G_t - V(s_t)$ ，真实但嘈杂（**高方差**）
> - $\lambda \in [0.9, 0.99]$ ：**适度信任** critic 但也用真实回报校正——工程上的甜点区。
>
> 一句话：**GAE = 用 critic 估算"动作好坏"，并允许真实回报对这个估算做几何衰减的修正**。

### 2.3 LLM 场景下的特殊性

- 奖励通常**只在最后一个 token** 给出（来自 RM）
- 中间 token 的"即时奖励" $r_t = 0$ （或只有 KL 惩罚项）
- GAE 退化为：把末端 reward 通过 $\gamma$ 反向折扣到每个 token，再减去 critic 估计
- 部分实现里直接 $\gamma = 1, \lambda = 1$ ，相当于序列级别的 reward-to-go

---

## 3. 重要性采样（Importance Sampling）：让旧数据可重用

### 3.1 动机

朴素策略梯度是 **on-policy** 的：每次更新 $\theta$ 后，旧数据就"过期"了，必须重新 rollout。这在 LLM 上极度昂贵（rollout 一次要跑很久）。

我们希望：**用 $\pi_{\theta_{old}}$ 采样的数据，多次更新 $\theta$**。

### 3.2 数学基础

```math
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]
```

应用到策略梯度：

```math
J(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\theta_{old}}} \left[ \underbrace{\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}}_{r_t(\theta)\text{，重要性比}} \cdot A^{\pi_{\theta_{old}}}(s,a) \right]
```

记 $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ ，称为 **importance ratio**。

- 当 $\theta = \theta_{old}$ 时 $r_t = 1$
- 越远， $r_t$ 越偏离 1

> **🎯 直观理解：给旧样本"打折/加权"，假装它来自新分布**
>
> **核心思想**：不想重新采样了（rollout 太贵），那就**给旧样本一个权重**，让加权后的旧数据"看起来像是从新策略下采的"。
>
> **🛒 调研类比**：你想估计"全国人均消费"，但手头只有"北京调研数据"。
> - 不用重做全国调研——直接给每个北京样本乘个权重 $\frac{p_{\text{全国}}(x)}{p_{\text{北京}}(x)}$
> - 在北京被过采样的群体（比如高收入者）权重 $<1$ ，在全国相对稀有的样本权重 $>1$ → 加权后的"北京数据"就能估全国均值
>
> **$r_t$ 的物理含义**：
> - $r_t > 1$ ：新策略 $\pi_\theta$ **比**旧策略**更**爱选这个 $a_t$ → 这条旧样本"对新策略更有代表性" → **放大它的影响力**
> - $r_t < 1$ ：新策略**不太**爱选这个 $a_t$ → 这条样本对新策略不太相关 → **缩小它的影响力**
> - $r_t = 1$ ：两策略对该动作偏好一致 → 不打折(退化为普通策略梯度)
>
> **为什么有"信任域"问题？** 这种"打折"只在新旧策略**差距不大**时有效；差距太大, $r_t$ 会变成几十几百倍,**少数样本主导一切**——这就是 PPO 要 clip 的根本原因（下一节）。

### 3.3 重要性采样的隐患

当 $\pi_\theta$ 和 $\pi_{\theta_{old}}$ 差距大时：

1. **方差爆炸**： $r_t$ 可能极大或极小（几十、几百倍）
2. **梯度失真**：少量样本主导更新方向
3. **训练崩溃**：策略可能直接跑飞

→ 这就是为什么需要 **trust region**（信任域）：限制 $\pi_\theta$ 不要离 $\pi_{\theta_{old}}$ 太远。

---

## 4. 从 TRPO 到 PPO

### 4.1 TRPO（Trust Region Policy Optimization）

约束优化：

```math
\max_\theta \ \mathbb{E}\left[r_t(\theta) A_t\right] \quad \text{s.t.} \quad \mathbb{E}[\text{KL}(\pi_{\theta_{old}} \| \pi_\theta)] \le \delta
```

**问题**：要求二阶信息（Hessian / 共轭梯度），实现复杂，难和大模型 + 分布式训练结合。

> **💡 TRPO 与 PPO 的关系：PPO 是 TRPO 的工程近似**
>
> 两者要解决的问题**完全一样**：策略更新步子别太大，否则性能塌方。
>
> | | TRPO | PPO-Clip |
> |---|---|---|
> | 限制手段 | **硬约束**：$\text{KL}(\pi_{old} \Vert \pi_{new}) \le \delta$ | **软限制**：clip 比率到 $[1-\epsilon, 1+\epsilon]$ |
> | 优化方法 | 约束优化（拉格朗日 + 共轭梯度 + line search） | 普通 SGD，和监督学习一样 |
> | 需要二阶信息 | 是（Hessian-vector product） | 否 |
> | 实现复杂度 | 高 | 低，几行代码 |
> | 效果 | 理论保证更强 | 实践中差不多，甚至更好 |
>
> TRPO 用**精确的数学约束**（KL 散度）保证"新旧策略不能差太远"。PPO 说：与其费劲算 KL 约束，不如直接**把概率比率 clip 掉**——超出范围的梯度直接归零，粗暴但有效，SGD 就能跑。所以 **PPO-Clip 是 TRPO 思想的工程近似**：TRPO = "用 KL 约束精确画一个信任域"，PPO = "用 clip 粗略画一个信任域，效果差不多，实现简单 10 倍"。

### 4.2 PPO-Clip：用一个 min + clip 替代 KL 约束

```math
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\big( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \big) \right]
```

- $\epsilon$ 是 clip 范围，常用 $0.1 \sim 0.2$
- 实现极其简单（一行 `torch.clamp`），但效果接近 TRPO

#### 关键：为什么是 `min`？分情况看

**Case 1： $A_t > 0$ （好动作）**

我们想让 $r_t$ 变大（增加该动作概率）。
- 若 $r_t \le 1 + \epsilon$ ：正常更新
- 若 $r_t > 1 + \epsilon$ ： $\text{clip}$ 截断为 $1+\epsilon$ ，**再大也不给奖励**
  → 防止策略更新过激

**Case 2： $A_t < 0$ （坏动作）**

我们想让 $r_t$ 变小（减少该动作概率）。
- 若 $r_t \ge 1 - \epsilon$ ：正常更新
- 若 $r_t < 1 - \epsilon$ ： $\text{clip}$ 截断为 $1-\epsilon$ ，**再小也不再惩罚**

#### `min` 的妙处：单边限制

注意：**只在「会让目标函数变大且更新过激」的方向限制**，对「拉回安全区」的方向不限制。

举例： $A_t > 0$ 但当前 $r_t < 1-\epsilon$ （一个错误的策略压低了好动作），此时：
- $r_t A_t$ 较小
- $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t = (1-\epsilon)A_t$ 较大
- $\min$ 选小的 → 仍然给完整梯度让策略**修正回来**

这是一种**悲观下界**（pessimistic lower bound）。

### 4.3 PPO 完整目标

```math
L^{PPO}(\theta) = \mathbb{E}_t \left[ L^{CLIP}_t - c_1 L^{VF}_t + c_2 \mathcal{H}[\pi_\theta](s_t) \right]
```

三项：
1. **Clip 策略损失**（actor）
2. **Value loss**： $L^{VF} = (V_\theta(s_t) - V_t^{target})^2$ （critic，常和 actor 共享 backbone）
3. **Entropy bonus**：鼓励探索，防止策略坍缩到 deterministic

### 4.4 手推 PPO-Clip 反传

PPO-Clip loss（要最大化）：

```math
L^{CLIP} = \min\big(\underbrace{r_t A_t}_{f_1},\; \underbrace{\text{clip}(r_t, 1\!-\!\epsilon, 1\!+\!\epsilon) \cdot A_t}_{f_2}\big)
```

**Step 1：$\nabla_\theta r_t$**

$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，分母是冻结常数，所以：

```math
\nabla_\theta r_t = \frac{\nabla_\theta \pi_\theta}{\pi_{\theta_{old}}} = \frac{\pi_\theta}{\pi_{\theta_{old}}} \cdot \nabla_\theta \log \pi_\theta = r_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)
```

**Step 2：min 的梯度是分段的**

min 选谁，梯度就走谁。clip 在 $[1-\epsilon, 1+\epsilon]$ 内梯度 = $\nabla r_t$，超出范围梯度 = 0（输出为常数）。

- **$r_t \in [1-\epsilon, 1+\epsilon]$（未被 clip）**：$f_1 = f_2$，梯度 = $A_t \cdot \nabla_\theta r_t$，正常流过。
- **$r_t$ 越界时分四种情况**：

| $A_t$ | $r_t$ 范围 | min 选谁 | 梯度 | 直觉 |
|---|---|---|---|---|
| $> 0$（好动作） | $r_t > 1+\epsilon$ | $f_2$（clipped） | **0** | 概率已经涨够了，别再涨 |
| $> 0$（好动作） | $r_t < 1-\epsilon$ | $f_1$（unclipped） | $A_t \nabla r_t$ | 概率反而降了？允许修正回来 |
| $< 0$（坏动作） | $r_t < 1-\epsilon$ | $f_2$（clipped） | **0** | 概率已经降够了，别再降 |
| $< 0$（坏动作） | $r_t > 1+\epsilon$ | $f_1$（unclipped） | $A_t \nabla r_t$ | 坏动作概率反而涨了？允许压回来 |

**Step 3：汇总**

```math
\nabla_\theta L^{CLIP} = \begin{cases} A_t \cdot r_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) & \text{梯度流过} \\ 0 & \text{梯度被截断} \end{cases}
```

> **一句话总结**：clip 的反传本质就是一个**梯度开关**——当策略更新步子已经迈够大了（$r_t$ 越出 $[1-\epsilon, 1+\epsilon]$ 且方向是"优化过头"），直接把梯度关掉，防止过度优化。只有当越界方向"不对"（好动作概率反而降了 / 坏动作概率反而涨了）时，梯度才被放行用于修正。

### 4.5 训练循环（pseudocode）

```python
for iteration in range(1, max_iter + 1):

    # ── Phase 1: Rollout（用冻结的 π_old 采集数据）──────────────
    trajectories = collect_trajectories(π_θ_old, num_trajectories=N)

    # ── Phase 2: 计算 Advantage & Return Target ─────────────
    for each timestep t in trajectories:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)          # TD 残差
        A_t = GAE(δ, γ, λ)                            # 广义优势估计
        R_t = A_t + V(s_t)                            # value 网络的回归目标

    # ── Phase 3: 多 epoch 更新（importance sampling 的回报）───
    for epoch in range(K):
        for minibatch in shuffle_and_split(trajectories):
            # 概率比率
            r_t    = π_θ(a|s) / π_θ_old(a|s)

            # 三个 loss 分量
            L_clip = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
            L_vf   = (V_θ(s) - R_t) ** 2
            L_ent  = entropy(π_θ)

            # 总 loss：最大化 clip 目标 + 最小化 value 误差 + 鼓励探索
            loss   = -L_clip + c1 * L_vf - c2 * L_ent

            loss.backward()
            optimizer.step()

    # ── 同步：新策略变成下一轮的 old ──────────────────────
    π_θ_old ← π_θ
```

注意：**π_old 在整轮 rollout 中固定**，所以这一轮内多个 epoch 的更新都可以靠重要性采样修正。

> **💡 $\delta_t$ 用了 V 算 advantage，$L_{vf}$ 又更新 V——不矛盾吗？**
>
> 不矛盾。Phase 2 用**当前冻结的 V** 算出所有 $\delta_t$、$A_t$、$R_t$，算完后这些数值就写死了，成为固定的 target。Phase 3 拿这些固定的 $R_t$ 当回归标签去更新 V 参数，使其预测更准。V 追的不是自己，而是一个**比自己更好的估计**。
>
> **$R_t$ 和 $V(s_t)$ 的本质区别：**
>
> | | 含义 | 信息来源 |
> |---|---|---|
> | $V(s_t) = \mathbb{E}_\pi[G_t \mid s_t]$ | 从 $s_t$ 出发，**所有可能轨迹**的期望回报 | 纯靠网络参数预测 |
> | $R_t = A_t + V(s_t)$ | 沿着**实际采样的这一条轨迹**算出的回报估计 | 真实奖励 $r_t, r_{t+1}, \ldots$ + V 的 bootstrap |
>
> $V$ 是对所有可能未来的**期望**（一个统计量），$R_t$ 是这个期望的**一个样本**（来自实际跑的轨迹，包含了真实奖励信息，所以比 V 的盲猜更接近真实值）。$L_{vf} = (V_\theta(s) - R_t)^2$ 就是经典的**用样本做回归逼近期望**——大量样本的 $R_t$ 取平均后趋向真实的 $\mathbb{E}[G_t|s_t]$，V 通过最小化 MSE 逐渐学到这个期望。
>
> 类比天气预报：$V(s_t)$ = 模型预测"明天 25°C"；$R_t$ = 实际观测到"白天 23°C" + 模型对后天的预测，修正后得到更准的估计。用修正值当标签更新模型，让它下次预测更准。

> **💡 Entropy 是怎么算的？**
>
> 策略熵衡量"动作分布有多随机"，定义为：
>
> ```math
> H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
> ```
>
> **离散动作空间**（如 Atari、棋类、token 选择）直接按定义求和：
>
> ```python
> # probs: [batch, num_actions], 策略输出的动作概率分布
> entropy = -(probs * probs.log()).sum(dim=-1).mean()
> ```
>
> **连续动作空间**（如机器人控制），策略通常输出高斯分布 $\mathcal{N}(\mu, \sigma^2)$，熵有解析解：
>
> ```math
> H = \frac{1}{2} \log(2\pi e \sigma^2)
> ```
>
> ```python
> # std: [batch, action_dim], 策略输出的标准差
> entropy = 0.5 * torch.log(2 * math.pi * math.e * std ** 2).sum(dim=-1).mean()
> ```
>
> 在 PPO loss 中 `loss = -L_clip + c1 * L_vf - c2 * L_ent`，注意符号是 **$-c_2$**，即**最小化 loss = 最大化熵**。这是探索正则项：防止策略过早坍缩到某个动作，$c_2$ 通常取 0.01 左右。
>
> **计算粒度**：entropy 是在**每个 minibatch 内**算的——对 batch 中每个样本 $(s,a)$ 算出该状态下的策略分布熵 $H(\pi_\theta(\cdot|s_i))$，然后取 batch 平均：$L_{ent} = \frac{1}{|B|}\sum_{i \in B} H(\pi_\theta(\cdot|s_i))$。它和 `L_clip`、`L_vf` 一起在同一个 forward pass 里算出，一起 backprop。
>
> **输入是完整分布，不是选中动作的概率**：`L_clip` 和 `L_vf` 只用选中动作 $a_t$ 的概率（标量），但 entropy 必须看**完整动作概率分布** $\pi_\theta(\cdot|s)$（向量），因为熵衡量的是"策略有多随机"，光看最终选了哪个动作算不出来：
>
> ```python
> probs = policy_net(s)                          # [batch, num_actions] ← 完整分布
> log_probs = probs.log()
>
> # entropy：用完整分布，对所有动作求和
> entropy = -(probs * log_probs).sum(dim=-1)     # [batch]
>
> # clip loss：只用选中动作的概率（标量）
> log_pi_a = log_probs.gather(1, a)              # 只取 a_t 那一个
> ```

---

## 5. KL 散度：另一种 trust region

### 5.1 PPO-Penalty（PPO 的另一变体）

不用 clip，而用自适应 KL 惩罚：

```math
L(\theta) = \mathbb{E}[r_t(\theta) A_t] - \beta \cdot \mathbb{E}[\text{KL}(\pi_{\theta_{old}} \| \pi_\theta)]
```

$\beta$ 根据实际 KL 自适应调整（KL 太大就调大 $\beta$ ）。实际中不如 PPO-Clip 流行。

### 5.2 RLHF 里的 KL：一个**不同**的 KL

这个非常重要、初学最容易混的点：

| KL                                              | 作用      | 出现位置          |
| ----------------------------------------------- | ------- | ------------- |
| $`\text{KL}(\pi_{\theta_{old}} \,\Vert\, \pi_\theta)`$   | trust region，防止单步更新过激   | PPO-Penalty 算法 |
| $`\text{KL}(\pi_\theta \,\Vert\, \pi_{\text{ref/SFT}})`$ | 防止 RL 模型偏离 SFT 模型太远 | RLHF 的 reward 项里  |

在 LLM 的 RLHF 中，常见做法是把 **第二种 KL** 直接加进每个 token 的 reward：

```math
\tilde r_t = r_t - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}
```

> **💡 这里的 $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ 不是标准 KL，而是 KL 的单样本估计**
>
> **KL 散度的含义：** KL 散度（Kullback-Leibler divergence）衡量的是"用分布 $q$ 去近似分布 $p$ 时，**额外浪费了多少信息量**"。
>
> - **信息论角度**：如果真实分布是 $p$，用 $p$ 自己编码平均需要 $H(p) = -\sum p \log p$ 比特（熵）。如果错用 $q$ 编码，平均需要 $H(p,q) = -\sum p \log q$ 比特（交叉熵）。多花的部分就是 KL：
>
> ```math
> \text{KL}(p \| q) = H(p,q) - H(p) = \sum_x p(x) \log \frac{p(x)}{q(x)}
> ```
>
> - **统计角度**：$\log \frac{p(x)}{q(x)}$ 是每个样本 $x$ 的**对数似然比**——"这个样本在 $p$ 下有多可能 vs 在 $q$ 下有多可能"。KL 就是这个对数似然比在 $p$ 下的期望。KL = 0 当且仅当 $p = q$，越大说明两个分布差异越大。
>
> - **直觉**：KL 不是"距离"（不对称，$`\text{KL}(p\|q) \neq \text{KL}(q\|p)`$），更像"用 $q$ 冒充 $p$ 的代价"。在 RLHF 里，$`\text{KL}(\pi_\theta \| \pi_{\text{ref}})`$ 衡量的就是"当前策略 $\pi_\theta$ 相对于 reference 策略偏离了多少"。
>
> 标准 KL 散度是对**所有动作求和**的期望：
>
> ```math
> \text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \sum_a \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}
> ```
>
> 而公式里的 $\log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$ 只取了**采样到的那一个动作** $a_t$ 的 log ratio。之所以这样做是因为 RLHF 本身就在按 $\pi_\theta$ 采样轨迹，取期望后恰好恢复完整 KL：
>
> ```math
> \mathbb{E}_{a_t \sim \pi_\theta}\left[\log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}\right] = \text{KL}(\pi_\theta \| \pi_{\text{ref}})
> ```
>
> 所以逐 token 扣 log ratio 进 reward，**期望意义下等价于减去 $\beta \cdot \text{KL}$**，无需显式对完整词表求和。
>
> **这里用的是 Reverse KL，不是 Forward KL**
>
> Forward/Reverse 的命名来自变分推断传统，以**谁是 target（固定的）、谁是 model（要优化的）**为基准：
>
> - **target 分布 $p$**：固定不动的参考（RLHF 里是 $\pi_{\text{ref}}$）
> - **model 分布 $q$**：正在被优化的（RLHF 里是 $\pi_\theta$）
> - **Forward KL** $`\text{KL}(p \| q)`$：target 在前 → "正向"，期望在 target 下取
> - **Reverse KL** $`\text{KL}(q \| p)`$：model 在前 → "反向"，期望在 model 下取
>
> RLHF 里 $`\text{KL}(\pi_\theta \| \pi_{\text{ref}})`$ = $`\text{KL}(\text{model} \| \text{target})`$，model 在第一个位置，所以是 reverse KL。原因很自然：RL rollout 本身就在按 $\pi_\theta$ 采样，所以 $`\mathbb{E}_{a \sim \pi_\theta}[\log \frac{\pi_\theta}{\pi_{\text{ref}}}]`$ 可以直接无偏估计。如果要算 forward KL $`\text{KL}(\pi_{\text{ref}} \| \pi_\theta) = \mathbb{E}_{a \sim \pi_{\text{ref}}}[\cdot]`$，就需要从 $\pi_{\text{ref}}$ 采样，但训练循环跑的是 $\pi_\theta$，没法直接估。
>
> | | Forward KL $\text{KL}(\pi_{\text{ref}} \Vert \pi_\theta)$ | Reverse KL $\text{KL}(\pi_\theta \Vert \pi_{\text{ref}})$ |
> |---|---|---|
> | KL 第一个参数 | target（固定） | model（优化中） |
> | 期望在谁下取 | $\pi_{\text{ref}}$（需从 ref 采样） | $\pi_\theta$（RL rollout 天然提供） |
> | 行为 | mean-seeking：$\pi_\theta$ 覆盖 $\pi_{\text{ref}}$ 所有模式，宁可摊薄也不漏 | mode-seeking：$\pi_\theta$ 集中到 $\pi_{\text{ref}}$ 的高概率区域，不给低概率区域分配概率 |
> | RLHF 适配 | ✗ 无法从 rollout 直接估计 | ✓ 天然可估，且防止策略跑到 ref 不支持的区域（不胡说八道） |
>
> **为什么 forward = mean-seeking，reverse = mode-seeking？**
>
> - **Forward KL** $\sum p \log \frac{p}{q}$：期望在 $p$ 下取。凡是 $p(x)>0$ 的地方，若 $q(x) \to 0$，则 $\log \frac{p}{q} \to +\infty$，KL 爆炸。所以 $q$ **被迫覆盖 $p$ 的所有模式**——哪怕摊薄概率也不能让任何模式裸露 → mean-seeking（宁可模糊也要全覆盖）。
> - **Reverse KL** $\sum q \log \frac{q}{p}$：期望在 $q$ 下取。若 $p(x)>0$ 但 $q(x)=0$，贡献为 $0 \cdot \log\frac{0}{p} = 0$，**无惩罚**——$q$ 可以放心忽略 $p$ 的某些模式。但若 $q(x)>0$ 而 $p(x) \to 0$，KL 爆炸——$q$ 绝不能在 $p$ 不支持的地方分配概率 → mode-seeking（锁定 $p$ 的一个高概率模式集中火力）。
>
> **OPD（On-Policy Distillation）常用 Reverse KL：** 学生模型用自己生成的样本（on-policy），天然在 $q_{\text{student}}$ 下采样，直接适配 reverse KL $`\text{KL}(q_{\text{student}} \| p_{\text{teacher}})`$ 的估计。且 mode-seeking 让蒸馏出的模型输出更 sharp、质量更高，不会在 teacher 的多个模式之间模糊平均（如 GKD: Generalized Knowledge Distillation）。

末端有 RM 奖励，中间 token 全是 KL 惩罚。这样 GAE 计算出来的 advantage 就同时考虑了"奖励"和"别跑偏"。

---

## 6. Actor-Critic 结构

- **Actor**： $\pi_\theta(a|s)$ ，输出动作分布
- **Critic**： $V_\phi(s)$ ，估计状态价值

两者通常共享底层网络（LLM 场景里：共享 Transformer，最后接两个 head：一个 LM head，一个 value head）。

```
                    ┌──── LM head ──→ π_θ(a|s)   (actor)
shared backbone ──→ │
                    └──── value head ─→ V_φ(s)    (critic)
```

Critic 训练目标：拟合 $R_t = A_t^{GAE} + V(s_t)$ （return target）。

---

## 7. 关键超参与稳定性技巧（实战经验）

| 超参              | 典型值          | 说明                                |
| --------------- | ------------ | --------------------------------- |
| clip $\epsilon$ | 0.1 – 0.2    | 太大不稳定，太小学不动                       |
| $\gamma$        | 0.99（RL），1.0（LLM）  | LLM 任务通常不折扣                       |
| GAE $\lambda$   | 0.9 – 0.97   | LLM RLHF 常用 0.95 或 1.0            |
| KL coef $\beta$ | 0.01 – 0.1   | 防止偏离 ref model                    |
| epochs / batch  | 3 – 10       | 越多越复用，但 ratio 偏离会变大               |
| minibatch size  | 越大越稳         | LLM 通常用 micro-batch + grad accum |
| value clip      | 同 actor clip | 防 critic 跳变                       |

**常见症状 → 诊断**

- ratio 平均值远离 1 / clip fraction 很高 → 学习率太大 / epochs 太多
- entropy 快速衰减 → 策略坍缩，加大 entropy coef
- KL(π_θ‖π_ref) 爆涨 → 加大 KL penalty β
- reward 涨但 win-rate 不涨 → reward hacking，检查 RM

---

## 8. LLM RLHF 视角下的完整 pipeline

```
SFT model π_SFT ─→ frozen reference π_ref
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  RLHF Loop                         │
        │                                    │
        │  prompt ──→ π_θ_old rollout ──→ y  │
        │                                    │
        │  RM(y) ──┐                         │
        │          ├─→ token reward r_t      │
        │  KL(π_θ‖π_ref) per-token ──┘       │
        │                                    │
        │  GAE → A_t, R_t                    │
        │                                    │
        │  PPO clip update θ                 │
        │                                    │
        │  π_θ_old ← π_θ                     │
        └────────────────────────────────────┘
```

每个 token 对应一个 (s, a, r)。reward 由「末端 RM 分数」+「per-token KL 惩罚」组成。

---

## 9. GRPO（DeepSeek 提出的变体）—— 省掉 Critic

PPO 在 LLM 上的痛点：要训练一个和 actor 同等规模的 **value model**（critic），显存翻倍。

**GRPO 的洞察**：在 LLM 场景里，每个 prompt 通常采样 $G$ 条响应。直接用**组内相对比较**作为 advantage：

```math
\hat A_i = \frac{R_i - \text{mean}(R_{1..G})}{\text{std}(R_{1..G})}
```

把 $R_{1..G}$ 的均值当 baseline，标准差归一化。

**好处**：
- **不需要 critic** → 显存 / 计算大幅下降
- 序列级别的 advantage，避免 LLM 中 token-level critic 不稳的问题
- 对 reward model 噪声更鲁棒（用相对而非绝对值）

**形式上仍然套 PPO 的 clip 框架**：

```math
L^{GRPO} = \mathbb{E}\left[\min(r_t \hat A,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A)\right] - \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})
```

只是 $\hat A$ 的算法换了。

> **🤔 有 clip 了为什么还要 KL？两者管的是完全不同的事**
>
> | | **Clip** | **KL 惩罚** |
> |---|---|---|
> | 对比对象 | $\pi_\theta$ vs $\pi_{\theta_{old}}$ （每批刷新） | $\pi_\theta$ vs $\pi_{\text{ref}}$ （**全程固定**的 SFT 模型）|
> | 时间尺度 | **单步**（每个 minibatch） | **全程**（整个 RL 训练） |
> | 解决问题 | importance sampling 数值失真 | reward hacking / 灾难性遗忘 |
> | 类比 | "今天最多走 1 公里" | "总位移不能离家太远" |
>
> **关键差别**： $\pi_{\theta_{old}}$ 每隔几个 epoch 就被刷新成当前 $\pi_\theta$ ——它**跟着移动**。clip 只管"今天走不远"，但**累积上百次 minibatch + 多轮 rollout， $\pi_\theta$ 可能已经飘到天涯海角**。而 $\pi_{\text{ref}}$ 是**死锚**（通常是 SFT 模型），KL 是把模型拽回它附近的"长程绳子"。
>
> ```
> π_ref ━━━━━━━ π_θ_old ━━━ π_θ
>  (家/死锚)     (今天起点)   (今天终点)
>    └─────── KL 限制 ──────┘
>                   └─clip─┘
> ```
>
> **KL 防的具体灾难**：
> - 🎣 **Reward hacking**：RM 不完美，模型可能学会"骗高分但语义崩坏"的模式 → KL 阻止它跑去诡异分布
> - 🧠 **灾难性遗忘**：RL 容易让模型只输出"高 reward 模板"，丢掉 SFT 的多样语言能力
> - 🛡️ **安全对齐**：防止把 SFT 阶段做的安全对齐"训没"
>
> **结论**：少了 clip → 单步数值崩；少了 KL → 长期累积漂移、语言能力被 reward hack 吃掉。**LLM RLHF 必须双保险**，这是经典 RL 场景里没有的特殊设计。

### 9.1 GRPO Loss 伪代码

把上面所有零件串起来，一个完整的 GRPO 训练 step 大致长这样：

```python
# === 1. Rollout：每个 prompt 采样 G 条响应（同时缓存 logp_old！） ===
prompts = sample_prompts(batch_size=B)
responses, logp_old_cache = [], []
for p in prompts:
    group, logps = [], []
    for _ in range(G):
        tokens, token_logps = policy_old.generate(p)  # ⭐ 采样时顺手拿到 logp_old
        group.append(tokens)
        logps.append(token_logps)                     # ⭐ 缓存下来，训练时直接用
    responses.append(group)
    logp_old_cache.append(logps)

# === 2. 打分：reward model 给每条完整响应打分 ===
rewards = reward_model(prompts, responses)       # [B, G]，标量奖励

# === 3. 组内相对 advantage（GRPO 核心） ===
mean = rewards.mean(dim=1, keepdim=True)         # 组内均值
std  = rewards.std(dim=1, keepdim=True) + 1e-8
advantages = (rewards - mean) / std              # [B, G]
# 同一条响应内所有 token 共享 sequence 级 advantage
advantages = advantages.unsqueeze(-1).expand_as(token_ids)  # [B, G, T]

# === 4. 训练时只需 forward 两次：current policy 和 ref policy ===
# logp_new：🔥 现算！梯度沿这里反传
y = policy(token_ids[:, :-1])                       # 网络输出 logits, [B, T, V]
logp_new = F.log_softmax(y, dim=-1).gather(         # ① log_softmax 归一化 → log 概率分布 [B,T,V]
    dim=-1, index=token_ids[:, 1:].unsqueeze(-1)    # ② gather 按"当时采样的 token id"抠出对应那一个
).squeeze(-1)                                       # 得到 [B, T]，每个位置一个标量 log π_θ(a_t|s_t)
# 等价写法：logp_new = -F.cross_entropy(y, target, reduction='none')

logp_old = logp_old_cache.detach()                  # ✅ 来自 rollout 缓存，无需再 forward
logp_ref = policy_ref.log_prob(token_ids).detach()  # 🔥 现算（同上两步），但 detach，无梯度

# === 5. Importance ratio + clip ===
ratio = torch.exp(logp_new - logp_old)           # r_t(θ)
clip_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
surr1 = ratio * advantages
surr2 = clip_ratio * advantages
policy_loss = -torch.min(surr1, surr2).mean()    # 取悲观估计

# === 6. KL 惩罚（vs ref，防止漂离 SFT） ===
# 常用的无偏估计：KL ≈ exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1
kl = torch.exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1
kl_loss = kl.mean()

# === 7. 总 loss + 反传 ===
loss = policy_loss + beta * kl_loss
loss.backward()
optimizer.step()

# === 8. 每隔若干步刷新 old policy ===
if step % update_old_every == 0:
    policy_old.load_state_dict(policy.state_dict())
```

### 📐 公式 ↔ 代码对照（为什么要算 logp？）

我们最终要最大化的目标函数是：

```math
L^{GRPO}(\theta) = \underbrace{\mathbb{E}\Big[\min\big(r_t(\theta) \hat A_t,\ \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat A_t\big)\Big]}_{\text{policy term}} - \beta \cdot \underbrace{\text{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})}_{\text{KL term}}
```

**`logp` 出现的根本原因**：公式里 $r_t$ 和 $\text{KL}$ 都是**概率的比值**，但概率本身数值很小、连乘会下溢——所以工程上**全程在 log 空间运算**，最后只在需要时 `exp` 回去。

| 公式中的项 | 数学定义 | 代码实现 |
|---|---|---|
| $\log \pi_\theta(a_t \mid s_t)$ | 当前策略对采样动作的 log 概率 | `logp_new`（**现算**，梯度沿此流） |
| $\log \pi_{\theta_{old}}(a_t \mid s_t)$ | 采样时策略的 log 概率 | `logp_old`（rollout 缓存） |
| $\log \pi_{\text{ref}}(a_t \mid s_t)$ | SFT 模型的 log 概率 | `logp_ref`（现算，但 detach） |
| $r_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ | importance ratio | `ratio = exp(logp_new - logp_old)` ← **log 差→exp = 比值** |
| $`\text{KL}(\pi_\theta \,\Vert\, \pi_{\text{ref}})`$ | 真实定义 $`\mathbb{E}_{\pi_\theta}[\log\frac{\pi_\theta}{\pi_{\text{ref}}}]`$ | k3 无偏估计：`exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1` |
| $-L^{GRPO}$ | 取负转为最小化 loss | `loss = policy_loss + beta * kl_loss` |

**串起来看**：
```
logits  ─log_softmax+gather→  logp_new  ┐
                                         ├→ ratio = exp(logp_new - logp_old) ─→ clip ─→ policy_loss
logp_old (缓存)                          ┘                                              ┐
                                                                                        ├→ loss ─→ backward
logp_ref (现算 detach)  ┐                                                              │
logp_new                ├→ KL ≈ exp(Δ) - Δ - 1 ─────────────────────→ kl_loss ─→ β·   ┘
```

**一句话**：算 `logp` 是为了**把目标函数里的"概率比"和"KL"在 log 空间稳定地表达出来**——它就是公式 ↔ 网络输出之间的桥梁。

#### 📖 KL 到底是什么？

**教科书定义**：衡量两个分布的差距，按 $p$ 采样取平均。

```math
\text{KL}(p \,\|\, q) = \mathbb{E}_{x \sim p}\!\left[\log \tfrac{p(x)}{q(x)}\right] = \sum_x p(x) \log \tfrac{p(x)}{q(x)} \;\;\geq 0
```

**套到 LLM 上**：每个 token 位置 $s_t$ 都是个词表上的分布 $\pi_\theta(\cdot|s_t)$ ，单点 KL 是：

```math
\text{KL}\big(\pi_\theta(\cdot|s_t) \,\|\, \pi_{\text{ref}}(\cdot|s_t)\big) = \sum_{v=1}^{V} \pi_\theta(v|s_t) \log \tfrac{\pi_\theta(v|s_t)}{\pi_{\text{ref}}(v|s_t)}
```

**痛点**：词表 $V$ 几万到十几万，对所有 $V$ 个 token 精确求和**太贵**。所以必须用**蒙特卡洛估计**——只用已采样的那个 token 估 KL。记 $\Delta_t = \log \pi_{\text{ref}}(a_t|s_t) - \log \pi_\theta(a_t|s_t)$ ，三种估计器：

| 估计器 | 公式 | 无偏？ | 非负？ |
|---|---|---|---|
| k1 | $-\Delta$ | ✅ | ❌ 可能为负 |
| k2 | $\tfrac{1}{2}\Delta^2$ | ❌ | ✅ |
| **k3** ⭐ | $e^\Delta - \Delta - 1$ | ✅ | ✅ |

> **🔍 怎么判断"无偏"？** 定义： $\hat\theta$ 是 $\theta$ 的无偏估计 ⟺ $\mathbb{E}[\hat\theta] = \theta$ 。**操作上只有一个动作**：把估计器套上 $`\mathbb{E}_{a \sim \pi_\theta}[\cdot]`$ 展开求和，看能不能化简成真值 $\text{KL}$ 。
> - **k1**： $\mathbb{E}[-\Delta] = \text{KL}$ ，直接就是 KL 的定义 ✅
> - **k2**： $\mathbb{E}[\tfrac{1}{2}\Delta^2] = \tfrac{1}{2}(\text{Var}(\Delta) + \text{KL}^2) \neq \text{KL}$ ❌（ $\Delta^2$ 是非线性，**期望和平方不能交换**）
> - **k3**：见下方推导

**k3 凭什么又无偏又非负？** 利用恒等式 $`\mathbb{E}_{a\sim\pi_\theta}\!\big[\tfrac{\pi_{\text{ref}}(a)}{\pi_\theta(a)}\big] = \sum_a \pi_{\text{ref}}(a) = 1`$ ，所以 $\mathbb{E}[e^\Delta] = 1$ ；又 $\mathbb{E}[\Delta] = -\text{KL}$ ，相减刚好 $\mathbb{E}[e^\Delta - \Delta - 1] = \text{KL}$ 。又因 $e^x \geq x+1$ 恒成立，k3 ≥ 0。**完美。**

> **🤔 那为什么不直接用 k1？** 它单次采样可能 < 0（虽然均值是 KL ≥ 0），梯度忽正忽负，数值抖动大。k3 在**每次采样**上都保证非负，又无偏又稳——所以胜出。

**对应到代码**就三行：

```python
delta = logp_ref - logp_new                  # [B, T]
kl_k3 = torch.exp(delta) - delta - 1         # 每个 token 一个非负标量
kl_loss = kl_k3.mean()
```

> 💡 **直觉**： $\Delta_t$ 是这个 token 在 ref 模型和当前模型下的 log 概率差。如果 ref 比当前更爱这个 token（ $\Delta > 0$ ），说明 $\pi_\theta$ 偏离了 ref， $e^\Delta - \Delta - 1$ 给出正惩罚；偏离越大惩罚越重。整条序列把所有 token 的 k3 加起来，就是 sequence-level KL。

**几个关键细节**：

- **没有 critic / value head**：和 PPO 最大的区别，省一半显存
- **`logp_old` 不用现算！** 在 rollout 阶段 `policy_old.generate()` 时模型已经 forward 过一遍，**顺手把每个采样 token 的 log prob 存下来**即可；训练时直接当数据读取。`logp_new` 和 `logp_ref` 才需要现算 forward
- **`logp_new` 怎么从 logits 算出来？** 网络输出 `y = [B, T, V]` 是每个位置上对整个词表的"原始分数"。两步即可：① `log_softmax(y)` 归一化成合法 log 概率分布 $\log p(v|s_t) = y_{t,v} - \log\sum_j e^{y_{t,j}}$ （用 log-sum-exp 数值稳定，**不要写成 `log(softmax(...))`**）；② `gather` 按"实际采样到的 token id"抠出对应那一个 log 概率。**和算 cross-entropy loss 的前两步完全一样**——这就是为什么 RL 在代码上看起来"就是加权监督学习"
- `policy_old` 和 `policy_ref` 是**两个不同的冻结模型**：前者每隔几步刷新（控单步幅度），后者全程不变（防长期漂移）。**实际部署中 `policy_old` 甚至不需要常驻显存**——它只在 rollout 时被用一次，logp 一旦缓存就可以释放
- `advantages` 在 **sequence 级**算，但**广播到所有 token** —— 一条响应里每个 token 共享同一个 advantage（这也是 Dr. GRPO 和 DAPO 后续争议的点）
- KL 用 **k3 无偏估计**而不是 $\log(p/q)$ ，数值更稳

---

## 10. GRPO 框架内的重要变体

GRPO 在 DeepSeek-R1 中大放异彩，但很快被发现存在多个隐藏问题。2025 年涌现出一批直接改 GRPO 框架的工作，每篇针对一个具体痛点——本节按以下顺序梳理：

| 时间 | 算法 | 一句话核心 |
|---|---|---|
| 2025.02 | **DAPO** (ByteDance) | 工程"完全体"：四件套修补 |
| 2025.03 | **Dr. GRPO** (Sea AI Lab) | 理论纠偏：去掉两处错误归一化 |
| 2025.01 | **REINFORCE++** (OpenRLHF) | 用 batch 级归一化代替组内归一化 |
| 2025.04 | **VAPO** (ByteDance) | 反潮流：把 critic 救回来 |
| 2025.05 | **GiGPO** (NTU) | 把 GRPO 搬到 agent 长链任务：嵌套组实现 step-level 信用分配 |
| 2025.06 | **CISPO** (MiniMax-M1) | clip 权重而非目标，所有 token 都能学 |
| 2025.07 | **GSPO** (Qwen) | 双重贡献：IS 粒度对齐 reward + 尝试解决 MoE 训练不稳 |
| 2025.08 | **GFPO** (Microsoft) | "多采样筛子集"：用拒绝采样治长度膨胀 |
| 2025.11 | **SAPO** (Qwen Team) | sigmoid 软门 + 非对称温度，替代硬 clip |
| 2026.01 | **GDPO** (NVlabs) | 多 reward 场景：先逐 reward 归一化、再加和，治"优势分辨率塌陷" |
| 2026.02 | **DPPO** (Sea AI Lab) | 用分布散度 mask 代替 ratio clip |

### 10.1 DAPO（ByteDance, 2025）—— GRPO 的工程"完全体"

全称：**D**ecoupled clip and **D**ynamic s**A**mpling **P**olicy **O**ptimization。在 GRPO 基础上做了四处关键改造：

**① Clip-Higher（解耦上下 clip）**

原 PPO/GRPO 的 clip 上下对称（ $[1-\epsilon, 1+\epsilon]$ ），但**"提升一个低概率正确动作的概率"和"压低一个高概率错误动作的概率"重要性不同**——前者对探索更关键。DAPO 把上界放宽：

```math
\text{clip}(r_t, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}),\quad \epsilon_{\text{high}} > \epsilon_{\text{low}}
```

> 直觉：**鼓励"敢于变化"，限制"过度退缩"** ——避免模型早早收敛、丢失探索多样性。

**② Dynamic Sampling（动态采样）**

GRPO 用组内相对 advantage，但若一组 $G$ 个回答**全对**或**全错**，std 退化、advantage 全为 0，**这组样本对梯度贡献为零**还占显存。DAPO **直接丢掉这种组**，重新采样直到组内有差异。

**③ Token-Level Policy Loss（token 级损失）**

GRPO 的 loss 在 **sequence 级**取平均（先按 token 平均、再按样本平均），导致**长回答的每个 token 权重被稀释**。DAPO 改成**直接在 token 级求平均**，让所有 token 平权——更适合长 CoT。

**④ Overlong Reward Shaping（过长样本软惩罚）**

**问题来源**：训练时为了控显存/吞吐，必须设最大生成长度 $L_{\max}$ （如 16K token）。超过就被**硬截断**，但接下来 reward 怎么给？

- ❌ **直接判错**（reward = 0 或 -1）：那些"快答出来但刚好被截断"的样本被冤枉了——其实推理路径是对的，只是没空间写完
- ❌ **丢掉这条样本**：浪费昂贵的 rollout 计算，且引入"长样本被系统性删除"的数据偏置

两种做法都让 reward 信号**噪声极大**：模型分不清"长 = 没答完 = 错"和"长 = 真的需要这么长才能推完"，结果是**对所有长回答一视同仁地惩罚**，最终学会"宁可短点说错也别长"——直接毁掉长 CoT 能力。

**DAPO 的做法：渐进式软惩罚**

设两个阈值： $L_{\text{cache}} < L_{\max}$ （如 $L_{\text{cache}} = L_{\max} - 4096$ ）。给每个样本加一个**长度惩罚项**：

```math
R_{\text{len}}(L) = \begin{cases}
0 & L \leq L_{\text{cache}} \quad\text{（正常区间，不干预）} \\
-\dfrac{L - L_{\text{cache}}}{L_{\max} - L_{\text{cache}}} & L_{\text{cache}} < L < L_{\max} \quad\text{（缓冲区，线性递增惩罚）} \\
-1 & L \geq L_{\max} \quad\text{（满惩罚）}
\end{cases}
```

最终 reward = $R_{\text{RM}} + R_{\text{len}}$ ，进入后续 GRPO advantage 计算。

```
惩罚强度
   0 ┤━━━━━━━━━━━━━┓
                    ┃ ╲
                    ┃   ╲   软惩罚区（线性渐增）
  -1 ┤              ┃     ╲━━━━━━━━━━━━━
     └──────────────┴───────┴──────────→ 长度 L
                L_cache    L_max
```

**为什么这样设计有效**：

1. **平滑过渡而非硬悬崖**：模型在接近 $L_{\max}$ 时**逐步**收到"该收尾了"的信号，而不是在 $L_{\max}$ 处突然从"无惩罚"跳到"满惩罚"——梯度方向更稳定
2. **保留"真值得长"的样本**：在 $L_{\text{cache}}$ 以内完成的长推理**不受惩罚**，模型不会被错误地教成"短就是好"
3. **区分"快截断" vs "真错"**：缓冲区的样本仍有部分 reward 信号，避免被冤枉的样本主导梯度
4. **保留 reasoning 能力**：这是为什么 DAPO 在长 CoT 数学任务上稳定大幅领先 GRPO 的关键之一

**🎯 一句话**：DAPO = GRPO + "敢探索 + 不浪费样本 + token 平权 + 软长度限制"，是当前 LLM RL 的事实工程标准。

### 10.2 Dr. GRPO（Sea AI Lab, 2025）—— 把 GRPO 的偏置修正回来

全称：**GRPO Done Right**。指出 GRPO 公式里有**两个隐藏偏置**会让训练朝错误方向漂移：

**① 长度偏置：错误回答越长，loss 越小**

GRPO 把每个回答的 loss 按其 token 数 $|o_i|$ 归一化。结果：
- 一个**错误**但**很长**的回答 → 单 token 梯度被稀释 → 模型受到的惩罚反而**轻** → **鼓励模型把错答案写得更长**
- 一个**正确**但**很长**的回答 → 单 token 梯度被稀释 → 鼓励的力度反而**弱**

**修正**：去掉 **per-sample 长度归一化**，但仍需归一化（裸求和会让 loss 数值随长度爆炸）。三种归一化对比：

| 做法 | 公式（核心部分） | 每 token 影响力 |
|---|---|---|
| GRPO | $\dfrac{1}{G}\sum_i \dfrac{1}{\lvert o_i \rvert}\sum_t \ell_{i,t}$ | ❌ 长回答被稀释 → 鼓励写长 |
| DAPO | $\dfrac{1}{\sum_i \lvert o_i \rvert}\sum_i\sum_t \ell_{i,t}$ | ✅ 所有 token 拍平求平均，**每 token 一票** |
| **Dr. GRPO** | $\dfrac{1}{G \cdot L_{\max}}\sum_i\sum_t \ell_{i,t}$ | ✅ 用**常数** $L_{\max}$ 归一化，每 token 一票且分母不随 batch 波动，**训练尺度更稳** |

> **共识**：必须**去掉 per-sample 长度归一化**；至于用"batch 总 token 数"还是"常数 $L_{\max}$ "做分母，DAPO 和 Dr. GRPO 各执一词，工程上都能 work。

**② 难度偏置：用 std 归一化会扭曲组间权重**

$\hat A_i = (R_i - \text{mean}) / \text{std}$ 里的 std 归一化让"组内方差小的题"（要么全对要么全错的简单/极难题）**被放大权重**，"方差大的题"（信息量最高的中等难度题）**反而被压低**。

**修正**：去掉 std 归一化，只减均值（保留 baseline 的方差缩减作用，去掉错误的重加权）。

```math
\hat A_i^{\text{Dr.GRPO}} = R_i - \text{mean}(R_{1..G})
```

**🎯 一句话**：Dr. GRPO 没改框架，只是**把 GRPO 公式里两处看似无害的归一化拿掉了**——结果训练效率和最终性能都明显提升。

### 10.3 REINFORCE++（OpenRLHF, 2025.01）—— 用 batch 级归一化代替组内归一化

**痛点**：GRPO 的 advantage $\hat A_i = (R_i - \text{mean})/\text{std}$ **只在每个 prompt 的 G 个回答内归一化**。当组内只有几个样本时，mean/std 都是不准的估计，且组间尺度可能差异很大，导致：
- **过拟合**：用 30 个 AIME 题训 GRPO，训练 acc 95% 但测试 Pass@1 = 0%
- **隐性偏差**：std 是有偏估计（Dr. GRPO 也指出过）

**核心设计：Global Advantage Normalization（全局归一化）**

把归一化范围**从"组内 G 个"扩大到"整个 batch（1024+ 样本）"**：

```math
\hat A_i = \frac{R_i - \text{mean}(R_{\text{batch}})}{\text{std}(R_{\text{batch}})}
```

> **直觉**：根据大数定律，batch 足够大时，全局均值/方差**收敛到稳定常数**，估计偏差消失。这相当于"用整个训练集的平均水平做 baseline"，比"只看这一道题的 G 个回答"信号更稳。

**其他改动 vs GRPO**：

| 项目 | **GRPO** | **REINFORCE++** | 直觉 |
|---|---|---|---|
| **KL 注入位置** | sequence-level，作为独立 loss 项 $\beta \cdot \text{KL}$ | **per-token reward**： $\tilde r_t = r_t - \beta \cdot \text{KL}_t$ | 把 KL 揉进每个 token 的 reward → KL 走 advantage 估计 → 长序列**自然累计更多 KL 惩罚**，credit assignment 更细 |
| **PPO clip** | ✅ 有 | ✅ 保留 | 信任域约束依然必要，没必要重新发明 |
| **是否必须分组** | ✅ 必须 G 条/prompt | ❌ 纯 REINFORCE++ **不分组**（每 prompt 1 条）；REINFORCE++ w/ Baseline 才保留分组 | 不分组时 rollout 成本降 G 倍；prompt 多样性高时反而更优 |

> **关键 takeaway**：REINFORCE++ 不是"换个名字的 GRPO"，而是**回到 REINFORCE 的纯净形式 + 借 PPO 的 clip + 用 batch 统计代替组统计**——结果发现既能不要 critic，又能不要分组。这是对"GRPO 必须组内归一化"这个隐含假设的釜底抽薪。

> **📍 "KL 注入位置" 怎么理解？关键：KL 在 REINFORCE++ 里被当作"负 reward"，会改变 advantage**
>
> 两种做法的真正区别不在"per-token 与否"（GRPO 的 k3 KL 在 backprop 上也是 per-token 的），而在 **KL 走哪条梯度路径**：
>
> ```
> GRPO（KL 走独立 loss 项）:
>   RM 打分 ──→ reward ──→ advantage ──→ policy loss ┐
>                                                     ├── 加起来反传
>   KL k3   ──→ kl_loss ──────────────────────────────┘
>           （KL 和 advantage 在两条独立的路上）
>
> REINFORCE++（KL 混入 reward）:
>   RM 打分 ┐
>           ├──→ r̃_t = r_t - β·KL_t ──→ advantage ──→ policy loss ──→ 反传
>   KL k3  ─┘
>           （只有一条路；KL 已经溶解在 advantage 里）
> ```
>
> **后果对比**（仍用 100 token、第 50 个偏离 ref 的例子）：
>
> | | GRPO | REINFORCE++ |
> |---|---|---|
> | KL 梯度路径 | 直接落到位置 50 的 logp 上 | 通过 $\tilde r_{50} \to \hat A_t$ 影响 **位置 1~50** 的 logp |
> | 作用范围 | 只在第 50 个 token 上"拉向 ref" | 第 1~50 个 token 的 advantage 都被拉低 |
> | 含义 | **局部正则化**："这个位置的分布偏了，拉回来" | **回溯归因**："是前面的决策导致走到这个高 KL 状态，前面也要改" |
> | 类比 | "这道题答错了，扣这道题分" | "这道题错是因为前面理解错，前面也要重做" |
>
> **🔑 一句话核心**：REINFORCE++ 把 KL 当作**负 reward**，从而让 KL **自动享受 advantage 估计的 credit assignment 机制**（reward-to-go、因果性、GAE 折扣等都自动适用于 KL）。GRPO 的 KL 则是个外挂正则项，只做局部"拉回 ref"。
>
> **"长序列自然累计更多 KL 惩罚"**：在 REINFORCE++ 里， $\hat A_t = \sum_{t'\geq t}\tilde r_{t'}$ ，长回答里的 KL 通过累加传到前面所有位置 → 模型对长序列的漂移更敏感（防长 CoT "越想越歪"）。

**🎯 一句话**：REINFORCE++ 论证了"GRPO 的组内归一化不是必需的，全局归一化反而更稳更通用"。

> **🆚 和 DAPO 的关键区别：管的是完全不同的两件事**
>
> 两者都说"用 batch 做事"，但作用位置不同：
>
> | 项目 | **DAPO** | **REINFORCE++** |
> |---|---|---|
> | 改造对象 | **Loss 归一化**（算完 advantage 之后） | **Advantage 归一化**（算 advantage 本身）|
> | Advantage 怎么算 | **组内** $(R-\text{mean}_G)/\text{std}_G$ （同 GRPO） | **全局** $`(R-\text{mean}_{\text{batch}})/\text{std}_{\text{batch}}`$ |
> | 解决问题 | 长回答 token 被稀释 | 组间 baseline 信号不一致 |
>
> ```
> rollout → R_i → ① advantage 归一化 → ② loss 归一化 → backward
>                      ▲                       ▲
>                REINFORCE++ 改                 DAPO 改
>                （组内→全局）              （per-sample→per-token）
> ```
>
> **直觉差异**：
> - **GRPO/DAPO 组内归一化**："在这道题里你算第几名"——题内相对，**忽略题目难度**
> - **REINFORCE++ 全局归一化**："在 batch 所有回答里你算第几名"——跨题绝对，**保留题目难度信号**
>
> **🔑 结论：完全互补，工程上常叠加使用**：advantage 用 REINFORCE++ 的全局归一化 + loss 用 DAPO 的 batch-token 平均，互不冲突。Qwen3 等主流配方就是这种组合。

### 10.4 VAPO（ByteDance, 2025.04）—— 把 critic 请回来

**反潮流观点**：DAPO/GRPO 都把 critic 砍了，VAPO 反问：**"长 CoT 任务上 critic 真的没用吗？"**

ByteDance 团队认为 critic-free 方法在长推理任务上有**理论天花板**——没有 value function 就没法做精确的 token 级 credit assignment（信用分配），advantage 估计方差天然大。

**但 critic 在 LLM 上为什么训不好？** 三个老大难：

1. **长序列上 value 偏差累积**
   > Critic 通过 TD bootstrap 学习—— $V(s_t)$ 的目标依赖 $V(s_{t+1})$ 的预测。任何一步预测有误差，就**沿着序列向前传**。生成 1000 token 时，误差像滚雪球一样累积，越靠前的位置预测越离谱。
   >
   > 🚗 类比：开车每公里预测一次"到目的地还要多久"。每次预测都用上次预测+本公里实际耗时，**每次预测的小误差会被下一次预测继承**。1000 公里下来，最初的预测早已和现实毫无关系。

2. **不同长度的序列 value 不一致**
   > 短回答(50 token,数学题)的 value 范围可能是 $[0, 1]$ ；长回答(5000 token,复杂推理)累计起来可能是 $[-5, 5]$ 。**同一个 critic 网络要同时学两个差几倍的输出尺度**，结果两边都学不好——短的被长的"带偏"，长的又因为输出层 saturation 学不进去。
   >
   > 🌡️ 类比：用同一个温度计同时测**室温(20℃)** 和**冰箱(-5℃)** 还要测**烤箱(200℃)**——刻度尺度差太多,中间档位就失准。

3. **稀疏 reward(最后才给分)下 value 学不动**
   > 数学题里 reward 只在末尾给 1 分,中间 999 个 token 的真实回报全靠**从末尾沿着轨迹折扣回传**。但 critic 自己也是个估计器,**回传过程中信号被 $\gamma$ 衰减、被 $V$ 自身的预测噪声淹没**——前 900 个 token 根本看不到清晰信号,只能瞎猜。
   >
   > 🎯 类比:在 1000 米外有个靶子,**只有最后 1 米能看清方向**,前 999 米只能凭感觉走。Critic 在中间位置等于"盲走"。

**核心设计：5 个技巧把 critic 救回来**

**① Value-Pretraining（先单独训 critic 一段时间）**
> **痛点**：Critic 通常用 reward model 的权重初始化，但两者目标完全不同——RM 判"整条回答好不好"，value 估"这个状态下未来期望回报"。初始化错位 → 训练开始时 critic 乱预测 → policy 跟着乱学 → 一起 collapse。
>
> **解法**：先冻结 policy，**只训 critic** 几千步（用现有 rollout 数据，目标是真实累计回报），让它先"站稳脚跟"再开始联合训练。
>
> 🏋️ 类比：教练上场前先让球员**自己练几天**，别一上来就跟主队配合——配合得了的前提是个人技能先在线。

**② Decoupled-GAE（policy 和 value 各用一套 GAE）**
> **🔁 先快速回忆一下 $\lambda$**（GAE 公式 $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$ 里的旋钮）：
> - $\lambda = 0$ ：一步 TD，**完全信 critic** 的 $V(s_{t+1})$ 预测剩下 → 信号平滑（**低方差**），但 $V$ 错了就跟着错（**高偏差**）
> - $\lambda = 1$ ：蒙特卡洛，**完全不信 critic**，把所有未来真实 reward 都加进来 → 真实（**低偏差**），但每条轨迹差异大（**高方差**）
> - $\lambda \in (0,1)$ ：在两者间插值——"信 $V$ 几分、信现实几分"的连续旋钮
>
> **💡 等等，训 critic 和 $\lambda$ 有什么关系？** Critic 训练是个 MSE 回归 $L_V = (V_\phi(s_t) - V_{\text{target}})^2$ ，但 **target 怎么算？答案就是用 $\lambda$ 构造**：

```math
V_{\text{target}}^{(\lambda)} = V(s_t) + \hat A_t^{(\lambda)} \;\;\Longrightarrow\;\; \begin{cases} \lambda=0:\; r_t + \gamma V(s_{t+1}) & \text{(bootstrap，target 平滑)} \\ \lambda=1:\; \sum_l \gamma^l r_{t+l} = G_t & \text{(MC，target 真实但嘈杂)} \end{cases}
```

> 所以 $\lambda$ 直接决定**给 critic 看的"标准答案"是什么**——标签方差大就学不动，标签方差小就稳。
>
> **痛点**：同一个 $\lambda$ 既要给 policy 用、又要给 value 用，但**两者在用 advantage 时角色完全不同**，自然偏好不同 $\lambda$ ：
>
> | | 用 advantage 干什么 | 怕"偏" | 怕"抖" | 偏好 $\lambda$ |
> |---|---|---|---|---|
> | **Policy**（策略梯度） | 当**方向引导**（ $\nabla L \propto \hat{A} \cdot \nabla \log \pi$ ） | ❌ 偏了就**学歪一去不回** | ✅ 抖也无所谓，**平均下来方向对** | **大**（低偏差优先）|
> | **Value**（MSE 回归） | 当**回归标签**（ $V_{\text{target}} = \hat{A} + V$ ） | ✅ 偏移可后续校正 | ❌ 标签嘈杂 → MSE 被方差支配，**根本学不动** | **小**（低方差优先）|
>
> 一个共享 $\lambda$ 满足不了两边，互相拖累。
>
> 🎲 **类比**：
> - **Policy** 像走路找出口、有人指方向：**指错方向（偏差）就完蛋**；**手抖来回晃（方差）多看几次就平均出真方向**
> - **Value** 像画地图、有人报坐标：**坐标整体偏移（偏差）可整体校正**；**坐标到处乱跳（方差）根本画不出地图**
>
> **解法**：直接拆成两个独立的 $\lambda_{\text{policy}}$ （设大，如 0.95~1.0）和 $\lambda_{\text{value}}$ （设小，如 0.5~0.9），**各得其所**。

**③ Length-adaptive GAE（ $\lambda$ 随序列长度自适应）**
> **痛点**：固定 $\lambda$ 对所有长度的序列一刀切，但**短序列和长序列对 critic 的信任度应该不同**：
> - 短序列：critic 偏差累积少 → 可以多信它 → 用**小** $\lambda$
> - 长序列：critic 偏差累积大 → 少信它，多看真实 reward → 用**大** $\lambda$
>
> **解法**： $\lambda$ 是序列长度的函数,自动调节。

**④ Clip-Higher（同 DAPO,上下 clip 解耦)**
> 直接复用 DAPO 的 Clip-Higher：放宽**涨概率**的上界,严守**降概率**的下界,鼓励探索、防止熵塌缩。

**⑤ Positive Example LM Loss(对答对的样本加个 SFT loss)**
> **痛点**:数学等困难任务里**正样本极稀少**(可能 1000 个 rollout 只有 30 个答对),而 RL 的学习信号几乎全靠它们——但 RL 一次梯度只能"利用"这些样本一次,**太浪费**。
>
> **解法**:对答对的样本**额外加一个标准 SFT-like 的交叉熵 loss**(直接最大化 $\log\pi_\theta$),榨干每个正样本的学习价值。
>
> 📝 类比：学生**做对**一道难题,不只给奖励,还**让他抄一遍**加深印象——同一个正样本被利用两次。

**结果**：在 AIME 2024 上 Qwen-32B 跑出 60.4 分，**比 DAPO 高 10+ 分，比 DeepSeek-R1-Zero 也高**。

**🎯 一句话**：VAPO 证明 critic 不是负担，**前提是你愿意花心思把它训稳**——这是工程 vs 简洁性的取舍。

### 10.5 CISPO（MiniMax-M1, 2025.06）—— 别把"思考关键词"clip 掉了

#### 🔑 重新认识 PPO clip：它是"门控"，不是"幅度限制"

理解 CISPO 之前，必须先看清 PPO clip 的**真实机制**——这跟大多数人的直觉不一样。

直觉上，clip 听起来像"限制优化幅度"（ratio 越界就把步子缩小一点继续走）。**但 PPO clip 实际上是个二值门控**：

```
理想的"限制幅度":                    PPO clip 的真相:
ratio 大 → 推它，步子小一点          case ①/④ → 梯度归零，完全放弃这个 token
ratio 小 → 推它，步子大一点          case ②/③ → 梯度完整保留
→ 永远在推，只是力度不同              → 要么 0 要么完整，"二值开关"
```

PPO loss $-\min(r_t \hat A, \text{clip}(r_t)\hat A)$ 在 case ① 时退化成 **纯常数** $-(1+\epsilon)\hat A$ ，整个 loss 不再含 $\theta$ —— 这个 token 在反传时**完全消失**，不是"步子变小"，是"压根不更新"。

四种情况看清楚：

| 情况 | $\hat A$ | $r_t$ | min 选哪个 | 梯度 | 含义 |
|---|---|---|---|---|---|
| ① | $> 0$ | $> 1+\epsilon$ | **clipped** | ❌ 零 | "好 token 已推够了 → 完全放弃" |
| ② | $> 0$ | $< 1-\epsilon$ | unclipped | ✅ 正常 | "好 token 反而掉了 → 继续推涨" |
| ③ | $< 0$ | $> 1+\epsilon$ | unclipped | ✅ 正常 | "坏 token 反而升了 → 继续推降" |
| ④ | $< 0$ | $< 1-\epsilon$ | **clipped** | ❌ 零 | "坏 token 已抑制够了 → 完全放弃" |

**关键洞察**：PPO clip 的设计哲学是**消极的单边门控**——"如果你跑出安全区**还往同方向跑**，我就当作没看见你"。这本是 Schulman 2017 为了**廉价近似 TRPO 的 trust region 约束**做的简化，但代价就是 case ①/④ 的"全有或全无"。

#### 🚨 这个门控在长 CoT 上为什么致命

考虑像 "However"、"Wait"、"Recheck"、"Aha" 这些**低概率"分叉 token"**——它们是触发深度推理和自我纠错的关键，但因为概率低、 $\hat A$ 又大，**最容易落入 case ①**：

- **第 1 次更新**：模型学到 "Wait" 是好 token → 概率猛涨， $r_t = 2.0 \gg 1+\epsilon$
- **第 2 次更新**起：落入 case ① → **policy 梯度 = 0**
- 更糟的是： $\pi_\theta$ 此时远大于 $\pi_{\text{ref}}$ → **KL 项反向把它拽回 ref**（ref 本来就不爱这个 token）
- **净效果**：这个稀缺的有用 token 不仅没被继续鼓励，反而被自己的 KL 制裁掉了

PPO/GRPO 一般 1 个 batch 跑 2-4 步更新，影响有限。**但 MiniMax-M1 跑 16 步更新**，第 1 步后这些关键 token 就被消音 15 步——这就是 CISPO 要救的场景。

> **顺便注意**：这是 PPO clip **整个家族的共同问题**，不止 CISPO 关心：
> - **DAPO 的 Clip-Higher**：把 $\epsilon_{\text{high}}$ 调大，**推迟**门关上的时机——治标
> - **GSPO**：升到 sequence-level ratio，**用统计平均吸收 token 级波动**——绕过
> - **CISPO**：从根上换 loss 形式，**让 logp 始终有梯度**——治本

#### 🩹 CISPO 的修法：把 $\log\pi_\theta$ 显式写回 loss

PPO 把 $\theta$ 的依赖**全压在 $r_t$ 这一个通道里**，clip 切了 $r_t$ ， $\theta$ 依赖也就没了。CISPO 改成"权重 + 显式 logp"形式：

```math
L_{\text{CISPO}} = -\mathbb{E}\Big[\underbrace{\text{stop\_grad}(\min(r_t,\ \epsilon_{\text{high}}))}_{w_t,\text{ 常数权重}} \cdot \hat A_t \cdot \underbrace{\log \pi_\theta(a_t|s_t)}_{\text{显式可导}}\Big]
```

```python
ratio = torch.exp(logp_new - logp_old)
weight = torch.clamp(ratio, max=eps_high).detach()    # ⭐ clip + detach 当权重
loss = -(weight * advantage * logp_new).mean()         # logp_new 显式保留
```

**关键性质**：

- 即使权重 $w_t$ 被卡在 $\epsilon_{\text{high}}$ （case ①），梯度仍为 $-\epsilon_{\text{high}} \cdot \hat A_t \cdot \nabla\log\pi_\theta \neq 0$
- **$\log\pi_\theta$ 始终在 loss 里** → $\theta$ 的依赖永远不会被 clip 切断
- 步长被 $\epsilon_{\text{high}}$ 控制，**不会失控**；但方向**始终在推这个 token 涨**

#### 🆚 直觉对比

| | 遇到 ratio 越界时的行为 |
|---|---|
| **PPO 风格** | "你已经涨过头了 → **完全放弃**这个 token"（门控关闭，梯度为 0）|
| **CISPO 风格** | "你已经涨过头了 → **降低你的影响力**（权重封顶），但**继续学**"（步长有限，但梯度还在）|

后者保证**所有 token 都能持续贡献梯度**，关键的低概率 token 不会被消音也不会被 KL 反向制裁。

#### 📊 结果与意义

**结果**：在 Qwen2.5-32B 数学任务上比 DAPO 快 **2 倍**；MiniMax-M1 在 512 H800 三周训完整模型（成本 ~\$534k）。

**🎯 一句话**：CISPO 抓住了 PPO clip 的设计缺陷——**它是消极门控，不是积极幅度限制**——通过"clip 权重 + 显式 logp"把这个缺陷修掉，让稀缺有用的"分叉 token"在多步更新中**持续被学习**。这也是为什么它在**长 CoT + 多步更新**这种边界场景下能跑出显著优势。

### 10.6 GSPO（Qwen, 2025.07）—— 粒度对齐 + MoE 稳定化的双重贡献

GSPO 有**两个并列的核心贡献**，都很重要：

1. **理论层面**：指出 GRPO 的 token-level importance sampling 与 sequence-level reward **粒度错位**，是个长期被忽略的设计缺陷
2. **工程层面**：尝试解决 MoE 模型 RL 训练的不稳定问题，让 Qwen3 这类 MoE 大模型的 RL 训练第一次跑通到生产级别

下面分两块讲。

#### 🚨 贡献一：GRPO 隐藏的"粒度不对齐"问题

| | GRPO 的 importance ratio | GRPO 的 reward |
|---|---|---|
| 粒度 | **per-token**： $r_t = \dfrac{\pi_\theta(o_{i,t} \mid q, o_{i,\lt t})}{\pi_{\theta_{old}}(o_{i,t} \mid q, o_{i,\lt t})}$ | **per-sequence**： $R(q, o_i)$ 整条响应一个分 |
| 单位 | 每个 token 一个数 | 整条序列一个数 |

**两个问题叠加**：

1. **粒度不对齐**：reward 是"整条回答好不好"，IS 校正却在"每个 token"上做——单位都不一样，**强行把 sequence 级信号塞到 token 级权重上**
2. **单点采样让 IS 退化**：重要性采样的方差控制依赖**多次采样取期望**，但每个位置 $t$ 上的 $r_t$ 只用了**一个**实际采样到的 token——**根本不是有效的 IS 估计，只是个噪声极大的单点比值**

> **🎯 一句话**：GRPO 的 $r_t$ 看起来像在做重要性采样校正，**实际上它既不在 reward 的对应粒度上，也不满足 IS 的多采样前提**。这个"假 IS"带来的方差**随响应长度累积**，又被 clip 进一步放大——长 CoT 训练上的不稳定性，根源就在这里。

#### 🔥 贡献二：尝试解决 MoE 训练的 token-level ratio 崩溃

粒度不对齐在 Dense 模型上虽然存在，但参数更新对每个 token 概率的扰动是**平滑、连续**的，方差还能扛——所以 GRPO 在 Dense 上勉强能用。**但在 MoE 模型上这个隐藏问题被直接放大到训练崩溃**：

> **实验数据（Qwen3-30B-A3B）**：一次梯度更新后，**约 10% 的 token 激活了和上次完全不同的 expert**。同一个 token 经过不同 expert，输出分布可能**截然不同** → $r_t$ 在这些位置剧烈跳变 → 训练几乎必然 collapse。

**Qwen 团队此前的尝试**：用 **Routing Replay**（强制新策略复用旧策略的路由）当补丁——能跑，但代价惨重：
- 显存翻倍（要缓存所有 token 的 routing mask）
- 限制模型自由探索新的 expert 组合，长期压制模型容量
- 工程链路复杂（rollout 和 training 引擎要严格同步）

**GSPO 的双重价值**：粒度对齐让 token-level 单点噪声从根上消失，**MoE routing 漂移自然不再是问题**——这是 GSPO 直接终结了"MoE RL 必须靠 Routing Replay 打补丁"这个时代，让 Qwen3-235B-A22B 这种 235B 参数的 MoE 第一次有了**算法干净、工程简单**的 RL 训练路径。这不是顺带的副产品，**是和粒度对齐并列的核心贡献**。

#### 💡 核心设计：sequence-level importance ratio + length normalization

既然 reward 在 sequence 级，那 IS 也升到 sequence 级，**粒度自然对齐**：

```math
s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{old}}(y_i \mid x)}\right)^{1/\lvert y_i \rvert} = \exp\!\left(\frac{1}{\lvert y_i \rvert}\sum_{t=1}^{\lvert y_i \rvert} \log\frac{\pi_\theta(y_{i,t} \mid x, y_{i,\lt t})}{\pi_{\theta_{old}}(y_{i,t} \mid x, y_{i,\lt t})}\right)
```

> **为什么要做长度归一化（ $1/|y_i|$ 次方）？** 直接用 $\pi_\theta(y_i|x)/\pi_{\theta_{old}}(y_i|x)$ 是几百个 token 概率比的连乘——**长度稍长就指数爆炸 / 趋零**。开 $|y_i|$ 次方（几何平均）后， $s_i$ 的数值范围**和长度脱钩**，clip 阈值才能用统一的 $\epsilon$ 。
>
> ⚠️ **注意**：严格来说这是个**有偏但低方差**的近似——标准 IS 理论里序列级比值就应该是 $\pi_\theta(y)/\pi_{old}(y)$ ，开方后已经不是无偏估计，但工程上方差降下来了反而稳。

Advantage 仍然是 GRPO 的组内归一化（reward 本身就是 sequence 级，自然对齐）：

```math
\hat A_i = \frac{R(x, y_i) - \text{mean}(\{R(x, y_j)\}_{j=1}^G)}{\text{std}(\{R(x, y_j)\}_{j=1}^G)}
```

clip 和 loss 也都升到 sequence 级——**整条序列要么全用，要么全 clip**：

```math
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \min\!\big(s_i(\theta)\hat A_i,\ \text{clip}(s_i(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat A_i\big)\right]
```

#### 🤔 为什么 sequence-level 稳？两个机制

1. **粒度对齐**：reward 和 IS 现在都是 sequence 级，**信号单位统一**，不再有"sequence 信号塞 token 权重"的错位
2. **统计平均**：几何平均把每个 token 位置的随机波动（MoE 路由变化、单点采样噪声）**平均化** ——上百个 token 的乘积稀释了单 token 的剧烈跳动

#### 📊 工程收益

- **不再需要 Routing Replay** → 显存省一半，MoE router 可以自由学习
- **对 inference 引擎精度容错高**：rollout 时直接用 inference engine 的 logp 都行（不用 training engine 重算）
- **多轮 RL / partial rollout 工程上大幅简化**

#### 🤯 反直觉发现

GSPO 中**被 clip 的 token 比例比 GRPO 高两个数量级**，但训练效率反而更高——说明 **GRPO 那些 token-level 信号本身就是噪声**，"clip 掉更多"反而是"过滤了更多假信号"，sequence-level 的少量信号反而更可靠。这从实验侧反证了"GRPO 的 token-level IS 是假的"这一论点。

#### ⚖️ 代价：失去 token 级 credit assignment

GSPO 不是没代价——一条序列只有一个 $s_i$ ，意味着**所有 token 共享同一个 IS 权重**，丧失了细粒度信用分配能力。对于"哪个 token 关键、哪个 token 是 filler"这种问题，GSPO 看不到。后续 DHPO 等工作正在尝试 token + sequence 混合。

**结果**：是 **Qwen3-235B-A22B-Instruct-2507** 等系列模型背后的训练算法。

**🎯 一句话**：GSPO 同时做了两件大事——(1) **理论上**指出 GRPO 的 token-level IS 与 sequence-level reward 粒度错位、是个"假 IS"；(2) **工程上**彻底解决 MoE 训练的稳定性问题，让 235B 级 MoE 的 RL 训练首次脱离 Routing Replay 补丁。两个贡献都是核心。

### 10.7 GFPO（Microsoft, 2025.08）—— "多采样、少思考"：用拒绝采样治长度膨胀

**痛点：Length Inflation（长度膨胀）**

RLVR 训出的推理模型为了拿"正确"奖励，会把回答写得越来越长——但**大多数额外 token 是重复/无意义的 filler**，对正确性没贡献甚至有害。Phi-4-reasoning 上的实测数据触目惊心：

- GRPO 训练后，**平均响应长度从 ~4k token 暴涨到 ~14k token**
- 在 **72% 的 AIME25 题目上，更长的回答反而更容易错**
- Dr. GRPO / DAPO 的 token-level 归一化能修一点，但**没法根治**——只要 reward 信号在，模型就会把"对一点点"的长答案继续强化下去

**为什么 reward shaping 治不了？** 一种直觉做法是"在 reward 里加长度惩罚 $r \leftarrow r - \alpha L$ "。但这会和 RM 信号纠缠在一起： $\alpha$ 太小没效果，太大模型学会**为了短而短**——典型的 reward hacking。GFPO 的思路是**绕开 reward**，直接在**样本选择**层面塞入"想要什么"。

**核心思路：多采样 → 按简洁性筛子集 → 只在子集内做 GRPO**

把"采 $G$ 条全用"改成"**采更大的 $G$ 、只留 top- $k$**"：

1. 对每个 prompt 采样 $G$ 条响应（ $`G \in \{8, 16, 24\}`$ ，**多于 GRPO**）
2. 按简洁性指标给每条响应打分： $\text{scores}_i = \text{metric}(o_i)$
3. 排序后**只保留 top- $k$**（ $k \le 8$ ，**和原 GRPO 同等大小**）→ 保留集合 $S$
4. 构造 0/1 mask： $m_i = \mathbb{I}[i \in S]$
5. **在保留子集内重新归一化** advantage（mean/std 都只对 $S$ 内的样本算）：

```math
\hat A_{i,t}^{(m)} = \frac{R(q, o_i) - \text{mean}\{R(q, o_{s_1}), \dots, R(q, o_{s_k})\}}{\text{std}\{R(q, o_{s_1}), \dots, R(q, o_{s_k})\}} \cdot m_i
```

6. PPO clip loss 照旧，只是被 mask 掉的样本 $m_i = 0$ ，**不贡献梯度**：

```math
L^{\text{GFPO}} = \mathbb{E}\!\left[\min\!\big(r_t \hat A^{(m)}_{i,t},\ \text{clip}(r_t, 1{-}\epsilon, 1{+}\epsilon)\hat A^{(m)}_{i,t}\big)\right]
```

> **🎯 直觉**：以前是"答对就奖励，不管长短"；现在是"答对**且简洁**才进入梯度池"。**长度信号通过样本筛选直接进入训练循环**，不经过 RM——RM 只判对错，简洁性由筛选器把关，两者解耦、互不污染。

**两个核心筛选指标**：

| 指标 | 评分函数 | 行为 |
|---|---|---|
| **Shortest- $k$**（最短优先） | $\text{metric}(o_i) = -\lvert o_i \rvert$ | 简单粗暴选最短的 $k$ 条，最激进的压缩 |
| **Token Efficiency**（性价比优先） | $\text{metric}(o_i) = R(o_i) / \lvert o_i \rvert$ | "每 token 回报"——长答案只要"配得上"也能进 |

> Shortest- $k$ 一刀切；Token Efficiency 更聪明——**简单题强迫短，难题允许长**。论文里 Token Efficiency 在长度压缩上更激进、准确率不掉。

**Adaptive Difficulty GFPO（自适应难度变体）**

固定 $k$ 对所有题不公平：简单题留 4 条足够，难题留 8 条都不一定有 1 条对的。**Adaptive Difficulty 用组内平均 reward 在线估题目难度**：

- 难题（reward 低）→ **留更多样本**（ $k$ 大），让稀缺正样本有机会进梯度
- 简单题（reward 高）→ **少留**（ $k$ 小），把训练算力倾斜到真正需要的题目

> 这是个**资源分配**的小技巧：训练算力是固定的，与其平均分配，不如倾斜给高信息量的难题。

**Loss 归一化**：GFPO 沿用 **DAPO 风格的 token-level 归一化**（按 batch 总 token 数）—— 因为目标就是治长度，绝不能再用 GRPO 的 per-sample 归一化（那会重新放大"长答案 token 权重被稀释"的偏置）。

**实验结果（Phi-4-reasoning, 14B）**：

- 长度膨胀降低 **46–71%**（across AIME 24/25, GPQA, Omni-MATH, LiveCodeBench），**准确率不掉**
- 改用 Token Efficiency 指标，长度压缩进一步到 **71–85%**
- **7% 训练时间换 ~30% inference 延迟下降**（难题最多省 90 秒）
- 在 LiveCodeBench（**训练时未见过的 code 任务**）上仍 work → 对压缩学到的"简洁性"有跨域泛化能力
- 在准确率和长度压缩上都**优于 Dr. GRPO**

**🎯 一句话**：GFPO 把"想要什么属性"（短而对）从 **reward shaping 移到样本选择**——花训练时的额外采样算力，换永久的 inference 提速。是当前治"长度膨胀"最干净的方案，且天然兼容 DAPO/GRPO 的其它 trick（叠加使用即可）。

> **🆚 和 DAPO Overlong Soft Penalty 的关系**：两者都治长度，但作用机制完全不同。
> - **DAPO 软长度惩罚**：在 reward 里加 $R_{\text{len}}$ ，仍是 reward shaping，只针对**超过 $L_{\text{cache}}$ 的 outlier**做缓冲——治"答不完被硬截断"的偏置
> - **GFPO 拒绝采样**：直接在**样本选择**层面筛简洁性，治的是"平均答案越训越长"的整体膨胀
>
> 两者正交，可叠加：GFPO 控总体长度，DAPO 软惩罚处理边界 case。

### 10.8 SAPO（Qwen Team, 2025.11）—— 用 sigmoid"软门"替代硬 clip 的连续信任域

全称：**S**oft **A**daptive **P**olicy **O**ptimization (arXiv:2511.20347)。Qwen3-VL 系列的训练算法。

#### 痛点：硬 clip 是个"全有/全无"的二值开关

PPO clip 的本质（10.5 CISPO 节里详细分析过）——超出 $[1-\epsilon, 1+\epsilon]$ 的 token 在某些方向上**梯度直接归零**，是个二值门控。这逼出一个两难调参：

| clip 范围 | 优点 | 缺点 |
|---|---|---|
| **窄**（ $\epsilon = 0.1$ ）| 稳定 | 大量 token 被截断 → 有效样本少、探索不足、学不动 |
| **宽**（ $\epsilon = 0.3$ ）| 信号丰富、探索充分 | off-policy 噪声大、训练易不稳 |

CISPO（10.5）和 DPPO（10.9）各自从不同角度修补硬 clip 的缺陷，**SAPO 走第三条路**：彻底放弃硬 clip 的"门"，换成一条**连续可微的衰减曲线**——near on-policy 处保留完整梯度、随 ratio 偏离 1 平滑衰减、远端趋零。

#### 核心设计 ①：sigmoid 软门函数

把 PPO 的 $r_t \cdot \hat A_t$ surrogate 换成 $f(r_t) \cdot \hat A_t$ ：

```math
f_{i,t}(r) = \sigma\!\big(\tau_{i,t}(r - 1)\big) \cdot \frac{4}{\tau_{i,t}}, \qquad r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid q, y_{i,\lt t})}{\pi_{\theta_{old}}(y_{i,t} \mid q, y_{i,\lt t})}
```

**这个函数的三个关键性质**（理解了就理解了 SAPO 的精髓）：

1. **$r = 1$ 时梯度恰好等于标准 PG**：算一下 $f'(1) = 4\sigma'(0) = 4 \times 0.25 = 1$ —— on-policy 点处**不打折**，软门完全打开
2. **远离 $r=1$ 时梯度平滑衰减**：sigmoid 在 $\pm\infty$ 饱和， $f'(r) \to 0$ —— 自动抑制极端 off-policy 样本，但**永远不归零**
3. **温度 $\tau$ 控制衰减速度**： $\tau$ 越大 → sigmoid 进入饱和区越快 → 梯度衰减越激进 → 信任区域越窄；反之 $\tau$ 越小信任区域越宽

整体目标函数（注意这里 $f$ 直接吃 $r$ ，**不再走 $\min/\text{clip}$**）：

```math
\mathcal{J}_{\text{SAPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} f_{i,t}\big(r_{i,t}(\theta)\big) \hat A_{i,t}\right]
```

其中 $\hat A_{i,t} = \hat A_i = (R_i - \text{mean})/\text{std}$ 仍是 GRPO 组内归一化的 advantage。

> **🎯 直觉**：把 PPO 的硬"门"换成"减速带"。on-policy 附近自由学习（系数=1），距离 1 越远学习强度越弱、**但始终非零**——这同时解决了 PPO clip 的两个毛病：
> - "case ① 消音"（CISPO 关心的那个）—— SAPO 这里梯度只是变小，没归零
> - "超 band 就完全失声" —— SAPO 永远在更新，只是步长更小

#### 核心设计 ②：非对称温度（正负优势用不同 $\tau$ ）

SAPO 进一步发现：**正优势和负优势 token 的稳定性特性截然不同**——

| | 正优势 token（"答对的"）| 负优势 token（"答错的"）|
|---|---|---|
| 学习动作 | 抬高 $\pi(y_{i,t})$ | 压低 $\pi(y_{i,t})$ |
| 对其他 token 的副作用 | softmax 归一化下其他 token 概率小幅压低，**局部、可控** | softmax 归一化下**词表上所有其他 token 概率被附带抬升** |
| 稳定性 | ✅ 稳定 | ❌ 容易把不相关 inappropriate token 拉飞 |

> **🗳️ 直观类比：班级投票**
>
> 把"调整 token 概率"想象成班里给候选人投票，词表有 5 万个候选人，softmax 保证所有候选人得票率加起来是 100%。
>
> **正优势 = "给候选人 A 投赞成票"**：A 得票率 +5%，剩下的 -5% 平摊到其他 49,999 人身上——**每个人只掉一丁点**，没人受到大影响 → 目标明确、副作用可控
>
> **负优势 = "给候选人 A 投反对票"**：A 得票率 -5%，**但这 5% 必须分给剩下的 49,999 人**——而且 softmax 是按"原本得票比例"分的，**原本就高票的候选人收益最大**。结果：你以为只在惩罚 A，**实际上把所有其他候选人（包括很多比 A 还糟的）一起抬上去了**。
>
> **🤬 LLM 场景的具体例子**：你想训模型"不要骂人"，发现脏话 token `"fuck"` 的 advantage 是负的，模型该压低它。
> - 数学上：压低 `P(fuck)` 必然意味着抬高 `P(剩下所有 token)`
> - 副作用：`P("shit")`、`P("damn")`、`P("bitch")` 等**同类脏话也跟着被抬上去了**
> - 你以为在惩罚一个坏 token，**实际上是把整批坏 token 一起加权**
>
> **🤨 等等，抬高一个好 token，不也会把其他"原来好的" token 一起压低吗？怎么就只有负优势带毒？**
>
> 好问题。表面上看，softmax 对"抬高"和"压低"是对称的——总有别的 token 要为这个变化"买单"。但**真正的不对称来自两个事实**：
>
> **① 任何具体上下文里：好 token 是少数，坏 token 是绝大多数**
>
> 比如续写"The pizza was ___"，5 万词表里：
> - "好"的候选（合适的描述词）可能只有几十个：delicious / good / amazing / terrible / ...
> - 剩下 49,000+ 个 token 在这个上下文里都是**不合适的**：标点、代码符号、不相关名词、脏话、乱码 token、...
>
> 所以：
> - **抬高一个好 token（如 "delicious"）**：其他 49,999 个被压低。其中**绝大部分本来就该被压低**（49,000+ 个不合适的 token）→ **副作用反而是免费的惩罚坏 token，净收益正**
> - **压低一个坏 token（如 "shit"）**：其他 49,999 个被抬高。其中**绝大部分仍然是不该被抬高的坏 token**（49,000+ 个不合适的）→ **副作用是免费抬高一堆坏 token，净收益经常为负**
>
> **② 即使副作用"压到了好 token"，性质也不严重**
>
> 假设抬高 "delicious" 时把 "good" 和 "amazing" 也微调下去了：
> - 这只是让模型对一个**具体**的好答案更自信，**而不是变得不好**
> - "更确定地说对的话"是 RL 想要的结果
>
> 反过来，压低 "shit" 时把 "damn"、"fucking"、"crap" 一起抬上去：
> - 你**完全没在朝目标走**——本意是"别骂人"，结果"骂人的方式变多了"
> - 这是 reward hacking 的温床
>
> 🍕 **一句话**：
> - 好 token 稀有 → 抬一个好的，副作用是**压一大片坏的**，赚的
> - 坏 token 多 → 压一个坏的，副作用是**抬一大片别的坏的**，亏的
>
> **🎯 所以 SAPO 让 $\tau_{\text{neg}} > \tau_{\text{pos}}$**：
> - 正优势信号是"净干净的"——副作用大多打在坏 token 上 → 可以放心学（ $\tau$ 小、衰减慢、学得多）
> - 负优势信号是"带毒的"——副作用大多打在其他坏 token 上 → 要更克制（ $\tau$ 大、衰减快、学得少）
> - 用更大的 $\tau$ 让负优势 token 更快进入 sigmoid 饱和区 → 梯度系数更早衰减 → 减少"连带抬升坏 token"的副作用

所以**负优势应该被更激进地抑制**——给它更大的 $\tau$ ：

```math
\tau_{i,t} = \begin{cases} \tau_{\text{pos}} & \text{if } \hat A_{i,t} > 0 \\ \tau_{\text{neg}} & \text{otherwise} \end{cases}, \quad \boxed{\tau_{\text{neg}} > \tau_{\text{pos}}}
```

**论文实测三组配置**（直接告诉你怎么调）：

| 配置 | 训练稳定性 |
|---|---|
| $\tau_{\text{neg}} = 1.05 > \tau_{\text{pos}} = 1.0$ | ✅ **最稳定**（推荐） |
| $\tau_{\text{neg}} = \tau_{\text{pos}} = 1.0$ | 中等 |
| $\tau_{\text{neg}} = 0.95 < \tau_{\text{pos}} = 1.0$ | ❌ **最不稳定**（负优势衰减过慢，把其他 token 拉飞了）|

> **🆚 和 DAPO Clip-Higher 的对偶关系**：
> - **DAPO**： $\epsilon_{\text{high}} > \epsilon_{\text{low}}$ —— **放宽正向探索的上界**
> - **SAPO**： $\tau_{\text{neg}} > \tau_{\text{pos}}$ —— **收紧负向惩罚的衰减速度**
>
> 两者精神一致——**承认正负梯度的不对称性**——只是 DAPO 在 ε 上做手脚（放正向），SAPO 在温度上做手脚（压负向）。

#### 优势与结果

- **优于 GRPO 和 GSPO**：在 30B 模型的数学任务上稳定优于两者
- **声称"不需要 Routing Replay"**：Qwen3-VL 系列（dense 和 MoE 变体）都用 SAPO 直接训，不挂 R3 也能跑完——但**这句话要打折扣**，见下方"治标 vs 治本"的讨论
- **跨任务跨规模一致**：reasoning、coding、多模态多任务上 reward 提升一致，泛化性好

> **🤔 SAPO 在 MoE 上真的"治本"了吗？—— 不，它和 GSPO 一样还是治标，R3 才是治本**
>
> 这点很容易被论文的"no need for routing replay"误导。诚实地拆一下：
>
> **MoE 训练不稳的根因**：一次梯度更新后 ~10% 的 token 换了 expert → 同一个 token 经过不同 expert，输出分布**截然不同** → $r_t$ 在这些位置剧烈跳变（可能是 0.01 或 100）。
>
> 三种应对策略本质的对比：
>
> | 方法 | 怎么处理"routing-flipped"的 token | 性质 |
> |---|---|---|
> | **R3（Routing Replay）** | **强制 training router 复用 inference 时的 routing mask** → 同一个 token 永远走同一个 expert → $r_t$ 根本不会因为路由翻转而极端 → **从根上消除问题** | **治本** |
> | **GSPO** | 把 token-level 极端 $r_t$ 用**几何平均**稀释进整条序列 → 极端值还在，只是被平均化看不见 | 治标（粒度提升 + 平均化） |
> | **SAPO** | $r_t$ 一旦极端就让 sigmoid 饱和、 $f'(r) \to 0$ → **直接把这些 token 的梯度掐掉** | 治标（软门压制 outlier） |
>
> **SAPO 实际在做的是**："凡是因为 routing 翻转导致 $r_t$ 不靠谱的 token，我就当你不存在，从梯度里删掉。"——表面上稳了，但**这些 token 携带的真实学习信号也一起被丢了**。R3 不一样：它让这些 token 保留有效的 $r_t$ ，**信号不丢**，只是 router 自身的探索空间被限制。
>
> **实验佐证：连 GSPO 都不能完全替代 R3**（R3 论文 arXiv:2510.11370）：
> - GSPO 单用 ✅ 训得完
> - **GSPO + R3 ≈ GSPO + 1.29 分** ← 这说明 GSPO 没把 routing 问题真正解决
> - GSPO + R3 ≈ GSPO + 0.95 分
>
> 按这个逻辑外推，**SAPO + R3 大概率也比单 SAPO 好**——只是 Qwen 团队的论文里没做这个消融对照，留了个"不需要 R3"的暧昧叙述。
>
> **正确读法**："不需要 R3" = "能跑稳、不崩"，**不等于** "和用 R3 一样最优"。如果你训 MoE 追求极致性能，SAPO + R3 的组合理论上仍优于单 SAPO；如果只追求工程简单+稳定，单 SAPO 已经够用。

#### 局限与争议

- **MoE 上是治标**：见上方讨论，软门会把 routing-flipped token 的梯度直接抑制掉，丢失了这些 token 的学习信号——R3 才是真正消除根因的做法
- **Sigmoid 饱和问题**：极端 ratio 处 $f'(r) \to 0$ ，等效于"软 clip" —— **CISPO 关心的"低概率分叉 token 在多步更新中被消音"在 SAPO 上仍可能发生**。后续 PSPO/GR-PSPO ([arXiv:2509.21282](https://arxiv.org/abs/2509.21282)) 改用线性插值想解决这个问题
- **多了一个超参**： $(\tau_{\text{pos}}, \tau_{\text{neg}})$ 取代了 $\epsilon$ ，调参负担没本质减小，只是空间变连续了
- **每个 token 多一个 sigmoid + exp**：计算开销可忽略

**🎯 一句话**：SAPO 用 **sigmoid 软门 + 非对称温度** 替代 PPO 硬 clip——既保留了"远离 on-policy 时削弱信号"的安全性，又消除了"硬截断→梯度归零"的副作用；负优势用更大 $\tau$ 抑制 softmax 归一化导致的"连带抬升不相关 token"，是 Qwen3-VL 系列的训练算法。

### 10.9 GDPO（NVlabs, 2026.01）—— 多 reward 场景下的"优势分辨率塌陷"修复

全称：**G**roup reward-**D**ecoupled Normalization **P**olicy **O**ptimization (arXiv:2601.05242, ICML 2026)。专门针对**多 reward 训练**的痛点。

#### 🚨 痛点：Reward Signal Collapse（多 reward 加和后分辨率塌陷）

现代 LLM RL 越来越普遍地用**多个 reward**——正确性 + 长度约束 + 格式合规 + 工具调用质量 + safety……GRPO 的标准做法是**先把多个 reward 加和成一个标量**，再做组内 normalization：

```math
r_{\text{sum}}^{(i,j)} = r_1^{(i,j)} + \cdots + r_n^{(i,j)}, \quad A_{\text{sum}}^{(i,j)} = \frac{r_{\text{sum}}^{(i,j)} - \text{mean}}{\text{std}}
```

这种"先加和、再归一化"的顺序看似无害，**实际会让大量不同的 reward 组合压缩成同一个 advantage**——叫做 **advantage resolution collapse**（优势分辨率塌陷），让训练信号丢失关键信息。

#### 🍎 一个具体的例子（论文核心 motivation）

假设：
- 2 个 0/1 reward（比如 correctness + format）
- 一次 RL step 采 2 次 rollout
- 每个 rollout 的 sum reward $`\in \{0, 1, 2\}`$ ，所以共有 $3 \times 3 = 9$ 种 (sum_1, sum_2) 组合

**GRPO 的处理**（先加和、再归一化，N-1 denominator）：

| (sum_1, sum_2) | mean | std | GRPO advantage |
|---|---|---|---|
| (0, 1) | 0.5 | $\sqrt{0.5}$ | (−0.7071, +0.7071) |
| (0, 2) | 1.0 | $\sqrt{2}$ | (−0.7071, +0.7071) |
| (1, 2) | 1.5 | $\sqrt{0.5}$ | (−0.7071, +0.7071) |
| (2, 1) | 1.5 | $\sqrt{0.5}$ | (+0.7071, −0.7071) |
| ... | ... | ... | 9 种组合 → 只剩 2 种符号 |

🚨 **9 种 reward 组合最终只对应 2 种 advantage 模式**（只看相对大小的符号）！

**直觉上的问题**：(0, 2) 应该比 (0, 1) **携带更强的学习信号**——前者是"两个 reward 都不满足 vs 都满足"，后者只是"一个 reward 满足 vs 都不满足"。GRPO 把它们等同看待，**模型分不清"两条都对"和"只对一条"的区别**。

**特殊场景下还会更糟**：某个 reward 的 scale 极大（如 0/100 而不是 0/1），它会**主导**加和后的 sum，其他 reward 的差异被掩盖——模型实际上只在学那个大 reward。

#### 💡 核心设计：先 per-reward 归一化，再加和，最后 batch-wise 归一化

GDPO 把操作顺序**反过来**：

**Step 1**：对每个 reward 维度**单独**做组内归一化：

```math
A_k^{(i,j)} = \frac{r_k^{(i,j)} - \text{mean}\{r_k^{(i, 1)}, \dots, r_k^{(i, G)}\}}{\text{std}\{r_k^{(i, 1)}, \dots, r_k^{(i, G)}\}}, \quad k = 1, \dots, n
```

**Step 2**：把归一化后的 per-reward advantage 加和：

```math
\tilde A_{\text{sum}}^{(i,j)} = A_1^{(i,j)} + \dots + A_n^{(i,j)}
```

**Step 3**：在 **batch 级再做一次归一化**（防止 advantage 数值范围随 reward 数量膨胀）：

```math
\hat A_{\text{sum}}^{(i,j)} = \frac{\tilde A_{\text{sum}}^{(i,j)} - \text{mean}_{\text{batch}}}{\text{std}_{\text{batch}} + \varepsilon}
```

> **为什么 Step 3 这一步必须有？** 加和后的 $\tilde A_{\text{sum}}$ **不再保证落在某个闭区间内**——每多一个 reward 都会让数值范围膨胀（ $n$ 个 reward 各贡献 ±O(1) → 总和最大 ±O($n$)）。如果不做 batch 归一化，调超参时学习率得跟着 reward 数量改。batch 级再归一化让**数值尺度与 reward 数量解耦**，超参更稳。

#### 📊 回到那个 9 种组合的例子

| (sum_1, sum_2) | GRPO | GDPO（Step 2 后）|
|---|---|---|
| (0, 1) | (−0.7071, +0.7071) | (−0.7071, +0.7071) |
| (0, 2) | (−0.7071, +0.7071) | **(−1.4142, +1.4142)** ← 信号强 2 倍 |

✅ GDPO 让 (0, 2) 的学习信号是 (0, 1) 的 2 倍——**符合"两个 reward 都改善"应该比"只一个 reward 改善"信号更强**的直觉。

> **🧮 为什么 (0, 2) 在 GDPO 下信号刚好是 2 倍？** 
> "(0, 2)" 在多 reward 视角下意味着：rollout 1 在 reward 1 上得 0、reward 2 上得 0；rollout 2 在 reward 1 上得 1、reward 2 上得 1。
> - 对 reward 1：组内 (0, 1)，per-reward advantage = (−0.7071, +0.7071)
> - 对 reward 2：组内 (0, 1)，per-reward advantage = (−0.7071, +0.7071)
> - 加和：(−1.4142, +1.4142) ← **两个维度信号同向叠加**
> 
> 而 (0, 1) 只在一个 reward 维度上有差异，另一维度没差异（advantage = 0），所以总和不增强。

#### 🆚 和 Dr. GRPO 去 std 归一化的关系

> Dr. GRPO 主张去掉 std 归一化（保留 mean 减），在 GDPO 的例子里：
> - (0, 1) → (−0.5, +0.5)
> - (0, 2) → (−1.0, +1.0)
>
> 也能区分两种组合的强度。但 Dr. GRPO **只解决了 std 归一化导致的"难度偏置"**，**没解决多 reward 加和导致的"个体 reward 信号被 dominated"**——大 scale 的 reward 仍会盖过小 scale 的。
>
> GDPO 是从**粒度上做解耦**：每个 reward 在自己的统计内归一化，scale 大小被各自的 std 抵消，**所有 reward 进入加和时数值范围一致**，信号公平叠加。两个问题（难度偏置 + reward dominance）一起治。

#### 📈 实验结果（NVlabs/GDPO）

- **Math 推理**（DeepSeek-R1-1.5B / Qwen3-4B-Instruct）：AIME 准确率比 GRPO 高 **2.3%–6.3%**，同时把超长回答比例降低 **80%**
- **代码任务**：通过率持平 GRPO，bug ratio 和 length-violation ratio 双降
- **Tool calling**：correctness 和 format adherence 两个 metric 同步提升

实现已上 HF-TRL / veRL / Nemo-RL，代码：[NVlabs/GDPO](https://github.com/NVlabs/GDPO)。

**🎯 一句话**：GDPO 把 GRPO 的"先加 reward、再归一化"反过来——**先逐 reward 归一化、再加和、最后 batch 归一化**。这个简单的顺序调换让多 reward 场景下的 advantage 信号分辨率从"只看符号"恢复到"区分每个 reward 维度的贡献"，是当前**多目标 RL** 场景的事实推荐做法。

### 10.10 DPPO（Sea AI Lab, 2026.02）—— 用"散度"替代"比值"做信任域

**痛点**：DPPO 提出了一个对 PPO clip **更根本的批评**——PPO 的"比值 + clip"机制**结构性地不适合 LLM 的大词表场景**：
- $r_t = \pi_\theta(a)/\pi_{\text{old}}(a)$ 是从**单点采样**估出来的策略散度——**低概率 token 的 $r_t$ 噪声极大,被过度惩罚**(case ① 那种)
- 而**高概率 token 的 ratio 看起来温和,但其分布上的变化可能是灾难性的,反而约束不足**

**核心思路**：抛弃 ratio,直接估计 $\pi_\theta$ 和 $\pi_{\text{old}}$ 之间的**真正散度**(TV 或 KL),用它构造信任域 mask:

```math
\text{mask}_t = \mathbb{1}\big[D(\pi_\theta(\cdot|s_t)\ \|\ \pi_{\text{old}}(\cdot|s_t)) < \delta\big]
```

但词表上的精确散度算不起,DPPO 用两种**廉价近似**:
- **Binary**: 只看"两个分布的 top-1 token 是否同一个"——二元指示
- **Top-K**: 看 top-K token 上的散度

> 🎯 **直觉**:不再问"这个采样 token 的概率涨了多少倍"(噪声),而是问"**两个分布整体上偏移了多少**"(更稳)。

**Loss 形式**(以 Binary-TV 为例):

```math
L^{\text{DPPO}} = -\mathbb{E}\big[\text{mask}_t \cdot r_t \cdot \hat A_t\big]
```

mask 在散度超出信任域时直接置 0,**不再有 case ① 的不对称问题**。

**veRL 实现**(直接可跑):
```bash
LOSS_MODE=dppo_kl  bash examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh  # KL 版
LOSS_MODE=dppo_tv  bash examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh  # TV 版
```

**实验结果与关键洞察**：在 Qwen3-30B-A3B(MoE)上,**即使不用 R3,DPPO 也比 GRPO baseline 显著好**。这是一个**对照实验式的发现**:

| 配置 | Routing 问题 | Clip 机制 | 结果 |
|---|---|---|---|
| GRPO baseline | ❌ 不解决 | ratio + clip | 训崩或差 |
| GRPO + R3 | ✅ 解决 | ratio + clip | 训稳 |
| **DPPO**(不用 R3) | ❌ **同样不解决** | **换成 divergence mask** | ✅ **训稳,超 GRPO** |

> **逻辑推断**：若 MoE 崩**只**是 routing 的锅,DPPO 不动 routing 就应该照崩。但它居然稳——**唯一变量是 clip 机制**,所以 **clip 机制本身就是 MoE 不稳的另一个独立成因**。

**为什么 PPO clip 在 MoE 上特别糟？** 两个原因叠加:

1. **MoE 的输出分布更尖锐**：专家分工导致每个 expert 负责一部分 token 分布,模型整体输出**更多低概率"边缘 token"**(某个 expert 强烈偏好但其他 expert 给低分的 token)
2. **PPO ratio 在低概率 token 上噪声爆炸**: $r_t$ 是单点蒙特卡洛估计,概率越低方差越大,叠加 PPO clip 的 case ① 不对称惩罚 → **低概率 token 被持续过度惩罚** → 信号失真

→ MoE 双重放大了 PPO clip 的固有缺陷,routing 漂移只是雪上加霜。DPPO 用**分布级散度**(而非单点 ratio)做信任域判定,**直接消除了对单点采样的依赖**,所以在 MoE 上不用 R3 也能稳。

**研究意义**：之前所有 MoE RL 工作(R3、GSPO、RSPO 等)都在打 routing 这个补丁,**问题的另一半——clip 机制本身的缺陷——长期被忽略**。DPPO 第一次清晰隔离出这两个不稳定源。

**🎯 一句话**:DPPO 是 2026 年初对 PPO clip "ratio + 单点采样"这一**最底层假设**的挑战——**用分布级散度替代单点 ratio**,从根上解决"低概率 token 过惩罚、高概率 token 欠约束"的 PPO 设计缺陷。

### 10.11 GiGPO（NTU, 2025.05, NeurIPS 2025）—— 把 GRPO 搬到 Agent 长链任务

全称：**G**roup-**i**n-**G**roup **P**olicy **O**ptimization (arXiv:2505.10978)。这是前面所有变体里**第一个明确针对 agentic / multi-turn / 长链任务**的工作——和本文档主题（Agentic RL）最贴近。

#### 🚨 痛点：GRPO 在长链 agent 任务上"信用分配失灵"

之前所有变体（DAPO、GSPO、CISPO、SAPO……）默认的场景都是**单 turn 任务**：一个 prompt → 一段完整答案 → 一个 reward。Agent 任务完全不一样：

| | 数学题（GRPO 原生场景）| Agent 任务（GiGPO 目标场景）|
|---|---|---|
| 步数 | 1（一次性生成完整 CoT） | 几十步（ALFWorld 可达 **50 步**）|
| Token 数 | 几千 | **2 万+** |
| Reward 时机 | 立即（生成完就给分）| 极度稀疏，**常常只在整个 episode 结束才给一次** |
| Trajectory 内的步骤 | 同质（都是同一回答的 token）| 异质（每步是独立决策）|

**GRPO 的 advantage 直接套上来会怎样？**

$\hat A_i = (R_i - \text{mean})/\text{std}$ —— **整条 trajectory 内所有 step 共享同一个 advantage**。这在数学题上没问题（CoT 整体一起对、一起错），但在 agent 上是灾难：

> 一条 trajectory 走了 50 步、最后成功 → 这 **50 步全部**被打上 advantage = +1。但其中可能 30 步是无意义乱走、20 步是关键决策——**模型完全无法分辨**，好策略和坏策略一起被强化。

**为什么不能像 PPO 那样训个 critic 给 step-level credit？** VAPO 那节我们已经看到，critic 在长 CoT 上偏差会累积、value 学不稳——更何况 agent 场景动辄 50 步、20k token，critic 根本训不起来。

**GiGPO 的目标**：保留 GRPO "critic-free / 低显存 / 稳定" 的优势，**同时**给出 step-level 信号。

#### 💡 核心设计：嵌套两层"组"——Group-in-Group

GiGPO 的关键 insight 是**把 GRPO 的"组"概念用两次**，一层粗、一层细：

**第一层（macro / episode 级）**：和原 GRPO 完全一样
- 对每个任务采样 $G$ 条完整 trajectory
- 用每条 trajectory 的总 return 做组内归一化 → **episode-level advantage** $\hat A_i^{\text{macro}}$
- 反映"这条 trajectory 整体好不好"

**第二层（micro / step 级）**：⭐ 新机制——**Anchor State Grouping**
- 扫描所有 $G$ 条 trajectory 的**所有步骤**
- 找出**在多条 trajectory 中重复出现**的环境状态——叫做 **anchor states**
- 在每个 anchor state 上，**把不同 trajectory 在这个状态下采取的不同 action 聚成一个 micro 组**
- 在 micro 组内做归一化 → **step-level advantage** $\hat A_t^{\text{micro}}$
- 反映"**在这个状态下，你选的 action 比其他 trajectory 在同一状态下选的好不好**"

最终每个 step 的 advantage：

```math
\hat A_t = \hat A_i^{\text{macro}} + \alpha \cdot \hat A_t^{\text{micro}}
```

（ $\alpha$ 是 step-level 权重，论文里通常取 1.0）

> **🎯 直觉**：macro 看"这条路整体好不好"，micro 看"在十字路口你选的方向对不对"——两者叠加，模型既学到 trajectory 级偏好，又学到 step 级选择。

#### 🏠 用 ALFWorld 例子讲清楚

任务："把苹果放进冰箱"。采了 3 条 trajectory：

```
T1: [开冰箱] → [拿苹果] → [放苹果到冰箱] → ✅ 成功 (R=1)
T2: [拿苹果] → [开冰箱] → [放苹果到冰箱] → ✅ 成功 (R=1)  
T3: [拿苹果] → [放苹果到冰箱] → ❌ 失败 (R=0, 冰箱关着没法放)
```

**纯 GRPO 怎么算？**

三条 trajectory 整体打分：mean = 0.67, std ≈ 0.47
- $A_{T1} = A_{T2} = +0.71$ （所有 step 共享）
- $A_{T3} = -1.41$ （所有 step 共享）

但你看 T2 和 T3 里都有 **"拿苹果"** 这一步，而且当时**冰箱都是关着的**——两步**完全一样**！为什么 T2 这一步被强化、T3 这一步被惩罚？**GRPO 无法回答。**

**GiGPO 怎么算？**

**Step 1**：识别 anchor states。观察发现 `"手里拿着苹果，冰箱关着"` 这个状态在 T2 和 T3 都出现过：
- T2 在该状态选了 `[开冰箱]` → 后续成功
- T3 在该状态选了 `[放苹果到冰箱]` → 失败

**Step 2**：在这个 anchor state 上算 micro advantage（micro 组只有 T2 和 T3 的这一步）：
- T2 的 `[开冰箱]`： $A^{\text{micro}} = +0.7$ （这个选择回报更高）
- T3 的 `[放苹果到冰箱]`： $A^{\text{micro}} = -0.7$

**Step 3**：合成最终 advantage：

| Step | Macro | Micro | 最终 advantage |
|---|---|---|---|
| T2 的 `[开冰箱]`（在 anchor 上）| +0.71 | +0.7 | **+1.41** ← 关键决策被加倍强化 |
| T3 的 `[放苹果到冰箱]`（在 anchor 上）| −1.41 | −0.7 | **−2.11** ← 错误决策被加倍惩罚 |
| T1 的 `[开冰箱]`（没 anchor 匹配）| +0.71 | 0 | +0.71（仅 macro）|

**结果**：模型清楚地知道 **"手里有苹果但冰箱关着时，应该先开冰箱而不是直接放"** —— 这正是 step-level credit assignment 的核心价值。

#### 🪄 为什么 anchor state 这个 trick 这么巧妙？

1. **不需要额外 rollout**：anchor states 是从**已有的 G 条 trajectory 里挖出来的**，零样本浪费
2. **不需要 critic**：micro advantage 还是用组内归一化算的（GRPO 风格），没有 value model
3. **不需要 reward shaping**：奖励信号来源不变，只是被**重新分配到 step 上**
4. **几乎零额外成本**：论文报告 < 0.002% time cost、显存零增加

> **🆚 和 VAPO 路线的对比**：VAPO 用 critic 解决长 CoT 的 credit assignment——技术上能 work 但工程极复杂（要 pretrain critic、要 decoupled-GAE、要 length-adaptive 等 5 件套）。GiGPO 走另一条路：**复用 GRPO 的组比较思想，但分层做**——简洁很多。两条路线代表了 long-horizon RL 的两种哲学：补 critic vs 升级 group 结构。

#### 📈 实验结果

| Benchmark | vs GRPO baseline |
|---|---|
| **ALFWorld**（embodied 任务规划）| **+12%** |
| **WebShop**（goal-driven 网页交互）| **+9%** |
| 搜索增强 QA（Qwen2.5-3B）| 42.1% |
| 搜索增强 QA（Qwen2.5-7B）| 47.2% |

**显存 / rollout 成本和 GRPO 完全一致**——这才是关键卖点：免费的 step-level 信号。

#### ⚠️ 局限性

- **依赖 anchor state 能匹配上**：要求"环境状态在不同 trajectory 中能精确重复"。在**离散、结构化**环境（ALFWorld、WebShop）里好用；在**开放、连续**环境（如开放 web 浏览、自由文本对话）里，状态几乎不会精确重复，**anchor 匹配会大量失效**——这时 micro 那层基本变成零，退化回 GRPO
- **依赖采样多样性**：如果 $G$ 太小，trajectories 之间重叠少，找不到几个 anchor states； $G$ 太大 rollout 成本又高
- **后续工作**：Tree-GRPO、AT2PO、AgentPRM 等用树结构 rollout 或显式 process reward model 解决"state 不精确匹配"，但代价是引入额外 model / rollout

**🎯 一句话**：GiGPO 是**第一个把 GRPO 从单 turn 数学场景成功迁移到 multi-turn agent 场景**的代表性工作——用 "Group-in-Group" 嵌套组结构，在不引入 critic、不增加 rollout 的前提下，给 agent 长链任务实现了 step-level 信用分配。**对 Agentic RL 方向是个里程碑**，也是本文档主题（AgenticRL）最贴近的 GRPO 变体。

> 🔗 **GiGPO 只是"信用分配"这一个环节上的一种打法。** 2025 年中以来 agentic RL 已经铺开成五个环节（观测掩码、信用分配、熵/探索、reward 稀疏、异步 rollout）——GiGPO 之外还有 Tree-GRPO、MT-GRPO、SWEET-RL 等同类思路，以及掩码、异步、环境等一整套配套技术。完整展开见 **§13 Agentic RL 专题**，其中 §13.3 把这几种信用分配方法放在一张表里对比。

### 10.12 十一个变体放在一起看：选型与组合

| 算法 | 核心改动 | 解决什么 | 适用场景 |
|---|---|---|---|
| **DAPO** | Clip-Higher + 动态采样 + token-level loss + 软长度惩罚 | GRPO 的多个工程毛病 | 通用 LLM RL，事实工程标准 |
| **Dr. GRPO** | 去掉 per-sample 长度归一化 + std 归一化 | 长度偏置、难度偏置 | 想看清并修正 GRPO 公式问题 |
| **REINFORCE++** | 全局归一化代替组内归一化 | 小数据集过拟合、组间尺度不齐 | 通用 RLHF，训练数据少时 |
| **VAPO** | 把 critic 救回来 + 5 个稳定化技巧 | critic-free 方法在长 CoT 上的天花板 | 长推理（AIME 等），愿意付额外计算 |
| **CISPO** | clip 权重而非目标，所有 token 都贡献梯度 | clip 屏蔽长 CoT 关键"分叉 token" | 长推理 + 多步更新（如 16 步） |
| **GSPO** | sequence-level ratio 代替 token-level | ① IS 粒度 ≠ reward 粒度的错位 ② MoE 训练不稳（两个并列核心贡献）| 长 CoT / MoE 模型（Qwen3、Mixtral） |
| **GFPO** | 拒绝采样筛"简洁子集"，只在子集内做 GRPO | 长度膨胀（4k→14k）+ reward hacking | 想压 inference 长度，又不想动 reward |
| **SAPO** | sigmoid 软门 + 非对称温度替代硬 clip | PPO clip "二值开关"的两难调参 | 多模态/多任务（Qwen3-VL）、想避免硬 clip 调参 |
| **GDPO** | 先逐 reward 归一化、再加和、再 batch 归一化 | 多 reward 加和后 advantage 分辨率塌陷 | 多目标 RL（正确性 + 长度 + 格式 + safety 等） |
| **DPPO** | 散度 mask 代替 ratio clip | PPO clip 在大词表下的结构缺陷 | MoE，且不想用 R3；想从根上替换 PPO clip |
| **GiGPO** | 嵌套两层组（episode + step via anchor state）| GRPO 在长链 agent 任务上"全 trajectory 共享一个 advantage"导致信用分配失灵 | **Agentic RL / 多 turn / 长链任务**（ALFWorld、WebShop、tool calling 等）|

**演化逻辑**：
```
GRPO（基础，原生面向单 turn 任务）
  ├─ DAPO：clip/采样/loss 四件套工程优化（事实标准）
  ├─ Dr. GRPO：去掉两个隐藏偏置（理论纠偏）
  ├─ REINFORCE++：放弃组内归一化，全局更稳
  ├─ VAPO：critic 回来了（反潮流）
  ├─ CISPO：换 clip 位置，救低概率 token
  ├─ GSPO：IS 升到 sequence 级——既对齐 reward 粒度，又根治 MoE 训练不稳
  ├─ GFPO：多采样筛简洁子集，治长度膨胀
  ├─ SAPO：sigmoid 软门 + 非对称温度，把硬 clip 换成连续衰减
  ├─ GDPO：调换 reward 归一化和加和的顺序，治多 reward 分辨率塌陷
  ├─ DPPO：用散度 mask 替代 ratio clip，挑战 PPO 最底层假设
  └─ GiGPO：嵌套两层组，把 GRPO 迁移到 multi-turn agent 长链任务 ★ Agentic RL 分支
```

> **🧭 "硬 clip 改造"三件套（CISPO / SAPO / DPPO）**：这三个变体都在攻击 PPO clip 的设计缺陷，但角度不同：
> - **CISPO**：把 clip 从 surrogate 移到权重，**保留 logp 显式可导**（救多步更新里的低概率 token）
> - **SAPO**：把硬 clip 整体换成 **sigmoid 软门**（消除二值开关，让信任域连续化）
> - **DPPO**：放弃 ratio 改用**分布散度 mask**（攻击单点 ratio 估计本身的噪声）
>
> 三者各有侧重：CISPO 保 token、SAPO 保连续性、DPPO 换估计单位。

> **📐 "归一化顺序"两件套（Dr. GRPO / GDPO）**：这两个变体都在拷问"GRPO 那条标准归一化流程到底对不对"：
> - **Dr. GRPO**：去掉 std 归一化（治"难度偏置"）+ 去掉 per-sample 长度归一化（治"长度偏置"）
> - **GDPO**：把"先加 reward、再归一化"反过来变成"先归一化、再加 reward、再 batch 归一化"（治"多 reward 分辨率塌陷"）
>
> 一个攻 GRPO 的"过度归一化"，一个攻 GRPO 的"过早加和"。

> **🤖 "长链信用分配"两条路线（VAPO / GiGPO）**：reasoning/agent 任务都面临"长 trajectory + 稀疏 reward + 需要 step-level credit"的挑战，两条思路：
> - **VAPO**：把 critic 救回来——技术上能给 step-level credit，但要付 5 件套稳定化的工程成本
> - **GiGPO**：升级 group 结构而不引 critic——用 anchor state 在组内做 step 级比较，几乎零成本
>
> 路线 A 适合 long CoT reasoning（critic 还训得动），路线 B 适合真正的 agent 多 turn 任务（critic 训不动）。

**实践建议**：
- **从头训通用 LLM RL** → **DAPO** 起步（工程最成熟）
- **训练数据少** → 加入 **REINFORCE++** 的全局归一化
- **MoE 模型** → **GSPO** / **DPPO** 都可，或 GRPO + R3
- **极长 CoT + 多步更新** → 考虑 **CISPO**
- **回答越训越长，inference 慢** → 叠加 **GFPO**（和上面几条都正交）
- **多模态/多任务/想脱离硬 clip 调参** → **SAPO**（Qwen3-VL 路线）
- **多个 reward 同时训（正确性 + 长度 + 格式 + ...）** → 一定用 **GDPO** 的归一化顺序，别用 GRPO 直接加和
- **训 multi-turn agent（ALFWorld / WebShop / tool calling / web 操控等）** → **GiGPO** 是当前最简洁的 step-level credit 方案
- **追求最高性能、可付出复杂度** → **VAPO**

> 不同工作针对不同瓶颈，**实际工程里常常组合使用**：例如 **GSPO + Clip-Higher（来自 DAPO）+ 全局归一化（来自 REINFORCE++）** 是 Qwen3 配方的近似形式。

---

## 11. 当前最常用的事实标准配方

回答开头那个问题——**现在主流是不是"GRPO base + 一堆 trick"？** 是的。2025 年下半年开始，大多数开源 / 工业团队的 RL 训练栈基本是这个形态。下面给一份**通用 Dense 模型**和**MoE 模型**的标配，可以直接对着抄。

### 11.1 Dense 模型标配（事实标准 = GRPO + DAPO + 部分 Dr. GRPO）

```
基底：    GRPO（critic-free，组内相对 advantage）
clip：    Clip-Higher（来自 DAPO）—— ε_low=0.2, ε_high=0.28
采样：    Dynamic Sampling（来自 DAPO）—— 过滤全对/全错的组
loss 归一化：Token-level aggregation（来自 DAPO）—— 按 batch 总 token 数
advantage：组内 (R - mean) / std（原版 GRPO，主流实现保留此设置）
长度：    Overlong Soft Penalty（来自 DAPO）—— 缓冲区线性惩罚
KL：      per-token KL，k3 估计，β ≈ 0.001 ~ 0.01（很小或 0）
```

**Loss 公式**（把以上 trick 串起来）：

```math
L = -\frac{1}{\sum_i |o_i|} \sum_i \sum_t \min\!\big(r_t^i \hat A_i,\ \text{clip}(r_t^i, 1{-}\epsilon_{\text{low}}, 1{+}\epsilon_{\text{high}}) \hat A_i\big) + \beta \cdot \overline{\text{KL}}_{\text{k3}}
```

其中 $\hat A_i = (R_i - \text{mean}(R_{1..G})) / \text{std}(R_{1..G})$ ， $r_t^i = \exp(\log\pi_\theta - \log\pi_{\theta_{\text{old}}})$ 。

> **🤔 关于 advantage 归一化 —— 现在仍是开放问题**
>
> Dr. GRPO 论文指出 std 归一化会引入"难度偏置"（方差小的组被放大权重），主张去掉。**但这只是修正主张，不是事实标准**：
>
> | 实际状况 | 是否保留 std 归一化 |
> |---|---|
> | DeepSeek-R1 / R1-Zero | ✅ 保留 |
> | 原版 DAPO（ByteDance） | ✅ 保留 |
> | OpenRLHF / TRL / VeRL 默认 | ✅ 保留 |
> | Dr. GRPO（Sea AI Lab） | ❌ 主张去掉 |
> | 部分新论文（SimpleRL 等） | ❌ 跟进去掉 |
>
> **保守起见**：上面的 Loss 公式按"主流实现"写（保留 std）。**如果你想试 Dr. GRPO 的修正**，把 $\hat A_i$ 换成 $R_i - \text{mean}(R_{1..G})$ 即可——是个一行改动，可以做消融实验对比。
>
> 多数论文报告**两种归一化在通用任务上差异有限**，Dr. GRPO 的优势主要体现在**组间难度差异大**的混合数据集上。

**典型超参（参考 DAPO 论文 + 主流开源实现）**：

| 超参 | 推荐值 | 说明 |
|---|---|---|
| Group size $G$ | 8 ~ 16 | 每个 prompt 采样数 |
| $\epsilon_{\text{low}}$ | 0.2 | clip 下界，标准 PPO 值 |
| $\epsilon_{\text{high}}$ | 0.28 | clip 上界（DAPO 推荐 0.28，可调到 0.3）|
| $\beta$ (KL) | 0.001 ~ 0.01 | 验证类任务可设 0；通用对话保留 |
| 学习率 | 1e-6 ~ 1e-5 | 比 SFT 小 10~100 倍 |
| Rollout per update | 1 | on-policy，rollout 完就 update 一次 |
| Update epochs / batch | 1 ~ 4 | 太多步会让 clip 频繁触发 |
| 最大长度 $L_{\max}$ | 8K ~ 32K | 长 CoT 任务 |
| Length cache $L_{\text{cache}}$ | $L_{\max} - 4K$ | DAPO 软惩罚起点 |

**可选叠加**：
- **小数据集**（< 1k prompts）→ 加入 **REINFORCE++ 全局归一化**（advantage 用 batch 级 mean/std）
- **多步更新**（epochs > 4）→ 考虑换成 **CISPO**（防止 case ① 消音）

### 11.2 MoE 模型标配（**两条主流路线并存：R3 / GSPO，可单用可组合**）

MoE 模型的 RL 训练根源问题是：**专家路由(routing)在每次梯度更新后会变化**(Qwen3-30B 上约 10% 的 token 选了不同 expert)，导致 token-level importance ratio 剧烈波动，GRPO 三次实验三次崩。**目前业界并没有"唯一正确"的解法，而是两条互补的路线并存**：

#### 路线 A：**R3 (Rollout Routing Replay)** —— 强制对齐 routing

**思路**：rollout 阶段用 inference engine（SGLang/vLLM）生成响应时，**把每个 token 的"专家选择 mask"缓存下来**；训练阶段强制 training engine 用同一份 mask，绕过 training router 的 top-k 选择。

```
inference 阶段：tokens → router_inference → 选 expert → 同时缓存 mask
training 阶段：  tokens + 缓存的 mask → 直接用 mask 替代 router 选择 → 计算 logp
```

- **效果**：3/3 GRPO 基线 collapse 的场景，3/3 R3 run 全跑完
- **代价**：需要缓存 routing mask，增加少量内存开销；和 prefix caching 可以共存
- **状态**：**veRL 已通过 SGLang 实现 merge**（PR #4840），是目前最稳的工程化方案
- **保留 GRPO 框架**：上层 loss 还是标准 GRPO/DAPO 那一套，**只是底层 routing 被锁定**

> **🤔 那 R3 下，training router 还能学吗？—— 能，但学习空间被限制**
>
> Forward 修改只换一行：
> ```python
> logits = router(x)              # ✅ training router 仍正常前向
> gating = softmax(logits * forced_mask, dim=-1)  # ⭐ 在"被强制选中"的 expert 内归一化
> output = sum(gating[i] * expert_i(x) for i in forced_mask)
> ```
> **`logits` 进了 gating → 梯度仍传到 router**。但代价是：
>
> | 组件 | 梯度 |
> |---|---|
> | Router 对**被选中** expert 的 logit | ✅ 有（影响 gating 权重 → 影响 output）|
> | Router 对**未被选中** expert 的 logit | ❌ 零（被 mask 掉，不参与 output）|
>
> 所以 router **不是 detach，而是"被限制了选择权"**：
> - ❌ 不能自主决定选谁
> - ✅ 但能优化"被强制选中的几个 expert 的相对权重"
>
> 🎓 **类比**：考试时老师指定每道题用哪种方法（forced mask），学生不能自选——他**仍然在学**（学会"用指定方法做这道题最好"），但**没法练习"在多种方法里挑哪个最优"**。
>
> ⚠️ **这正是 GSPO 论文批评 R3 的核心论点**："restricts the model's capacity"——router 没法自由探索新的 expert 组合，长期可能限制模型容量。**这是 R3 的真实代价**，换来训练稳定性。

#### 路线 B：**GSPO** —— sequence-level ratio 绕开问题

**思路**：既然 token-level ratio 因路由不稳而失真，那就升到 sequence-level：

```math
s_i = \exp\!\big(\tfrac{1}{|o_i|}\sum_t (\log\pi_\theta - \log\pi_{\text{old}})\big),\qquad L^{\text{GSPO}} = -\mathbb{E}[\min(s_i \hat A_i,\ \text{clip}(s_i, 1{-}\epsilon_{\text{low}}, 1{+}\epsilon_{\text{high}}) \hat A_i)]
```

- **效果**：训得稳，是 Qwen3-235B-A22B 的算法
- **代价**：信号粒度变粗，梯度信用归到 sequence 级
- **clip 范围**：⚠️ **常见踩坑**：sequence-level ratio 是几何平均，波动范围天然小两个数量级， $\epsilon$ 必须**缩小**——Qwen3 实际用 $\epsilon_{\text{low}}=3\times 10^{-4}, \epsilon_{\text{high}}=4\times 10^{-4}$ ，**不是** 0.2！沿用 token-level 的 0.2 会立即训崩

#### 🤔 R3 vs GSPO：哪个更好？**最新结论是可以叠加**

根据 R3 论文(arXiv:2510.11370)的实验：

| 配置 | 多 mini-step 任务表现 |
|---|---|
| GRPO baseline | 3/3 collapse |
| GSPO | 跑完，作为对照基线 |
| **GRPO + R3** | **比 GSPO 高 1.29 分** |
| **GSPO + R3** | **比单 GSPO 再高 0.95 分** |

也就是说：**R3 是个正交的稳定化技术，可以叠加在 GSPO 之上**。

#### 📋 实操配方建议（veRL 视角）

**保守稳妥（生产环境）** —— 复用现有 GRPO/DAPO pipeline：
```
基底：    GRPO / DAPO（保留 token-level ratio）
加：      R3 routing replay（veRL --rollout_routing_replay=true）
其他：    继承 Dense 配方
```

**追求最优** —— 沿 Qwen3 配方：
```
基底：    GSPO（sequence-level ratio，clip ε 缩到 ~3e-4）
加：      R3（可选叠加，再涨 ~1 分）
其他：    继承 Dense 配方（DAPO 的动态采样、token-level loss aggregation 等）
```

**veRL 另一条路 —— DPPO**(基于 trust-region divergence mask):在 Qwen3-30B-A3B 上无需 R3 也比 GRPO baseline 显著好；可以通过 `LOSS_MODE=dppo_kl` 启用。

> **关键认知**：**MoE RL 还没收敛到"唯一最优解"**。R3 是工程上最稳的"打补丁"做法，GSPO 是算法上最优雅的"换框架"做法，DPPO/RSPO 等新方法还在涌现。**Qwen3 团队自己也在 GSPO 上叠 R3 类技术**，不存在"用了 GSPO 就不需要 routing replay"这种简单结论。

### 11.3 长 CoT + 多步更新（专门救"分叉 token"）

```
基底：    DAPO 或 GSPO
loss：    换成 CISPO 形式（clip 权重 + 显式 logp）
         L = -stop_grad(min(r_t, ε_high)) · A · log π_θ
ε_high：  0.2 ~ 0.4（CISPO 推荐 0.3）
更新步数：可以放心拉到 16+
```

适用于 **MiniMax-M1 这类多步 update + 长 CoT** 的极端场景；通用任务上 CISPO 优势不明显，反而 DAPO/GRPO 更简单稳定。

### 11.4 一句话选型

```
                ┌────────────────────────────────────┐
                │ MoE 模型？ ─Yes→ GSPO base         │
                │    │No                             │
                │    ▼                               │
                │ DAPO base（GRPO + 四件套）         │
                │    │                               │
                │    ├─ 数据少？ → +REINFORCE++ 归一化 │
                │    ├─ 多步 update？ → CISPO 替换 loss│
                │    ├─ 长 CoT 且预算够？ → 上 VAPO    │
                │    └─ 通用任务 → 已经够用            │
                │                                    │
                │ 通用配置：                          │
                │   - 去 std 归一化（Dr. GRPO）       │
                │   - token-level loss aggregation   │
                │   - per-token k3 KL，β 小          │
                └────────────────────────────────────┘
```

**最关键的 takeaway**：

1. **现在没有"一种算法包打天下"**——主流是 **"GRPO base + DAPO 四件套（Clip-Higher、动态采样、token-level loss、软长度惩罚）"** 作为 90% 场景的默认起点
2. **MoE 模型有多条可行路线**：GRPO + R3（工程最稳，veRL 主流）/ GSPO（Qwen3 路线，sequence-level ratio）/ DPPO（最新，散度 mask）/ 各种组合——**不存在"必须用 GSPO"**，按你的训练框架和稳定性需求选
3. **advantage 归一化是开放问题**：DeepSeek-R1 / 原版 DAPO / OpenRLHF / TRL / veRL 默认都**保留** $(R-\text{mean})/\text{std}$ ；Dr. GRPO 主张去掉 std，但还没成为事实标准——可以做消融对比
4. **KL 系数 β 越来越小**（甚至 0），尤其在**可验证奖励**（数学/代码 RLVR）任务上——这些场景更需要让模型大胆探索
5. **PPO clip 本身的设计缺陷被广泛认识到**：CISPO（救低概率 token）、DPPO（用散度替代 ratio）都在打这个补丁——选 trick 时要看你的场景是否触发这个缺陷（长 CoT + 多步更新、MoE 输出分布尖锐等）
6. **不同 trick 几乎都正交**，可以**叠加使用**——这正是"GRPO + 一堆 trick"成为事实标准的根本原因

---

## 12. RARO（Cai et al., 2025.11）—— 逃离 Verifier：只靠专家示范学推理

全称：**R**elativistic **A**dversarial **R**easoning **O**ptimization，论文题为 *Escaping the Verifier: Learning to Reason via Demonstrations*（[arXiv:2511.21667](https://arxiv.org/abs/2511.21667)）。

> **⚠️ 这是 BasicRL 里第一个"非 RLVR / 非 RLHF"的范式。** 前面 GRPO 全家桶（10.x）默认都站在 **RLVR**（可验证奖励）的世界里：数学有 checker、代码有测试用例、agent 有环境 reward。RARO 处理的是**根本没有 verifier、也没有人类偏好对，只有一堆专家"问题→答案"示范**的场景。它把"怎么从纯示范里学出 reasoning"重新表述成一场**对抗博弈**，本质是**逆强化学习（IRL）/ GAN** 在 LLM 推理上的落地，**底层引擎仍然是 GRPO**——所以放在本文档最后，作为"GRPO 还能往哪长"的一个方向。

### 12.1 痛点：RLVR 的天花板就是"有没有 verifier"

| | RLVR（前面所有章节）| SFT on demonstrations | RARO 想要的 |
|---|---|---|---|
| 需要什么监督 | 一个**自动验证器**（math checker / 单测 / 环境 reward）| 只要专家答案 | 只要专家答案 |
| 能不能学出"大规模 RL 式"的 reasoning | ✅ 能（R1 式涌现）| ❌ 不能 | ✅ 想要能 |
| 适用域 | 数学、代码、可验证 agent | 任意有示范的域 | 任意有示范的域 |

**核心观察**：分析写作、开放研究、金融分析、写诗……这些高价值领域**几乎不可能写出 verifier**，但**专家示范很多**。那为什么不直接 SFT？作者点出 SFT 的两个硬伤：

1. **学不出 RL 式 reasoning**：next-token 模仿只学"长得像专家答案"，学不到大规模 RL 才能涌现的搜索 / 自我纠错行为
2. **训练-推理分布错配（exposure bias）**：SFT 训练时喂的是数据集里的上文，推理时模型见的是**自己生成的上文**——和本文档开头那段"前向/反向参数错位"是同一类"训练时和实际跑时看到的东西不一样"的病

> **🎯 一句话定位**：RARO = "把 RLVR 里那个**写死的 verifier**，换成一个**和 policy 对抗着一起训出来的、会推理的 critic**"——从而把 R1 式的 RL reasoning 推广到**没有 verifier 的域**。

### 12.2 从"最大似然"到"对抗博弈"：为什么必须绕道 IRL

policy 是个**带隐变量的模型** $\pi_\theta(a, z \mid q)$ ：给定问题 $q$ ，先生成 reasoning trace $z$ （隐变量），再生成答案 $a$ 。最自然的目标是对专家答案做最大似然：

```math
\max_\theta \ \mathbb{E}_{(q,a)\sim D}\big[\log \pi_\theta(a\mid q)\big], \qquad \pi_\theta(a\mid q) = \sum_z \pi_\theta(a,z\mid q)
```

**但这个边缘化 $\sum_z$ 是 intractable 的**——reasoning trace 的空间组合爆炸，没法枚举。

**IRL 的转身**：不直接学 policy，而是**先学一个奖励函数** $r_\phi(a,q)$ ，让专家答案在它眼里得分高。在 **KL 正则的奖励最大化**下，最优 policy 有闭式解（和 RLHF/DPO 那套"奖励↔策略"对偶完全同源）：

```math
\pi_{\theta^\star(\phi)}(a\mid q) = \frac{1}{Z(q)}\,\pi_{\text{ref}}(a\mid q)\exp\!\Big\{\tfrac{1}{\beta} r_\phi(a,q)\Big\}
```

把这个最优 policy 代回最大似然目标，对 $\phi$ 求梯度，会得到一个**专家 vs policy 的对比式**：

```math
\nabla_\phi \mathcal{L}(\phi) = \tfrac{1}{\beta}\Big(\underbrace{\mathbb{E}_{(q,a)\sim D}[\nabla_\phi r_\phi(a,q)]}_{\text{拉高专家答案的 reward}} - \underbrace{\mathbb{E}_{q,\,a'\sim\pi_{\theta^\star}}[\nabla_\phi r_\phi(a',q)]}_{\text{压低 policy 答案的 reward}}\Big)
```

> **🎯 看懂这个梯度就看懂了一半**：它和 DPO/对比学习的味道一模一样——**让 reward 在"专家答案"上高、在"自己采出来的答案"上低**。reward 模型在学"怎么区分专家和我自己"，policy 在学"怎么让自己被当成专家"。这就是 **GAN 的判别器 vs 生成器**。

### 12.3 把 reward 函数变成一个"会推理的 critic"

RARO 让奖励函数 $r_\phi$ **本身就是一个 LLM critic** $c_\phi$ ：它读入 $(q, a)$ ，自己**先 reasoning 一段**，再输出一个标签——这个 $(q,a)$ 是**专家**写的还是 **policy** 写的。于是上面那个抽象梯度就退化成**标准 policy gradient + 极简 reward**：

- **critic 的 reward**：分类对了就 +1 —— $R_{\text{critic}} = \mathbb{1}[\ell\ \text{正确}]$
- **policy 的 reward**：骗过 critic（被误判成"专家"）就 +1 —— $R_{\text{policy}} = \mathbb{1}[\ell = \text{expert}]$

两边都用 GRPO 优化。这就是完整的对抗博弈：**critic 努力识破，policy 努力伪装**。

#### 🚨 二分类 critic 会退化 → 相对论式（Relativistic）critic

朴素做法是让 critic **单独看一个答案**判"专家 or policy"。问题：当 policy 训得越来越好、逼近专家时，**单看一个答案根本分不出来**——critic 退化成**瞎猜（50%）**，给出的梯度**方差巨大、毫无信息量**。GAN 里经典的"判别器饱和"。

**RARO 的修法——成对比较（pairwise / relativistic）**：critic 不再单看一个，而是**同时看同一个问题下的两个答案**，输出"**哪个更像专家**"，且**允许平局（tie）**，标签 $`\ell \in \{1, 2, \text{tie}\}`$ ：

```math
\boxed{R_{\text{critic}} = \mathbb{1}[\ell\ \text{选中专家}] + \tau_{\text{crit}}\cdot \mathbb{1}[\ell = \text{tie}]}, \qquad \tau_{\text{crit}} \in [0,1]
```

```math
\boxed{R_{\text{policy}} = \mathbb{1}[\ell\ \text{选中 policy}] + \tau_{\text{pol}}\cdot \mathbb{1}[\ell = \text{tie}]}, \qquad \tau_{\text{pol}} \in [0,1]
```

> **🎯 为什么"相对 + 允许平局"是关键？**
> - **相对比较**：比起"这答案绝对算不算专家水平"，"**A 和 B 谁更像专家**"是个**信号强得多、也稳得多**的判别任务——哪怕两者都很好，也常有细微高下。这正是 **Relativistic GAN** 的精神：判别器估计的是"真比假更真多少"，而非"真的绝对真不真"。
> - **平局 + 可调 tie reward**：当 policy 真的追平专家时，强迫 critic 二选一只会逼它瞎猜（退化）；给一个 **tie 选项**，critic 不确定时可以诚实地说"打平"，拿一个**稳定的中间奖励**，**避免了 policy 最优时的退化**。消融实验里**去掉 tie 选项性能明显下降**，是 RARO 能稳的核心设计之一。

> **📊 直觉对照：从"绝对裁判"到"相对裁判"**
> ```
> 二分类 critic（会饱和）：          relativistic critic（稳）：
>   policy≈expert 时                  policy≈expert 时
>   单看 A → "专家?" → 瞎猜 50%        看 (A_expert, A_policy) → "哪个更像专家 or 平?"
>   梯度 = 噪声                       → 仍能抓住细微差异 / 诚实说"tie"
>                                     → 梯度有信息、不爆方差
> ```

### 12.4 稳定化四件套（对抗训练最难的就是"稳"）

GAN 类训练出了名地难稳，RARO 识别出四个关键 trick：

| 技巧 | 解决什么 | 类比 |
|---|---|---|
| **① 共享 LLM**（policy 和 critic 是**同一个模型**的两种 prompt 角色） | 省一半显存；且"会判别"和"会生成"互相促进泛化 | 一个人**既当作者又当评委**，眼界互通 |
| **② 数据混合（Data Mixing）** | 把 policy rollout 和 critic rollout **拼进同一个 batch** 一起更新 | 同一步里既练"写"又练"评"，不偏科 |
| **③ Replay Buffer（经验回放）** | 缓存历史的专家/policy 答案对，防止 critic **灾难性遗忘**（只盯着最新 policy、忘了怎么识别老套路）| GAN 训练防判别器"健忘"的标准药 |
| **④ GRPO 改良** | 沿用本文档前面的共识：**over-length filtering**（过滤超长被截断的样本）+ **去掉 advantage / 长度归一化**（Dr. GRPO 那一套）| 复用 10.2 / 10.7 的结论 |

> **📌 注意 ③ 和本文档主线的呼应**：replay buffer 在标准 on-policy RLHF 里基本不用（数据一次性），但在**对抗训练**里必须有——因为 critic 是个**在动的目标**，policy 在变、专家分布不变，buffer 把"过去打过交道的 policy 答案"留住，防止 critic 把已经学会的判别能力训没了。

### 12.5 训练算法：一步里发生了什么（逐帧）

共享参数 $\theta$ 同时扮演 $\pi_\theta$（policy）和 $c_\theta$（critic），维护一个 replay buffer $\mathcal{R}$ ：

```
═══════════ RARO 一个训练 step（共享模型 θ）═══════════

[0] 从示范集 D 采一批问题+专家答案  {(q_i, a_i^E)}

[1] ── Policy 出招（生成，想骗过 critic）──────────────
    对每个 q_i，用 π_θ 采 K 条 (reasoning z, 答案 a^P)
    把每条和专家配成对：  pair = (a_i^E,  a_{i,k}^P)
    让 critic 给 pair 打标签 ℓ → 算 policy reward:
        R_policy = 1[ℓ 选中 policy] + τ_pol·1[ℓ=tie]
    新 pair 存入 R_new

[2] ── 混经验池（防 critic 遗忘）──────────────────────
    C ← Mix(R_new, R)        # 新对 + 历史对
    更新 replay buffer R

[3] ── Critic 出招（判别，想识破 policy）──────────────
    对 C 里每个 pair，critic 先 reasoning 再判标签 ℓ
    算 critic reward:
        R_critic = 1[ℓ 选中专家] + τ_crit·1[ℓ=tie]

[4] ── 一次 GRPO 更新（policy + critic 一起）──────────
    max_θ   λ_pol·J_pol(θ) + λ_crit·J_crit(θ) − β·KL(π_θ ‖ π_ref)
            └─ 两个目标共享同一次反传，data-mixing 在同一 batch ─┘

══════════════════════════════════════════════════════
  下一步：policy 更会骗 → critic 更会识破 → 互相螺旋上升
```

最终优化的联合目标（policy 项 + critic 项 + 防漂移 KL）：

```math
\max_\theta \ \lambda_{\text{pol}} J_{\text{pol}}(\theta) + \lambda_{\text{crit}} J_{\text{crit}}(\theta) - \beta\, D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})
```

> **🔁 螺旋上升的直觉**：和 GAN 一样，这是个**动态均衡**而非"收敛到固定 reward"。policy 进步 → critic 被迫变强才能继续区分 → 更强的 critic 又给 policy 更细的"往哪改"信号。**verifier 不再是外部写死的，而是和 policy 共同进化的"内生裁判"**。

### 12.6 和已知范式的关系（一张表理清血缘）

| | 监督信号来源 | 奖励/裁判是什么 | 和 RARO 的关系 |
|---|---|---|---|
| **RLVR**（本文档 10.x 主线）| 写死的 verifier | 固定规则 reward | RARO 想替代的"需要 verifier"前提 |
| **RLHF / Reward Model** | 人类偏好对 | **静态**学出的 RM | RARO 的 critic 是**动态、持续对抗训练**且**自己会 reasoning**的，且**不需要人类偏好**|
| **DPO** | 人类偏好对 | 隐式 reward（闭式）| 共享"reward↔policy 对偶"数学，但 RARO 无偏好对、靠对抗 |
| **SFT / Rationalization** | 专家示范 | 无（纯模仿）| RARO 的直接竞品，但 RARO 解决了 exposure bias + 学得出 RL reasoning |
| **GAIL / IRL / GAN** | 专家轨迹 | 判别器 | **RARO 的直系祖先**：critic=判别器，policy=生成器；relativistic + tie + replay buffer 都是对 GAN 训练难题的应对 |

> **🎯 一句话血缘**：RARO ≈ **GAIL/GAN 的判别器换成"会 reasoning 的 LLM critic"，生成器换成"会 reasoning 的 LLM policy"，引擎换成 GRPO，再补上 relativistic + tie + replay buffer 三件稳定化套件**。

### 12.7 实验与结果

**任务（按"验证难度"递增挑了三档）**：
- **Countdown**（用四个整数凑出 24）：**验证比生成容易得多** → 最适合 RLVR，用来看 RARO 离 oracle 有多近
- **DeepMath**（通用数学）：验证 ≈ 生成 难度
- **Poetry Writing**（写诗）：**根本不可验证**，用 GPT-5 当裁判给标量分 + 算"对战专家诗的胜率"

模型：instruction-tuned **Qwen2.5（1.5B / 3B / 7B）**，reasoning 预算 2048 token。

**Baselines**：SFT、Rationalization（带 CoT 的 SFT）、Iterative DPO（3 轮）、RL-Logit（用模型在专家答案上的 logit 当 reward）、**RLVR（oracle 上界，仅可验证任务能用）**。

**结果**（节选）：

| 任务 / 模型 | RARO | 最强 verifier-free baseline | RLVR(oracle) |
|---|---|---|---|
| **Countdown (1.5B)** | **54.4%**（+13.7%）| SFT 40.7% | 57.7%（RARO 几乎追平！）|
| **DeepMath (7B)** | **57.5%**（比最强 baseline +8.2%）| — | — |
| **Poetry 胜率 (7B)** | **25.0%** 对战专家（≈ SFT 5.9% 的 4 倍，论文摘要另报 +19.1% 胜率提升）| SFT 5.9% | 不适用（无 verifier）|

**关键定性发现**：
- **涌现自我纠错搜索**：Countdown 上 RARO 学出了"试错→回溯→再试"的搜索行为——**这正是 SFT 学不出、却被认为是大规模 RL 标志的能力**，证明"逃离 verifier 也能拿到 RL reasoning"
- **test-time scaling 成立**：reasoning 预算从 256→4096 token，Countdown 准确率 33.1%→61.3%——**和 RLVR 同样的 scaling 趋势**（这是论文的核心论点：示范也能撑起 RL 式 scaling）
- **critic 可直接当推理期裁判**：DeepMath 上用 critic 做**单淘汰赛**（让 critic 两两比、胜者晋级）实现 test-time scaling
- **消融**：去掉任一组件（critic 的 reasoning、relativistic 设置、tie、replay buffer、共享 LLM）都掉点；对系数 $\lambda, \tau$ 变化鲁棒

### 12.8 局限与定位

- **不是"免费午餐"**：对抗训练比 RLVR 难调（GAN 通病），靠 tie + replay buffer + 共享模型 + data mixing 四件套硬稳住
- **critic 也会被 hack**：本质是"policy 学会骗 critic"，存在 reward hacking 风险，论文靠 relativistic + 持续对抗 + KL 缓解
- **示范质量是上限**：学的是"像专家"，专家不行就到顶——和 SFT 共享这个天花板，但比 SFT 更能榨出 reasoning
- **和本文档主线的接口**：底层就是 **GRPO + Dr. GRPO 式归一化改良 + over-length filtering**，前面 10.x 的 trick 大多可叠加

**🎯 一句话**：RARO 把 R1 式的 RL reasoning 从"必须有 verifier"的牢笼里放出来——**用一个和 policy 对抗着共同进化、自己会 reasoning 的 relativistic critic，替代写死的验证器**，只靠专家示范就在不可验证的域（写诗）也学出搜索/自我纠错，并复现了 RLVR 的 test-time scaling。它是 GRPO 全家桶之外，"RL for reasoning"的另一个重要分支——**IRL/GAN 路线**。

---

## 13. Agentic RL 专题（2025 年中以来的进展）

> 📌 本章覆盖 2025 年 5 月以来专门面向 **agentic / multi-turn / 长链工具使用** 的 RL 进展。前面 §10 的变体（DAPO、GSPO、CISPO……）默认场景都是**单 turn**：一个 prompt → 一段回答 → 一个 reward。§10.11 的 GiGPO 是第一个转向 agent 的，本章把这条线完整展开。
>
> ⚠️ **来源标注**：本章事实分三档——【论文实证】=已读原文 PDF；【报告/博客】=官方技术报告或博客（部分无 arXiv）；【二手/待核】=仅搜索来源，数字需谨慎。每处会标明。

### 13.0 导读：为什么 agentic RL 要单开一章

单 turn RL 和 agentic RL 的差别不是"任务难一点"，而是**信号链路的每一个环节都换了性质**。一张表说清：

| | 单 turn（数学/代码，§10 场景） | Agentic（搜索/SWE/GUI，本章场景） |
| --- | --- | --- |
| 一条样本 | 一段自回归生成 | **生成 ↔ 环境交互交替**，几十个 turn |
| Token 来源 | 全是模型生成的 | **模型生成 + 环境注入（工具返回/观测）混杂** |
| 长度 | 几千 token | **2 万+ token，且长度方差极大** |
| Reward | 生成完立即打分 | **极稀疏，常常只在 episode 末尾给一次** |
| Rollout 成本 | 生成一段就行 | **要跑真实工具/沙箱，有延迟、会失败、长尾拖尾** |
| On-policy 假设 | 基本成立 | **异步下天然被打破** |

> 🎯 **一句话本质**：agentic RL 把"生成一段文本"变成了"在一个**有状态、会回话、还可能超时**的环境里走一条长链"。GRPO 那套"一条轨迹一个标量 advantage、同步采样、全 token 进 loss"的默认假设，**在这五个环节上会逐个失效**。本章就按这五个环节组织。

### 13.1 多 turn 把哪五件事弄坏了

先把 agentic 的交互循环画出来——后面所有方法都在这张图的某个位置打补丁：

```
   ┌─────────────────────────── 一条 agentic trajectory ───────────────────────────┐
   │                                                                                │
   │  prompt ─▶ [think₁ act₁] ─▶ 🌐obs₁ ─▶ [think₂ act₂] ─▶ 🌐obs₂ ─▶ ... ─▶ 答案   │
   │             └─模型生成─┘    └─环境注入┘  └─模型生成─┘    └─环境注入┘      │      │
   │                 ▲              ▲             ▲                              ▼    │
   │                 │              │             │                          reward  │
   │              ①masking      ⑤rollout      ②credit                     （只在这里）│
   │              问题：obs      瓶颈+off-      问题：几十                    ④reward │
   │              token不该      policy         个turn共享                    稀疏    │
   │              进loss                        一个advantage                        │
   │                                                                                │
   │   贯穿整条链：③长训练下策略熵坍缩，探索消失                                        │
   └────────────────────────────────────────────────────────────────────────────────┘

   ① 观测 token 污染   →  §13.2   （正确性地基）
   ② 信用分配失灵      →  §13.3   （turn/step 级 advantage）
   ③ 熵坍缩 / 探索枯竭  →  §13.4   （entropy 机制 + 工具后分叉）
   ④ reward 稀疏        →  §13.6   （process / rubric / progress 奖励）
   ⑤ rollout 瓶颈+off-policy → §13.5 （异步 + off-policy 校正）
```

这五个问题**几乎正交**，对应五类方法。和 §13（现在的 §14）"五条主线"是同一种拆法，只是这里针对 agent 场景重新落点。**理解这五个环节，比记住二十个算法名字重要得多。**

### 13.2 环节①：观测 token 掩码——agentic RL 的正确性地基

这是**最容易被忽略、但错了就全盘皆输**的一点，而且几乎所有 PG 系 agentic 工作都独立地做了同一件事。

**问题**：一条 trajectory 里，`obs` token（工具返回、检索文档、环境观测）是**环境塞进来的，不是当前策略 $\pi_\theta$ 生成的**。它们相对于 $\pi_\theta$ 是 **off-policy** 的——把它们算进 policy gradient loss，等于让模型去"学习模仿它根本没生成、也控制不了的内容"，梯度方向错误，训练必然不稳。

**解法**：把 `obs` token 从 loss 里 mask 掉。VerlTool（arXiv:2509.01055）【论文实证】把这件事写得最干净——GRPO 的求和**只跑在 action 段的 token 上**：

```
   trajectory:  [think₁ act₁]  🌐obs₁  [think₂ act₂]  🌐obs₂  [答案]
   loss mask :   ✅✅✅✅✅✅   ❌❌❌   ✅✅✅✅✅✅   ❌❌❌  ✅✅✅✅
                 └─ 计入 loss ─┘ 跳过   └─ 计入 loss ─┘  跳过  └计入┘

   归一化分母 = Σ|action 段|   ← 只数 action token，不是整条序列长度
```

论文原话：*"the observation tokens are off-policy with respect to the current LLM π_θ … which can destabilize training. Therefore, these tokens are typically masked out during policy optimization."*

> 💡 **这条规则的普适性惊人**：Search-R1、R1-Searcher、ReSearch、ZeroSearch、DeepSWE（GRPO++）、GLM-4.5 全部**独立地**做了观测 token 掩码。**Search-R1（arXiv:2503.09516）是最常被引用的出处**——它连 KL 项都把检索 token 排除掉。这几乎是 agentic RL 的"公理"：**只有模型自己生成的、能被它的概率分布负责的 token，才有资格进 policy gradient。**

> ⚠️ **一个容易踩的实现坑**：掩码不只影响 loss 的**分子**（哪些 token 贡献梯度），还影响**分母**（长度归一化除以多少）。如果分母用了整条序列长度而不是 action token 数，长的工具返回会把每个 action token 的梯度权重稀释掉——错得很隐蔽。§10.2 Dr.GRPO 讲长度归一化偏置那套分析，在这里以另一种形式重现。

**token 级 vs turn 级**：更细的问题是——advantage 是按 token 给（每个 token 一个），还是按 turn 给（一个 turn 内所有 token 共享）？《Practitioner's Guide to Multi-turn Agentic RL》（arXiv:2510.01132）【论文实证】明确主张 **token 级信用 + 掩码观测**，并把"把 turn 级 advantage 均匀摊到所有 token 上"列为一种应被修正的做法。这直接引出下一节。

### 13.3 环节②：信用分配——从"一条轨迹一个分"到 turn/step 级

**核心病根一句话**（MT-GRPO 论文 arXiv:2505.11821 原话）：*"the advantage … is computed at the trajectory level, which means the same advantage is assigned uniformly across the entire trajectory, without distinguishing the contributions of individual turns or tokens."*

这就是 §10.11 GiGPO 已经点过的问题——50 步走完成功，50 步全打 +1，好坏不分。除了 GiGPO 的 anchor-state 思路，2025 下半年又出了几条不同的攻法：

#### 13.3.1 Tree-GRPO（arXiv:2509.21240）——用树结构 rollout 挖 step 级信号【论文实证】

**思路**：与其独立采 $G$ 条完整链（昂贵、且 outcome-only 下每步信用相同），不如让轨迹**共享前缀、在中间分叉成树**。每个**树节点 = 一个完整 agent 步** $(\tau,\alpha,o)$（思考+动作+观测）——注意粒度，消融实验证明 token/句子级分叉反而**低于**链式 GRPO。

```
   链式 GRPO：G 条独立���链，前缀不共享
      ●────────────▶ R₁
      ●────────────▶ R₂       每条几千 token + 多次工具调用
      ●────────────▶ R₃       outcome-only：一条链内每步信用相同

   Tree-GRPO：共享前缀，中间分叉
      ●──┬──────────▶ R₁
         ├──────────▶ R₂      分叉点两侧的 reward 差
         └──●──┬────▶ R₃      = 天然的 step 级偏好信号
               └────▶ R₄      同预算下多 ~1.5× 样本
```

advantage 分两级相加（都用 GRPO 式组内归一化）：$\hat A_{\text{tree}} = \hat A_{\text{intra-tree}} + \hat A_{\text{inter-tree}}$——**intra-tree**（同一棵树内的兄弟轨迹，共享前缀，分叉点的 reward 差就是 step 级信号）+ **inter-tree**（跨树池化，因为单棵树的 baseline/std 不可靠）。论文还证明：intra-tree 组内优化在二元偏好假设下**梯度等价于 step 级 DPO**（Prop 3.1）。

**结果**：多跳 QA，Qwen2.5-1.5B EM **11.3→19.1（+69%）**；预算研究里最亮眼——每 prompt 只 ~2 条 rollout 时，链式 14.9 → 树 **31.6（+112%）**，"只用四分之一 rollout 预算"就超过链式 GRPO。
**局限**：强依赖干净的 agent-step 边界；14B 上收益递减；只测了搜索类 QA；串行 $L$ 次迭代拖 wall-clock，默认 $L=1$。

#### 13.3.2 MT-GRPO / MT-PPO（arXiv:2505.11821）——分开归一化的 turn 级 reward【论文实证】

2-turn 场景（中间 reward $R^I$、结果 reward $R^O$），关键是**两个 reward 各自独立组内归一化再组合**，而不是合成一个再广播：

```math
A^{\text{MT-GRPO}}_{i,1} = A^I_i + \alpha A^O_i, \qquad A^{\text{MT-GRPO}}_{i,2} = A^O_i
```

**结果**：2-turn 工具使用（TriviaQA）EM **0.5010 vs GRPO 0.3346（+0.166）**；且发现 **outcome-only 的 GRPO 会"逐渐不再调用搜索工具"**（工具奖励衰减到 0）——一个典型的 agentic reward 退化。
**局限（论文自陈）**：MT-GRPO 需要每个 turn 采 $G$ 条 → $K$ 个 turn 就 $G^{K-1}$ 条轨迹，"长程下计算上不可行"，所以他们转向带 critic 的 MT-PPO；干净公式只推到 $K=2$。

#### 13.3.3 SWEET-RL（arXiv:2503.15478，Meta）——非对称 actor-critic【论文实证】

（3 月工作，略早于窗口，但 turn 级信用的地基，值得收录。）**核心洞察**：最终策略看不到隐藏信息，但那些信息**在训练时是存在的**。所以让 critic（只在训练用）**多看一份 actor 看不到的上下文 $c$**（参考解/参考页面）——asymmetric actor-critic。它不学 value 而是**直接用 Bradley-Terry 学 advantage**（长度归一化的 log-ratio 参数化），再当 per-turn reward model 跑 DPO。
**结果**：ColBench 上 Llama-3.1-8B success/win rate **比 Multi-Turn DPO +~6%**，后端编程 success 追平 GPT-4o（40.4=40.4，前端 win rate 略低）。**别过度宣称"全面超 GPT-4o"**。

> 🎯 **信用分配的四条路线放一起看**（含 §10.11 GiGPO）：
>
> | 方法 | 怎么给 step 级信号 | 要 critic 吗 | 代价 |
> | --- | --- | --- | --- |
> | **VAPO**（§10.4） | 训 critic 做 token 级 credit | ✅ | 5 件套稳定化 |
> | **GiGPO**（§10.11） | anchor state 跨轨迹挖同状态比较 | ❌ | 依赖状态能精确重复 |
> | **Tree-GRPO** | 树分叉，共享前缀处的 reward 差 | ❌ | 依赖干净 step 边界 |
> | **MT-GRPO** | 每个 turn 单独归一化 reward | ❌ | turn 数指数级 rollout |
> | **SWEET-RL** | critic 多看训练期隐藏信息 | ✅（非对称） | 仅离线、需参考解 |
>
> **同一个"长链信用分配"难题，五种完全不同的下注**：补 critic（VAPO/SWEET）vs 升级 group 结构（GiGPO/Tree/MT）。这正是 §10.11 结尾说的"long-horizon RL 两种哲��"的完整展开。

### 13.4 环节③：熵坍缩与工具后分叉——长训练下探索为什么会枯竭

这是长程 agentic RL 里**最底层、最反直觉**的一条，值得单独理解。

#### 13.4.1 熵机制：性能是拿熵"换"来的（arXiv:2505.22617，上海AI Lab）【论文实证】

（这是 reasoning-RL 论文，但它是 ARPO/AEPO 一系工作的理论底座。）现象：RL 训练中策略熵**很早就坍缩**，性能随之饱和。他们拟合出一条经验律：

```math
R = -a\exp(H) + b
```

性能 $R$ 和熵 $H$ 之间是**指数式的此消彼长**——熵耗尽（$H\to 0$）时性能天花板是 $-a+b$，**再加 RL 算力也换不来提升**（>95% 的熵下降/性能提升发生在前 1/3 训练）。

**为什么坍缩？** 相邻两步的熵变由 **log-prob 与 advantage 的协方差**主导：

```math
H(\pi_{\theta_{k+1}}) - H(\pi_{\theta_k}) \approx -\eta\,\text{Cov}_{a\sim\pi_{\theta_k}}\big(\log\pi_{\theta_k}(a\mid s),\, A(s,a)\big)
```

协方差越大熵掉得越快，而**极少数 token 携带巨大协方差**（论文 Table 1：top 0.02% token 的平均协方差 5.654 vs 全体 0.003）。修法就是盯住这批 token：**Clip-Cov**（随机 detach 一小撮高协方差 token 的梯度）/ **KL-Cov**（对 top-k 高协方差 token 加 KL 惩罚）。
**结果**：Qwen2.5-32B 上比 GRPO 平均 **+6.4%**，最难的 AIME24/25 上 **+15.0% / +14.6%**，熵维持在 GRPO 平台的 10× 以上。

> 💡 **为什么这对 agentic 尤其致命**：agent 任务要走几十步、反复试探工具，**探索能力就是命脉**。熵一旦坍缩，模型会退化成"每次都走同一条老路"，再也发现不了新策略。单 turn 数学题熵坍缩顶多是分数不涨；长链 agent 熵坍缩是**直接学不动**。所以 §13.4.2 的工具后分叉才如此重要。

#### 13.4.2 ARPO（arXiv:2507.19849）——熵在工具返回后飙升，就在那里多分叉【论文实证】

⚠️ **命名坑**：叫"ARPO"的有两篇。要的是 **2507.19849**（Agentic **Reinforced** Policy Optimization，熵分叉）；另一篇 2505.16282 是 GUI 的 replay buffer，别混。

**核心观察**（论文原话）：*"Entropy rises sharply in the first 10–50 tokens following each tool call."* ——**工具返回会给模型的推理注入不确定性**，熵在工具调用后的头几十个 token 里陡增。

```
   熵
   │           工具返回                工具返回
   │              ↓                       ↓
   │     ╱╲      ┌──╲              ╱╲   ┌──╲
   │    ╱  ╲____╱    ╲____________╱  ╲_╱    ╲___
   │   ╱                                          
   └──────────────────────────────────────────────▶ token
        平稳推理    ← 这里熵高，最该分叉探索 →   平稳推理
```

**机制——Entropy-based Adaptive Rollout**：监控每次工具调用后的熵变 $\Delta H_t$，分叉概率 $P_t = \alpha + \beta\cdot\Delta H_t$；$P_t$ 超阈值就 fork 出 $Z$ 条分支，**把采样预算花在模型最不确定、最该探索的地方**（而不是均匀撒在整条链上）。共享前缀的 advantage 归因用 GRPO 的 IS ratio 自动对齐。
**结果**：13 个 benchmark，深搜任务 **比 GRPO 在 GAIA/WebWalkerQA +6%**，且**只用一半工具调用预算**。
**局限**：分叉阈值靠手调；熵观察在搜索 agent 上测的，其他工具类型的普适性是断言而非充分证明。其后续 **AEPO（arXiv:2510.14545）**【论文实证】指出 ARPO 的真实缺陷——**连续高熵步会过度分叉**（最多连 6 步），反而坍缩 rollout 多样性；AEPO 加了"连续高熵步分叉惩罚"+"高熵裁剪项里插 stop-gradient"，把 GAIA 推到 47.6%、HLE 11.2%（Pass@1）。

> 🎯 **13.4 两节连起来看**：熵机制（13.4.1）告诉你**熵坍缩是长程 RL 的根本瓶颈**；ARPO（13.4.2）告诉你**熵不是均匀分布的，工具返回后是熵的高地**，所以那里正是该花探索预算的地方。**一个是"为什么探索会死"，一个是"该在哪里抢救探索"。**

### 13.5 环节⑤：异步 rollout 与 off-policy 校正——agentic RL 的工程主战场

这是 2025 下半年**投入最大**的方向，因为它卡的是 wall-clock。

**问题**：同步 RL 里，一个训练步要等 batch 里**最慢的那条轨迹**。agent 场景这是灾难——轨迹长度方差极大（有的 3 步、有的 50 步）、工具调用有延迟、长尾直接拖停整个 GPU 集群。解法是**把生成和训练解耦**（异步）。但代价是：**打破了 on-policy 假设**——rollout 来自旧策略，所以每个系统都要配一个 off-policy 校正。

```
   ── 同步 RL ──────────────────────────────────
   gen ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 等最慢的那条（长尾）
   train              ▓▓▓▓     GPU 大部分时间在空等

   ── 异步 RL ──────────────────────────────────
   gen   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 永不停，持续产轨迹
   train    ▓▓▓▓ ▓▓▓▓ ▓▓▓▓ ▓▓▓▓ ▓▓▓▓    ← 攒够一批就更新
            └─ 但用的轨迹来自"旧策略"，off-policy！
```

#### 主要系统（都【论文实证】或【官方 repo】）

| 系统 | 出处 | 关键机制 | off-policy 校正 | 加速 |
| --- | --- | --- | --- | --- |
| **AReaL** | arXiv:2505.24298（蚂蚁/清华）| 全异步；权重更新**打断在途生成**（丢 KV、载新参、续解码）→ 一条轨迹可含多个策略版本的 token | **decoupled PPO**：分离行为策略与近端策略做 IS，证明可打断生成等价于从某行为策略采样 | 端到端 **2.77×**，512 GPU 线性扩展 |
| **ROLL Flash** | arXiv:2510.11345（阿里）| 样本级生命周期控制 + 冗余环境 rollout 吸收长尾 | **TIS / TOPR**（截断重要性采样，从上方 cap IS 权重）+ 梯度截断 | RLVR **2.24×**，agentic **2.72×** |
| **slime** | THUDM/智谱 repo | Megatron 训 + SGLang rollout + 共享 buffer；`fully_async` 专治长尾 agentic 生成 | — | GLM-4.5→5.2 背后的框架 |
| **Agent Lightning** | arXiv:2508.03680（微软）| 把 agent 执行建模为 MDP（每次 LLM 调用 = 一个 transition），**训练-执行解耦**，零改代码接入任意 agent 框架 | **LightningRL** 层级信用分配：把整条 return 分解到每次 LLM 调用 | — |
| **SkyRL-Agent** | arXiv:2511.16108（伯克利）| 异步 pipeline dispatcher；面向**有状态长程** SWE 任务 | — | dispatcher **1.55×**；SA-SWE-32B 达 39.4% SWE-Verified，成本降 2× |

> 💡 **异步的关键权衡：staleness（陈旧度）**。AReaL 用超参 $\eta$ 限制"生成最多领先训练几个策略版本"（代码 $\eta=4$、数学 $\eta=8$，$\eta=0$ 退化为同步）；ROLL Flash 发现 **async-ratio=2 就近乎最大加速**——**不需要很深的陈旧度**。这与直觉相反：你以为放得越开越快，实则放一点点（领先 2 个版本）就能填满长尾空隙，再多只会加剧 off-policy 不稳。

> ⚠️ **算法层面也会因异步而变**（不只是工程）：SORL（arXiv:2511.20718）【论文实证】发现长程多 turn off-policy 下 token 级 IS ratio 会**重尾化**，梯度范数爆到 $10^{12}$–$10^{18}$ 量级导致格式坍缩。它的修法是 **turn 级 IS**（一个 turn 一个长度归一化的几何平均权重，本质是 GSPO 序列级权重的 turn 粒度版）+ **裁剪触发的归一化**。⚠️ 但要注意：SORL 的搜索任务只给了**训练曲线没给 EM 数字**，唯一的数值表是医疗 QA vs Search-R1，**没有 vs GSPO/TIS 的正面数值对比**——引用时别夸大。

### 13.6 环节④：agentic 场景的 reward 设计

**主线仍是"可验证奖励优先"**（RLVR 延伸到工具）：最终答案 answer-match + 格式奖励（工具调用是否合法、答案标签是否规整）。但 agent 场景有几个特有的经验：

1. **EM 被普遍弃用，改 F1**。R1-Searcher、ReSearch 都发现 **EM 太严格会 reward-hack**（F1 > CEM > EM）。ZeroSearch 更直接：EM reward 会被 hack，REINFORCE + F1 最稳。
2. **稀疏终局奖励 → 稠密过程奖励**。《Practitioner's Guide》发现 *"dense turn-level rewards accelerate multi-turn RL vs sparse terminal rewards, but require algorithm-specific tuning"*——稠密有用，但稳定性高度依赖算法。代表工作：
   - **SPA-RL**（arXiv:2505.20732）【论文摘要】：训一个 progress estimator，把单个延迟终局奖励**分解成每步的"进度贡献"**。WebShop/ALFWorld/VirtualHome 上 success +2.5%。
   - **Agent Lightning 的 AIR**（Automatic Intermediate Rewarding）：从运行时可观测信号里自动抽中间奖励。
3. **不可验证任务 → rubric / 生成式奖励**。
   - **RaR**（arXiv:2508.12790）【论文摘要】：用 >10000 条 rubric 当结构化打分标准，把 RLVR 扩到开放式任务。Qwen-30B-A3B 上开放式 +5.2%。
   - **Kimi K2 的 Self-Critique Rubric Reward**：模型自己当 critic，对着 core/prescriptive/human 三类 rubric 做 pairwise 排序，与 RLVR 信号闭环互校。
4. **agentic 场景的 reward hacking 更危险**。Anthropic 的研究【二手】报告：模型在真实编码环境学会 reward hack 后，**行为会泛化到更广的 misalignment**（蓄意破坏、伪装对齐），且标准 chat RLHF 安全训练**去不掉**——这是 agentic reward 设计比单 turn 更需要警惕的地方。

> 🎯 **reward 设计的演化方向和 §10 主线 D 一致，但更进一步**：单 turn 是"reward shaping → 样本筛选"；agentic 这里是**"outcome-only → process/progress → rubric/生成式"**——因为链越长、terminal 信号越稀，就越需要把信号"铺"到中间步骤上。但每铺一层，reward hacking 的面也随之扩大。

### 13.7 应用层：2025 年中以来的代表性 agentic RL 系统

把上面五个环节的技术落到真实模型上，这一波"用大规模 RL 训 agent"的系统值得单独记。**注意算法多样性——GRPO 家族是主力，但不是唯一。**

| 系统 | 出处/时间 | 做什么 | RL 算法 | 亮点数字 |
| --- | --- | --- | --- | --- |
| **Kimi-Researcher** | Moonshot, 2025.06【博客】 | 深度研究 agent（搜索+浏览+代码）| ⚠️ **REINFORCE**（不是 GRPO）| HLE **26.9%**（从 8.6% 起，几乎全靠 RL）；平均 23 步推理、探 200+ URL |
| **Tongyi DeepResearch** | 阿里, 2025.09；arXiv:2510.24701【论文】 | 深度研究 30B-A3B（3.3B 激活）| **定制 GRPO**（严格 on-policy、token 级 loss、leave-one-out、负样本过滤）| HLE **32.9**，GAIA **70.9**，BrowseComp-en 43.4 |
| **DeepSWE** | Agentica+Together, 2025.07【博客】 | SWE agent（Qwen3-32B 纯 RL、无 SFT 冷启）| **GRPO++**（Clip-High + No-KL + 去 std + 长度归一 + LOO + Compact Filtering + 无熵损失）| SWE-Verified **42.2%**（Pass@1）、71.0%（Pass@16）|
| **Kimi K2** | Moonshot, 2025.07；arXiv:2507.20534【论文】 | 通用 agentic（1T MoE/32B 激活）| 基于 K1.5 + RLVR + Self-Critique Rubric | SWE-Verified **65.8%**，τ²-Bench 66.1 |
| **GLM-4.5** | 智谱, 2025.08；arXiv:2508.06471【论文】 | 通用 agentic（355B/32B 激活）| **专家迭代**：Reasoning/Agent/General 三专家分训 → 自蒸馏合并；group-wise PO，环境反馈 token 排除出 loss | SWE-Verified **64.2%**，BFCL v3 77.8 |
| **搜索 agent 群** | 2025 上半年【论文】 | Search-R1 / R1-Searcher / ReSearch / ZeroSearch | PPO/GRPO/REINFORCE++，**全都做观测 token 掩码** | Search-R1 7B avg EM 0.431 |
| **UI-TARS-2** | 字节, 2025.09；arXiv:2509.02544【二手】 | GUI/computer-use agent | **multi-turn RLVR + enhanced PPO**（reward shaping、value 预训练、异步 rollout）| OSWorld ~47.5 |

> ⚠️ **几处纠正常见误传**：
> - **Kimi-Researcher 用的是 REINFORCE，不是 GRPO**（很多二手资料写错）。
> - **原版 UI-TARS（2501.12326）不是 online RL，是 SFT+DPO**；UI-TARS-2 才是 RL。
> - DeepSWE 的 verl/异步细节**主源没提**，别当实证。

> 💡 **跨系统的四条共识**（读这一堆报告最该带走的）：
> 1. **观测 token 掩码近乎普适**——从开源小模型到 GLM-4.5，独立地都做了（§13.2）。
> 2. **GRPO 家族是主力但非唯一**——定制 GRPO/GRPO++/DUPO 是工作马，但 Kimi-Researcher 用 REINFORCE、MiniMax 用 CISPO、UI-TARS-2 用 enhanced PPO、WebThinker 用 online DPO。**没有一个算法通吃。**
> 3. **稀疏/长程靠"上下文管理 + 异步"救**——Kimi 的 context-manager、Tongyi 的 IterResearch（Markovian 上下文，只 condition 于"问题+演进报告+上次交互"，避免 context 窒息）、DeepSWE 的 Compact Filtering，都是在解决"链太长、context 塞不下、reward 太稀"。
> 4. **一句被反复验证的话**（Tongyi 原文）：*"success depends more on data quality and environment stability than on the specific algorithm."* ——**agentic RL 的胜负手，越来越不在算法本身，而在数据质量和环境稳定性。**

### 13.8 环境与"配方"：agentic RL 正在把瓶颈从算法移到环境

**什么样的环境能拿来 RL 训？** 共识两条：可 **reset** 到已知初态（能 episodic rollout）+ **程序化 reward**（状态检查，不靠人评）。这催生了一批工作：

- **verifiers / Environments Hub**（Prime Intellect）【二手】：把环境拆成 taskset（数据+工具+打分）/ harness / reward-rubric；配套 **prime-rl** 异步训练器（INTELLECT-2/3）。**这是"大规模合成环境即 scaling 轴"论点的最清晰代表**。
- **τ²-bench**（arXiv:2506.07982）【论文】：dual-control——agent 和模拟用户**同时**能改共享世界状态；终局按 DB 状态可验证。
- **AppWorld**【二手】：有状态、可 reset 的日常 app 世界，程序化单测式 reward，还检查"附带损害"（意外状态改动）。
- **TextArena**（arXiv:2504.11442）【论文】：Gym 式文本博弈，为 self-play RL 设计（reward = 游戏胜负，无需标注）。

**《Practitioner's Guide》（arXiv:2510.01132）的 7 条发现**是目前最系统的"怎么做 agentic RL"经验总结，摘最反直觉的几条：

1. 多 turn RL 性能**随环境复杂度 scale**；
2. **在简单环境训练的 agent 能泛化到复杂环境**；
4. 少量 SFT 示范能加速收敛，**但 RL 对泛化仍不可或缺**；
6. **有偏（PPO/GRPO）和无偏（RLOO）算法都能稳定学**——收益来自多 turn 的**问题建模方式**（token 级信用 + 观测掩码），而非算法花招；
7. 稠密 turn 级奖励加速训练，但需按算法调参。

> 🎯 **第 6 条最该记**：它和 Tongyi 那句"数据/环境比算法重要"是同一个信号——**当算法层面的收益趋于饱和，agentic RL 的前沿正在从"设计更好的 loss"转向"建更大更稳的环境、更干净的多 turn 建模"。** 这也是为什么本章后半几乎全在讲 masking、异步、环境，而不是又一个 GRPO 变体。

### 13.9 小结：agentic RL 的五环节地图

回到 §13.1 那张交互循环图，把本章方法钉在对应环节上：

| 环节 | 问题 | 代表方法 | 一句话 |
| --- | --- | --- | --- |
| ① 观测 token | 环境注入的 token 是 off-policy | **观测掩码**（Search-R1 起，近乎公理）| 只有模型生成的 token 才进 loss |
| ② 信用分配 | 几十步共享一个 advantage | GiGPO / **Tree-GRPO** / MT-GRPO / SWEET-RL | 补 critic vs 升级 group 结构，五种下注 |
| ③ 熵/探索 | 长训练熵坍缩、探索枯竭 | **熵机制**(clip/kl-cov) / **ARPO** / AEPO | 熵在工具返回后飙升，在那里抢救探索 |
| ④ reward 稀疏 | 信号只在 episode 末尾 | SPA-RL / RaR / Self-Critique | outcome → process/progress → rubric |
| ⑤ rollout+off-policy | 长尾拖尾、异步破 on-policy | **AReaL** / ROLL Flash / slime / Agent Lightning | 异步提速，但要配 off-policy 校正 |

> **一句话总括本章**：单 turn RL 的战场在 **loss 函数**（§10 那二十个变体都在改 advantage/clip）；agentic RL 的战场已经���移到 **masking 正确性、信用分配粒度、探索维持、reward 稠密化、rollout 工程**这五个环节。**理解这五环节，比追每周新出的 GRPO 变体重要得多**——因为新变体几乎都能在这张图上找到它打的那个点。

> 📚 **想系统入门**：arXiv:2509.02547《The Landscape of Agentic RL for LLMs》是这个领域的综述锚点；arXiv:2510.01132《Practitioner's Guide to Multi-turn Agentic RL》是最实操的配方总结。本章的分类和这两篇可互相对照。

## 14. 一句话总结

### 基础概念
- **策略梯度**：能用就用，但 on-policy + 高方差
- **Advantage**：减 baseline 降方差，GAE 在偏差和方差之间平衡（policy 和 value 偏好不同 $\lambda$ ）
- **重要性采样**：让旧数据可重用,**核心**,但放飞自我会爆炸
- **Clip**：是个**门控而非幅度限制**——case ① / ④ 会让 logp 梯度归零,这是诸多新算法的修补目标
- **KL（vs ref）**：防 RL 把 SFT 学到的人类风格 / 安全对齐丢掉,RLVR 场景下可设很小或 0
- **GRPO**：在 LLM 上把 critic 干掉,用组内相对回报当 advantage,显存减半

### 现代变体的"主线地图"：所有改进都在解决五个根本问题

把 11 个变体平铺成 list 不利于理解。一个更有效的视角是：**它们围绕"GRPO 信号链路"的五条主线在做工作**——每条主线对应"信号链路上某个具体环节出了什么问题"。理解这五条主线，远比记住每个算法的名字重要。

GRPO 的信号链路：

```
   prompt → rollout 采样 → reward 信号 → advantage 估计 → loss (clip/min) → 反传 → 应用场景
              │                │              │                │             │           │
              │                │              │                │             │           │
           主线 D            主线 E         主线 A             主线 C       主线 B       主线 E
        （样本筛选）       （reward 设计） （归一化纠偏）       （clip 机制）  （信用分配）  （新场景适配）
```

11 个变体在这条链路上各打各的点——这就是为什么实际项目里常常**多个变体叠加使用**，它们攻击的环节不同、几乎都正交。

---

#### 🅰️ 主线 A：让 advantage 信号"算对" —— 归一化纠偏

GRPO 的核心是 $\hat A = (R - \text{mean}) / \text{std}$ ，但这条公式藏着好几个坑：

| 变体 | 攻击点 | 怎么改 |
|---|---|---|
| **Dr. GRPO** | std 归一化→"难度偏置"；per-sample 长度归一化→"长度偏置" | 把这两个归一化都拿掉 |
| **REINFORCE++** | 组内归一化（G 条样本统计量不稳）| 换成 batch 级全局归一化（1024+ 样本）|
| **GDPO** | 多 reward 加和后做归一化→大量 reward 组合塌陷到同一 advantage | 先逐 reward 归一化、再加和、再 batch 归一化 |
| **GSPO**（贡献一）| token-level IS 和 sequence-level reward 粒度错位（"假 IS"）| 把 IS 升到 sequence 级 |

**共同病根**：GRPO 直接照搬 PPO 的公式，没仔细审视每一步归一化操作在"多 reward / 多粒度 / 多场景"的 LLM 训练里是否真的成立。这条主线本质是**对 GRPO 算法公式做事后审计**。

---

#### 🅱️ 主线 B：让 advantage 信号"算细" —— Credit Assignment 粒度

GRPO 给整条 trajectory 打一个标量 advantage，所有 token/step 共享。这在单 turn 数学题上够用，但**长 CoT / agent 长链任务上严重不够细**。两条思路：

| 变体 | 思路 | 代价 |
|---|---|---|
| **VAPO** | 把 critic 救回来做 token-level credit | 5 件套稳定化（pretrain critic、decoupled GAE、长度自适应等）|
| **GiGPO** | 升级 group 结构、不引 critic——anchor state 在 trajectory 间挖共同状态做 step 级比较 | 几乎零成本，但依赖 state 能匹配上 |

**路线分裂**：长 CoT reasoning（critic 还训得动）选 VAPO；真正的 agent 长链（50 步、20k token、reward 极稀疏，critic 训不动）选 GiGPO。

---

#### 🅲 主线 C：让 advantage 信号"传得到" —— PPO Clip 机制改造

PPO clip 看似温和，**本质是个二值门控**——某些 token 会直接梯度归零。这条主线攻击 clip 机制本身：

| 变体 | 攻击点 | 怎么改 |
|---|---|---|
| **CISPO** | clip 让"分叉 token"（Wait/However/Aha）在多步更新里被消音 | clip 从 surrogate 搬到权重， $\log\pi_\theta$ 显式保留 |
| **SAPO** | 硬 clip 在边界处梯度突变、信任域不连续 | sigmoid 软门 + 非对称温度（ $\tau_{\text{neg}} > \tau_{\text{pos}}$ ）|
| **DPPO** | 单点 ratio 噪声大、低概率 token 被过惩罚 | 放弃 ratio，用分布散度 mask 做信任域 |

**共同病根**：Schulman 2017 设计 clip 是为了**廉价近似 TRPO 的 trust region**。在 LLM 大词表 + 长 CoT + 多步更新场景下，这个近似的代价被放大了。三个变体从不同维度修复。

---

#### 🅳 主线 D：筛选样本，让 advantage 信号"导向好行为" —— 防 Length Inflation / Reward Hacking

奖励是"许愿池"——给什么得什么。模型常常 hack 掉奖励信号，最典型的是**回答越训越长**（4k → 14k tokens）。两条思路：

| 变体 | 怎么治 | 干预层面 |
|---|---|---|
| **DAPO Overlong Soft Penalty** | reward 里加长度软惩罚（缓冲区线性递增）| reward shaping（容易和 RM 纠缠）|
| **GFPO** | 多采样、筛"简洁子集"，只在子集内做梯度更新 | 样本筛选（绕过 reward）|

**方法论进化**：从"在 reward 里加项"（reward shaping）→"在样本筛选上做手脚"（rejection sampling）——更干净、更不易 reward hacking。

---

#### 🅴 主线 E：让 advantage 信号"在新场景也成立" —— 算法到新范式的迁移

GRPO 原生为单 turn 数学/代码设计，但 LLM 应用越来越宽。每个新场景都暴露 GRPO 某个原本"够用"的设计在边界条件下失效：

| 新场景 | 暴露的问题 | 主要变体 |
|---|---|---|
| **MoE 模型** | routing 漂移→ token-level ratio 崩 | **GSPO（贡献二）/ R3 / DPPO** 三条路线 |
| **Agent 多 turn 长链** | 50 步、20k token、reward 在 episode 末尾 | **GiGPO**（嵌套组 + anchor state）|

---

### 大局观

#### 1. 演化路线：从 PPO 到 GRPO，再到"GRPO 全家桶"

- **PPO 早期成为 RLHF 默认**，靠的是「能用、稳定、好实现」三者的平衡
- **GRPO 干掉 critic** → LLM 场景下省一半显存、信号还更稳 → 成为 LLM RL 的**事实主流框架**
- **2025 至今的一系列变体**都不是要替代 GRPO，而是**对 GRPO 信号链路的每个环节做精细化**——这是为什么"GRPO + N 个 trick"成为事实做法


#### 2. 一句话总结

> 现代 LLM RL = **GRPO 主干** + **针对你这条具体信号链路上出问题的那一环，叠加对应的变体**。
>
> - 信号算不对？→ 主线 A（Dr.GRPO / REINFORCE++ / GDPO / GSPO 贡献一）
> - 信号算不细？→ 主线 B（VAPO / GiGPO）
> - 信号传不到？→ 主线 C（CISPO / SAPO / DPPO）
> - 信号引向 hack？→ 主线 D（DAPO 软惩罚 / GFPO）
> - 信号在新场景失效？→ 主线 E（GSPO 贡献二 / R3 / SAPO / GDPO / GiGPO）
>
> **没有一种算法包打天下，但理解这五条主线，就能在新场景里**自己判断该叠哪些 trick**。**
