# RSI survey 图片来源

这些图片用于 [RSI 学习笔记](../../RSI-Survey.md)，均来自 Yi Duan 等人的 *The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement*。

- 原报告：[arXiv:2609.11873v3](https://arxiv.org/abs/2609.11873v3)，2026-09-22，79 页。
- 原 PDF：[2609.11873v3](https://arxiv.org/pdf/2609.11873v3)。
- arXiv 标注的许可：[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)。版权与原图内容归原作者。
- 按 PDF 图形边界渲染为 PNG，保留图内标签与图例，未翻译、改绘或修改图内内容。原英文图注未嵌入图片，笔记紧邻图片提供图号、PDF 页码与中文说明。
- 所有图片保持原始宽高比、居中显示，并可点击查看原始 PNG。

| 本地文件 | 原图与 PDF 页码 | 内容 | PNG 尺寸 |
| --- | --- | --- | --- |
| `figure-01-landscape-overview.png` | Figure 1，p.1 | RSI 自主性路线图与代表研究和产业实践 | 1784 × 784 |
| `figure-02-autonomy-loops.png` | Figure 2，p.5 | 五层自主性及改进闭环的责任边界 | 1900 × 656 |
| `figure-03-capability-headroom.png` | Figure 3，p.8 | 跨领域 HCI 能力轨迹及人为设定的未来延伸 | 1808 × 1444 |
| `figure-04-strategy-search.png` | Figure 4，p.19 | 证据驱动的策略选择、诊断与搜索方法 | 1768 × 548 |
| `figure-05-experience-acquisition.png` | Figure 5，p.23 | 自适应出题和环境练习两种经验获取 | 1848 × 932 |
| `figure-06-data-pipeline-automation.png` | Figure 6，p.24 | 六项数据工作从人工到自动执行的定性分布 | 1724 × 908 |
| `figure-07-deployment-adaptation.png` | Figure 7，p.27 | 部署轨迹蒸馏、系统修订与更新验证 | 1848 × 964 |
| `figure-08-meta-improvement.png` | Figure 8，p.32 | 固定过程下的适应与改进过程本身的继承 | 1900 × 980 |
| `figure-09-application-regimes.png` | Figure 9，p.36 | 科学、具身、软件和医疗的更新对象与反馈条件 | 1960 × 948 |
| `figure-10-theseus-coevolution.png` | Figure 10，p.38 | Theseus 提出的环境、数据和模型共演化 | 1860 × 436 |
| `figure-11-lark-data-foundation.png` | Figure 11，p.39 | Lark 企业数据整理、评价与真实使用反馈 | 1836 × 388 |
| `figure-12-ima-dual-timescale.png` | Figure 12，p.40 | 小红书即时记忆与慢速后训练的双时间尺度 | 1884 × 516 |
| `figure-13-humanlaya-quality-loops.png` | Figure 13，p.41 | Humanlaya 当前数据修复与跨批次检查器更新 | 1796 × 512 |
| `figure-14-forge-engineering.png` | Figure 14，p.42 | ModelBest 项目实现优化与跨项目工程经验 | 1836 × 572 |
| `figure-15-hyra-experience-search.png` | Figure 15，p.43 | Hyra 并行探索与可执行经验库 | 1868 × 492 |
| `figure-16-agent-native-research.png` | Figure 16，p.44 | Agent-Native Research Lab 可核验研究记录和继承 | 1796 × 364 |
| `figure-17-frontis-meta-evolution.png` | Figure 17，p.45 | Frontis 专家演化与跨任务元改进 | 1884 × 644 |
| `figure-18-literature-targets.png` | Figure 18，p.63 | 491 篇文献的自主性和改进对象分布 | 1816 × 1208 |

提取坐标采用 PDF point，以页面左上角为原点，依次为 `(x0, y0, x1, y1)`；PyMuPDF 以 `Matrix(4, 4)`、不透明背景渲染：

| 图号 | 页码 | 边界坐标 |
| --- | ---: | --- |
| 1 | 1 | `(90, 450, 536, 646)` |
| 2 | 5 | `(70, 76, 545, 240)` |
| 3 | 8 | `(74, 78, 526, 439)` |
| 4 | 19 | `(90, 83, 532, 220)` |
| 5 | 23 | `(74, 72, 536, 305)` |
| 6 | 24 | `(87, 87, 518, 314)` |
| 7 | 27 | `(74, 72, 536, 313)` |
| 8 | 32 | `(70, 78, 545, 323)` |
| 9 | 36 | `(72, 76, 562, 313)` |
| 10 | 38 | `(75, 386, 540, 495)` |
| 11 | 39 | `(76, 77, 535, 174)` |
| 12 | 40 | `(70, 71, 541, 200)` |
| 13 | 41 | `(83, 76, 532, 204)` |
| 14 | 42 | `(80, 76, 539, 219)` |
| 15 | 43 | `(72, 79, 539, 202)` |
| 16 | 44 | `(83, 172, 532, 263)` |
| 17 | 45 | `(70, 71, 541, 232)` |
| 18 | 63 | `(78, 118, 532, 420)` |

Figure 3 的 2026 年后曲线是人为设定的示意；Figure 6 是公开实践的定性归纳；Figures 8、10 描述可能进展或拟构建循环；Figure 18 的统计反映综述收录与归类。它们都不能单独证明完整 RSI 已实现或改进持续加速。产业机制图的实验支持范围见笔记相邻正文。
