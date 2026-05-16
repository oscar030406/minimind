# MiniMind 预训练复现与优化报告

更新时间：2026-05-16  
公开仓库：<https://github.com/oscar030406/minimind>  
当前分支：`codex/pretrain-repro-report`

本分支 README 就是当前阶段的主报告；`master` 分支保持原 MiniMind 项目的干净同步版本。原项目说明请看 [master 分支](https://github.com/oscar030406/minimind/tree/master)。

## 0. 预训练优化总览

### 0.1 可优化参数清单

| 类别 | 可调项 | 说明 |
| --- | --- | --- |
| 数据 | 数据文件、数据比例、去重、过滤、长短文本比例、领域配比、shuffle、采样权重 | 预训练最重要变量之一。低质量和重复数据会直接浪费训练预算。 |
| 序列长度 | `max_seq_len` | 控制上下文长度、显存和 padding 浪费。MiniMind 当前重点比较 `340/384/512/768`。 |
| batch | `batch_size`、`accumulation_steps`、有效 batch、每 step token 数 | 影响显存、吞吐、梯度噪声和泛化。对比时应尽量做等 token 对比。 |
| 训练预算 | `epochs`、`max_steps`、总 token 数、保存间隔、断点续训 | “跑几轮”不如“看了多少有效 token”精确。 |
| 学习率 | `learning_rate`、warmup、cosine、constant、min lr、cooldown、WSD、分阶段 continuation | 当前本机实验里，学习率策略影响非常明显。 |
| 优化器 | AdamW、fused AdamW、`betas`、`eps`、`weight_decay`、Sophia、Lion、Adafactor | fused AdamW 已验证有稳定速度收益。 |
| 精度 | `bfloat16`、`float16`、TF32、FP8 | 本机采用 BF16；FP8 适合服务器验证。 |
| 模型规模 | `hidden_size`、`num_hidden_layers`、层宽比例 | 决定参数量和算力需求。当前主线是 64M Dense：`hidden_size=768`、`layers=8`。 |
| 注意力 | attention heads、KV heads、GQA/MQA、FlashAttention、长上下文注意力 | MiniMind 已有 GQA 和 PyTorch SDPA/Flash 路径。 |
| FFN | `intermediate_size`、SwiGLU 比例、激活函数 | 小模型可试 depth/width/FFN ratio 的性价比。 |
| 位置编码 | RoPE theta、max position、YaRN/NTK/RoPE scaling | 长上下文继续预训练时重点关注。 |
| 正则与稳定 | dropout、RMSNorm eps、QK norm、梯度裁剪、loss spike 处理 | 小模型一般先保持简单，遇到不稳定再动。 |
| MoE | `use_moe`、expert 数、top-k、router aux loss | 本机不优先，适合算力平台。 |
| 工程 | packing、bucketing、num_workers、pin memory、torch.compile、梯度检查点、DDP/FSDP | 影响训练速度和显存，不一定改变最终质量。 |
| 评估 | train loss、val loss、PPL、random holdout、tail holdout、strict holdout、SFT 后效果 | 不应只看训练 loss。 |

### 0.2 业界可尝试方法

| 方法 | 简短说明 | 对 MiniMind 的价值 |
| --- | --- | --- |
| Scaling laws / Chinchilla | 用模型大小、token 数、算力预算决定训练配比。 | 上服务器前估算该训多大模型、多少 token。 |
| 数据清洗 | 清掉乱码、模板页、低质文本、重复段落。 | 高优先级，尤其小模型更容易被脏数据带偏。 |
| 去重 | 文档级、段落级、MinHash、n-gram。 | 减少背诵和重复训练。 |
| 数据混合 | 中文、英文、百科、代码、数学、对话等按比例采样。 | 决定最终能力分布。 |
| Data annealing | 后期提高高质量数据、数学、代码、长文本比例。 | 值得在服务器做。 |
| Curriculum | 先短文本/简单数据，再长文本/难数据。 | 和我们已有 `seq384 -> seq512` 思路一致。 |
| Packing / bucketing | 拼接短文本或按长度分桶，减少 padding。 | 本机已验证非常值得。 |
| 文档边界控制 | packing 时用 EOS 或 attention mask 保留文档边界。 | 防止模型学习错误跨文档关系。 |
| 长上下文继续预训练 | 先短上下文训练，再低 LR 继续训长上下文。 | 当前最有希望方向之一。 |
| Tokenizer 优化 | 重新设计 vocab、中文/代码 token、特殊 token。 | 成本高，适合后期专题。 |
| LR schedule | warmup+cosine、WSD、cooldown、分阶段低 LR。 | 已证明有效，应继续系统化。 |
| 优化器替换 | Sophia、Lion、Adafactor 等。 | 可试，但工程风险高于 AdamW。 |
| μP 超参迁移 | 小模型调参，迁移到大模型。 | 算力平台上有价值。 |
| batch scaling | 找有效 batch 的最佳区间。 | 必须做等 token 对比。 |
| 架构调参 | depth/width、GQA、FFN ratio、QK norm、RoPE theta。 | 可作为服务器阶段第二优先级。 |
| MoE | 更多总参数、更少激活参数。 | 本机不优先。 |
| Multi-token prediction | 一次预测多个未来 token。 | 需要改模型和 loss，创新性较强。 |
| FIM / 代码目标 | 对代码数据做中间填空训练。 | 仅在代码能力目标明确时做。 |
| 合成数据 | 用强模型生成/过滤预训练数据。 | 有潜力，但要防止风格污染。 |
| 蒸馏式预训练 | 学 teacher logits 或 teacher 生成数据。 | 成本较高，后期再做。 |
| FlashAttention / SDPA | 提升长序列训练效率。 | 工程优化，服务器可继续测。 |
| 分布式训练 | DDP、FSDP、ZeRO、tensor parallel。 | 平台阶段必备。 |
| checkpoint averaging | 平均后期多个 checkpoint。 | 低成本，可在服务器验证。 |
| 评估体系优化 | 固定多套 holdout，并绑定 SFT 后效果。 | 必须做，否则优化结论不稳。 |

## 1. 当前仓库整理结果

- 官方源码：`model/`、`trainer/`、`dataset/`、`scripts/`、`eval_llm.py`。
- 预训练报告：根目录 `README.md`，也就是本文件。
- 预训练实验脚本：`experiments/pretrain/scripts/`。
- 平台实验配置：`experiments/pretrain/configs/`。
- 大数据集、权重、日志和本地虚拟环境不进入 Git。

## 2. 基础版项目复现

### 2.1 环境

- 系统：WSL Ubuntu 24.04
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，约 8GB 显存
- Python：3.12.3
- PyTorch：`2.11.0+cu128`
- Transformers：`4.57.6`
- Datasets：`3.6.0`
- CUDA：可用

公开仓库复现入口：

```bash
git clone https://github.com/oscar030406/minimind.git
cd minimind
git checkout codex/pretrain-repro-report

python -m venv .venv
source .venv/bin/activate

# 先按自己的 CUDA/驱动环境安装 PyTorch；本机复现环境是 torch 2.11.0+cu128。
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2.2 数据

公开仓库不包含数据集。复现时需要按原项目说明把 mini 数据集下载到 `dataset/`；本机复现时使用的数据文件如下，但这些文件不进入 Git：

| 文件 | 用途 |
| --- | --- |
| `pretrain_t2t_mini.jsonl` | 预训练 |
| `sft_t2t_mini.jsonl` | Full SFT |
| `lora_medical.jsonl` | LoRA medical |
| `dpo.jsonl` | DPO |
| `rlaif.jsonl` | PPO/GRPO/RLAIF |
| `agent_rl.jsonl`、`agent_rl_math.jsonl` | Agentic RL |

### 2.3 已完成复现

| 阶段 | 脚本 | 结果 |
| --- | --- | --- |
| Pretrain | `trainer/train_pretrain.py` | 完成 2 epoch，生成 `out/pretrain_768.pth` |
| Full SFT | `trainer/train_full_sft.py` | 完成 2 epoch，生成 `out/full_sft_768.pth` |
| SFT 推理 | `eval_llm.py` | 成功 |
| LoRA medical | `trainer/train_lora.py` | 1 epoch smoke 成功，生成 `out/lora_medical_repro1_768.pth` |
| LoRA 推理 | `eval_llm.py --lora_weight ...` | 成功 |
| DPO smoke | `trainer/train_dpo.py` | 64 条小样本 smoke 成功 |

关键训练结果：

| 阶段 | 最终记录 |
| --- | --- |
| Pretrain | `Epoch:[2/2](39695/39695), loss: 1.6362` |
| Full SFT | `Epoch:[2/2](56608/56608), loss: 1.7258` |
| LoRA medical 1 epoch | `Epoch:[1/1](1580/1580), loss: 1.5423` |
| DPO smoke | `Epoch:[1/1](64/64), loss: 0.7281` |

模型效果判断：

- 已能中文问答、生成简单解释、输出代码片段。
- 仍有重复、幻觉、概念混乱和长回答失控问题。

## 3. 已试过的预训练优化

| 已试方向 | 结论 |
| --- | --- |
| 官方 pretrain mini 全量 | 已完成，可作为基础复现基线。 |
| `seq340/384/512/768/1024/1536/2048` 探测 | 本机可跑多档，但稳定主线应保守。 |
| batch / 显存边界 | `seq512 batch24` 较稳；更大 batch 风险上升。 |
| fused AdamW | 明确提速，建议保留。 |
| packing | token 利用率大幅提升，是最值得工程化的方向之一。 |
| gradient checkpoint | 省显存但降吞吐，适合长 seq，不适合作默认。 |
| torch.compile | Windows 原生因 Triton 问题失败；当前 WSL/服务器可重测。 |
| constant LR | 明显不如 cosine，不推荐。 |
| cosine decay | 有效，是当前基础策略。 |
| min lr ratio = 0 | 收尾太冷，tail 变差，不推荐。 |
| cooldown / continuation | 当前最有价值，尤其 `cooldown -> seq512`。 |
| `seq512 continuation` | 比单纯从头加长更划算。 |
| 多 seed 复验 | `context512 lr2e-5` 较稳定。 |
| 轻度质量过滤 | 本机短跑几乎没收益。 |
| 长文本子集 | 可改善 tail，但泛化指标变差。 |
| 原始+长文本 80/20、90/10 | 没超过 cooldown/seq512 主线。 |
| strict late holdout | 支持 cooldown 泛化强、post-context512 更均衡。 |
| 高温高显存极限 | 不建议继续，本机安全收益比太低。 |

旧实验中较重要的候选：

| 用途 | 配置 | 结论 |
| --- | --- | --- |
| 泛化最强 | `cooldown1800_to_seq512_lr1e5_s600` | orig/random/strict 指标最强，但 tail 有牺牲。 |
| 综合主推 | `post_context512_lr2e5_base_lr5e6_w10_s600` | 三类验证集较均衡，适合作 SFT base。 |
| tail 最强 | `context512_best5000_lr1e5_w20_s1200_cosine_seed7` | tail loss 最好，但综合略弱。 |

## 4. 平台下一步优化方向

按优先级：

1. `smoke_s128_b8_s20`：先验证平台环境。
2. `reproduce_local_baseline_s384_5000`：复现本机 baseline。
3. `best_general_cooldown_two_stage`：验证 cooldown 泛化优势是否放大。
4. `balanced_seq512_post_three_stage`：验证均衡 SFT base。
5. `wsd_s384_long_probe`：测试 WSD 是否优于 cosine。
6. `scaling_probe_h512_l8`：小模型 scaling law 探针。

长文本混合实验暂不放入可直接运行配置，因为它依赖额外生成的长文本子集；历史结论保留在上文。

对应配置文件：

```text
experiments/pretrain/configs/pretrain_platform_experiments.json
```

## 5. 关键参考

- Chinchilla / compute-optimal training: <https://arxiv.org/abs/2203.15556>
- Llama 3 technical report: <https://arxiv.org/abs/2407.21783>
- DCLM / DataComp-LM: <https://arxiv.org/abs/2406.11794>
- Dolma: <https://arxiv.org/abs/2402.00159>
- OLMo: <https://arxiv.org/abs/2402.00838>
- FlashAttention: <https://arxiv.org/abs/2205.14135>
- WSD schedule: <https://huggingface.co/papers/2410.05192>
- μP: <https://arxiv.org/abs/2203.03466>
- Sophia: <https://arxiv.org/abs/2305.14342>
