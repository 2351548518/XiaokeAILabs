# 训练自己的视觉版 Jev（多模态决策模型）

`text_jev_train` 走的是纯文本路线。这个目录把同样的判别式决策模型搬到视觉上：底座换成本身就懂图像的多模态模型，每条候选同时吃图像、上下文和候选文本，由 score 头给出一个标量分，再在本问题的候选之间做归一。

模型侧没有现成的开源训练实现可抄（AgentJev 是纯文本，Jev-Omni 训练权重不开源），这里是自建路线，参考了 `pngwn/system-one-qwen3.5-4b-scorer` 的架构组织方式（LoRA 加 scalar score 头、按问题归一、温度校准），但数据、协议、训练脚本都独立写。

> 当前版本修正了参数未更新、重复温度缩放及答案解析进入输入的问题。旧交接记录中的准确率、ECE 和耗时不能作为当前脚本的复现目标；修正后的完整训练尚需在训练机重新验证。

## 用到的东西

| 项 | 取值 | 说明 |
|---|---|---|
| 底座 | Qwen3.5-2B | Apache 2.0，原生多模态，权重约 4 GB |
| 数据集 | ScienceQA | CC BY-SA 4.0，非 gated，图像内嵌在 parquet 里 |
| 参数高效微调 | LoRA r=16 | 4B 全参在 128 GB 统一内存上跑不动，LoRA 是必选项 |
| 损失 | 软交叉熵 + 0.1×sum-candidate Brier | 与文本版一致 |
| 校准 | 单一标量温度，0.5～4.5 共 41 点 | 只在 calibration 集上拟合 |

底座也可以用 `Qwen3.5-0.8B`（权重 1.7 GB）或 `4B`（权重 8.8 GB），下载时改 `download_backbone.py --size`，训练和服务时传对应的 `--model`。`protocol.json` 是配方记录，运行参数以命令行及脚本默认值为准。

## 目录结构

```
vision_jev_train/
├── scripts/
│   ├── download_backbone.py   # 下载 Qwen3.5 多模态底座
│   ├── download_data.py       # 下载 ScienceQA test split
│   ├── prepare_data.py        # parquet 转 jsonl，按图片 sha256 去重切分
│   ├── train.py               # 训练、内存闸门、校准、测试
│   ├── evaluate.py            # 独立评测（图像预 resize，避开 processor 单线程瓶颈）
│   ├── inputs.py              # 训练、评测与服务共用输入预处理
│   ├── calibration.py         # 温度计算、保存与加载
│   ├── serve.py               # 起推理 HTTP 服务
│   └── protocol.json          # 实验协议与超参
├── data/               # 原始 parquet（下载生成）
├── models/             # 底座（下载生成）
├── prepared/           # 切分后的 jsonl（切分生成）
└── runs/               # 训练产物（adapter、报告）
```

## 流程

以下训练和数据准备应在训练机或私有实验工作副本执行。运行前设置 `set -o pipefail`，终端输出使用 `2>&1 | tee <日志文件>` 留存；KB 级证据日志整理后放入 `docs/执行日志/`。

### 1. 下载底座和数据

```bash
python scripts/download_backbone.py --size 2b --mirror
python scripts/download_data.py
```

### 2. 切分数据

```bash
python scripts/prepare_data.py data/all/test-00000-of-00001.parquet
```

产出 `prepared/{train,dev,calibration,test}.jsonl` 和 `prepared/split_manifest.json`。

先将图像统一成 RGB PNG 后计算 sha256，去重后按 seed 与图片哈希排序。约 60% 用于训练、10% 用于 dev、10% 用于 calibration、20% 用于 test，小数据集保证四个划分非空。数量以 manifest 为准；这属于从源 parquet 重新划分的实验集，不是 ScienceQA 官方测试协议。

旧脚本固定配额总计 4800，无法从文档所述的 4241 行源文件中满足，且没有检查空划分。现在不足四张可用独立图片时直接报错。旧 `prepared/` 必须重新生成，训练和评测会拒绝没有 `scienceqa.question_only.v2` 标记的数据。

每行字段：

| 字段 | 含义 |
|---|---|
| `image_png_b64` | 图像 base64 |
| `state` | 仅备选答案与题面提示；不含 `lecture`、`solution`、`answer` |
| `question` | 判别问题 |
| `options` | 候选列表 |
| `gold` | 概率分布，正确项 1.0 其余 0 |

### 3. 训练

```bash
python scripts/train.py \
  --model models/Qwen3.5-2B \
  --data prepared \
  --out runs/corrected \
  --steps 1200 \
  --bs 2
```

训练对每道题计算 soft CE 与 Brier，再按题平均。每步反向传播并更新 LoRA 和评分头，显式启用训练模式。当前保存最后一次更新的 adapter，不做 dev checkpoint 选优，dev 仅用于诊断。温度拟合完成后才读 test。输出目录必须为空，防止覆盖旧结果。

训练前会先读 `/proc/meminfo` 的 `MemAvailable`，低于 `--mem-min-start`（默认 25 GB）直接拒绝启动。训练中每步采样内存，可用量跌破阈值或峰值超过 `--mem-cap`（默认 70 GB）就停训、保存现场、写 `aborted.json` 然后退出，不进入评估。

内存检查在更新前后采样，不能保证阻止单个前向、反向或优化器操作内部的 OOM。Qwen3.5 是 Gated DeltaNet 加门控注意力的混合架构，激活开销模型和普通 transformer 不一样，`gradient_checkpointing_enable()` 默认的 reentrant 实现在这类非线性结构上可能静默失效。脚本里强制传了 `use_reentrant=False`，但如果模型或 batch 再放大，仍然可能把机器拖死，所以显式留一道闸。

### 4. 评测

训练、独立评测和服务现在共用图像缩放与文本构造。独立评测每个划分只跑一次原始 logits，分别计算未校准与校准指标：

```bash
python scripts/evaluate.py \
  --model models/Qwen3.5-2B \
  --ckpt runs/corrected/final \
  --data prepared \
  --out runs/corrected/eval
```

训练与评测均在 calibration 上搜索 0.5～4.5 的 41 点网格，统一使用 `softmax(raw_logits / T)`。最优温度落在边界时发出提示，不宣称已找到无约束最优解。

训练将温度写入 `runs/corrected/final/temperatures.json`；独立评测写入 `--out/temperatures.json`，不会修改原 adapter。使用独立评测结果起服务时显式传 `--temperatures runs/corrected/eval/temperatures.json`。

### 5. 起服务

```bash
python scripts/serve.py --ckpt runs/corrected/final --model models/Qwen3.5-2B --port 8010
```

```bash
curl -X POST http://127.0.0.1:8010/decide -H 'Content-Type: application/json' -d '{
  "image_png_b64": "<base64>",
  "state": "备选答案：\nA. ...\nB. ...\n\n提示：...",
  "question": "图中正在发生什么？请选出正确备选答案。",
  "options": ["A. ...", "B. ..."]
}'
```

默认加载 adapter 目录内的 `temperatures.json`，缺失时直接报错；需要未校准推理时显式传 `--temperature 1`。返回 `probs`、`prediction`、`confidence` 和实际使用的温度。

## 与 text_jev_train 的差异

| 维度 | text_jev_train | vision_jev_train |
|---|---|---|
| 底座 | Qwen3-0.6B | Qwen3.5-2B |
| 数据 | typed-decisions | ScienceQA |
| 微调方式 | 全参 | LoRA r=16 |
| 输入 | 上下文 + 候选文本 | 图像 + 上下文 + 候选文本 |
| 图像占位符 | 无 | `<|vision_start|><|image_pad|><|vision_end|>` |
| 切分主键 | case id | 图片 sha256 |
| 内存管理 | 无 | 启动体检 + 运行硬闸 |

## 已知边界

gold 是选择题标准答案，不是教师模型分布，也不是行动成功率实测。所以这个模型学的是"按图选对答案的能力"，不直接等价于"选对动作的概率"。要做后者，需要把 gold 换成带实测打分的样本，本目录不覆盖。

Qwen3.5 系列 2026 年 2 月起改为统一多模态基座，早期的 Qwen-VL 产品线已停更，不要用 `Qwen2-VL` 或 `Qwen2.5-VL` 的占位符和 processor 写法套这里。
