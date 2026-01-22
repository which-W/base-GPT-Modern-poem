# [base-GPT-Modern-poem]

<div align="center">
*[项目简短描述 - 基于现代诗的基础GPT]*
</div>

## 🎯 项目简介
*[详细介绍你的项目是什么，做什么用，解决什么问题]*

这是一个从头实现的 GPT 语言模型训练框架，支持：

- 自定义 Transformer 架构
- BPE 分词器训练与使用
- 单卡/分布式训练
- 对话生成推理

## ✨ 功能特性

| 特性 | 状态 | 描述 |
|------|------|------|
| 自定义 GPT 模型 | ✅ | 基于 PyTorch Transformer 实现 |
| BPE 分词器 | ✅ | 支持从零训练分词器 |
| 分布式训练 | ✅ | 支持 DDP 多卡训练 |
| 对话生成 | ✅ | 支持交互式对话 |
| 模型断点续训 | ✅ | 支持训练中断恢复 |
| 模型量化推理 | ⏳ | 后续规划 |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.1+
- CUDA 12.1+ (GPU 训练)
- 8GB+ RAM
-本人自己的机子为RTX3060 8G

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备

1. 将训练数据放入指定目录
2. 运行数据预处理脚本：

```bash
python data_clean.py
```

#### 数据格式

训练数据 `all_poems.json` 的 JSON 格式如下：

```json
[
  {
     "title": "xxxxxx",
    "para": [
      "xxxxxxxx/nxxxxxxx"
    ],
    "author": "xxx"
  },
  }
]
```

| 字段 | 类型 | 描述 |
|------|------|------|
| `title` | string | 诗歌标题 |
| `para` | array | 诗歌内容段落数组，每段用 `\n` 分隔 |
| `author` | string | 作者姓名 |

### datasets准备
1，运行train_dataset_one.py(基础)而xx_all.py为强化版本
2，生成dataset.bin文件
```bash
python train_dataset_one.py
```

### tokennizer准备
1，运行hug_face_tokenizer.py
2，生成tokennizer_hug.json文件
```bash
python hug_face_tokenizer.py
```

### 开始训练

```bash
# 单卡训练
python train_gpt.py

# 分布式训练 (多卡)
python -m torch.distributed.launch --nproc_per_node=8 train_gpt_ddp.py
```

### 模型推理

```bash
python infere.py
```

## 📁 项目结构

```
base-ChatGPT/
├── 📄 README.md                 # 项目说明文档
├── 📄 config.py                 # 配置文件
├── 📄 requirements.txt          # 依赖列表
│
├── 🤖 模型定义
│   ├── 📄 gpt.py               # GPT 模型实现
│   ├── 📄 embeding.py          # 词嵌入层
│   └── 📄 tokenizer.py         # BPE 分词器
│
├── 📊 数据处理
│   ├── 📄 dataset.py           # 数据集加载
│   ├── 📄 data_clean.py        # 数据清洗
│   └── 📄 train_BPEtokennizer.py  # 分词器训练
│
├── 🏋️ 训练脚本
│   ├── 📄 train_gpt.py         # 单卡训练
│   └── 📄 train_gpt_ddp.py     # 分布式训练
│
├── 💬 推理脚本
│   └── 📄 infere.py            # 对话生成
│
└── 📦 模型文件
    ├── 📄 tokenizer_hug.json   # 分词器文件
    .......
```

## ⚙️ 配置说明

主要配置参数在 `config.py` 中：

```python
# 模型配置
VOCAB_SIZE = 20000      # 词表大小
MAX_SEQ_LEN = 512       # 最大序列长度
GPT_DIM = 256           # 模型维度
GPT_HEAD = 8            # 注意力头数
GPT_FF = 1024           # 前馈网络维度
GPT_BLOCKS = 6          # Transformer 层数

# 训练配置
TRAIN_ITER = 50000      # 训练迭代次数
BATCH_SIZE = 8          # 批次大小
LEARNING_RATE = 0.0001  # 学习率
WARMUP_STEPS = 500      # Warmup 步数
ACCUM_STEPS = 8         # 梯度累积步数

# 推理配置
TEMPERATURE = 0.7       # 温度参数
TOP_K = 20              # Top-K 采样
REPETITION_PENALTY = 1.4  # 重复惩罚
```

## 📖 训练模型

### 单卡训练

```bash
python train_gpt.py
```

训练过程会：
1. 自动加载数据
2. 每 1000 步保存检查点
3. 每 1000 步生成示例文本
4. 自动记录最佳模型

### 分布式训练 (DDP)

```bash
# 8卡训练示例
python -m torch.distributed.launch --nproc_per_node=8 train_gpt_ddp.py
```

### 从检查点恢复训练

训练会自动从 `checkpoint.bin` 恢复：

```bash
python train_gpt.py
```

## 💬 模型推理

### 交互式对话

```bash
python infere.py
```

示例交互：

```
Chat with GPT (type 'exit' to quit)
输入：一只鸟在想方向
回答："亟欲飞翔却张不开翅膀
张开了翅膀又遗忘了飞翔
展翅飞翔又失掉了方向
鸟在想 反覆的想
是鸟该栖息在岛身上
还是岛该栖息在鸟的翅膀上
或是两者该缱绻而孕而产
而生命而历史
鸟演算方向 以
鸟与岛 岛与翅膀 翅膀与冥苍
交互演绎 归纳 重覆交互
豁然发现 辽阔就是正确方向"
```

### API 调用

```python
from infere import chat

response = chat("你的问题")
print(response)
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request


<div align="center">

⭐ 如果这个项目对你有帮助，欢迎 Star 支持！

</div>
