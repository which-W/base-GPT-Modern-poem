VOCAB_SIZE=20000    # 词表大小
MAX_SEQ_LEN=512     # GPT模型输入限制 输入加输出的总长度

# transformer
GPT_DIM = 512      # embedding dimension
GPT_HEAD = 8       # attention heads
GPT_FF = 2048      # feedforward dimension (通常是 dim 的 4倍)
GPT_BLOCKS = 8     # transformer blocks

# training - 优化配置以解决loss停滞问题
TRAIN_ITER=10000   # 增加训练步数用于fine-tune
BATCH_SIZE=32
LEARNING_RATE=2e-5  # 降低学习率: 2e-4 -> 5e-5
WARMUP_STEPS=1000    # 添加warmup stabilize训练
# inference
TEMPERATURE=0.4
TOP_K=10
REPETITION_PENALTY = 1.4
# special tokens
PAD='<|padding|>'
IM_START='<|im_start|>'
IM_END='<|im_end|>'

#DDP
ACCUM_STEPS =8
# chat or generate
GPT_MODE='chat'