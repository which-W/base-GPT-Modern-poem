from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from config import *
import os
import sys
import json

# 检查 tokenizer 是否已存在
if os.path.exists('tokenizer.json'):
    print('tokenizer.json 已存在，跳过训练')
    sys.exit(0)

# 加载多作者诗歌数据
INPUT_JSON = 'all_poems.json'
with open(INPUT_JSON, 'r', encoding='utf-8') as fp:
    ds = json.load(fp)

print(f'共加载 {len(ds)} 首诗歌')

# 提取所有文本（标题 + 段落 + 作者）
text_list = []
for sample in ds:
    text_list.append(sample['title'])
    text_list.extend(sample['para'])
    text_list.append(sample["author"])

print(f'共提取 {len(text_list)} 段文本用于训练')

# === 使用 Hugging Face tokenizers 训练 BPE ===

# 1. 初始化 tokenizer（Byte-level BPE）
tokenizer = Tokenizer(models.BPE())

# 2. 设置 pre-tokenizer 和 decoder（支持中文、英文、符号等）
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 中文不需要前缀空格
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True) #防止Byte-Level 编码引入的多余空格或伪字符边界，影响token与id对

# 3. 定义特殊 tokens（必须是字符串）
special_tokens = [PAD, IM_START, IM_END]  # 来自 config.py

# 4. 创建 trainer
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=special_tokens,  # 自动添加到词表开头
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True
)

# 5. 训练！
print(f'开始训练 Byte-level BPE tokenizer (vocab_size={VOCAB_SIZE})...')
tokenizer.train_from_iterator(text_list, trainer=trainer)

# 6. 保存为 JSON（推荐）或 .bin
tokenizer.save("tokenizer_hug.json", pretty=True)
print('tokenizer_hug.json 生成成功！')

# （可选）如果你坚持要 .bin，可以用 pickle，但 JSON 更标准
# import pickle
# with open('tokenizer.bin', 'wb') as f:
#     pickle.dump(tokenizer, f)