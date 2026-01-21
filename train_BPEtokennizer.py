from tokenizer import BPETokenizer
from config import *
import os
import sys 
import json 

# 检查 tokenizer 是否已存在
if os.path.exists('tokenizer.bin'):
    print('tokenizer.bin 已存在，跳过训练')
    sys.exit(0)

# 加载多作者诗歌数据（新格式）
INPUT_JSON = 'all_poems.json' 
with open(INPUT_JSON, 'r', encoding='utf-8') as fp:
    ds = json.load(fp)  

print(f'共加载 {len(ds)} 首诗歌')

# 提取所有文本（标题 + 段落）
text_list = []
for sample in ds:
    text_list.append(sample['title'])
    text_list.extend(sample['para'])  

# 训练 BPE 分词器
print(f'开始训练 BPE 词表 (vocab_size={VOCAB_SIZE})...')
tokenizer = BPETokenizer()  
tokenizer.train(text_list, VOCAB_SIZE)
tokenizer.add_special_tokens([IM_START, IM_END, BOS, EOS, PAD])
tokenizer.save('tokenizer.bin')

print('✅ tokenizer.bin 生成成功！')