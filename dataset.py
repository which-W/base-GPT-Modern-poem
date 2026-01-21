from torch.utils.data import Dataset
from tokenizer import BPETokenizer
from config import *
import json
from tqdm import tqdm
from tokenizers import Tokenizer
import random
# 数据集
class NalanDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open("all_poems.json", "r", encoding="utf-8") as fp:
            self.raw_ds = json.loads(fp.read())

    def build_train_data(self):
        #自己的tokenizer
        # tokenizer = BPETokenizer()
        # tokenizer.load("tokenizer.bin")
        tokenizer = Tokenizer.from_file("tokenizer_hug.json")
        self.data = []
         # 定义多样化的通用prompt(如果没有标题)
        generic_prompts = [
            "请创作一首爱情诗",
            "写一首现代诗",
            "来一首友情现代诗",
            # ... 增加多样性
        ]
        
        for sample in tqdm(self.raw_ds, desc="building dataset"):
            try:
                text = sample['para'][0].strip() if sample.get('para') and len(sample['para']) > 0 else ""
                if not text:
                    continue
                
                 # 随机选择训练格式（增加数据多样性）
                format_type = random.choice(['title_only', 'with_author', 'generic'])
                
                if format_type == 'title_only':
                    # 60%: 只用标题
                    inputs = f"{IM_START}user\n{sample['title']}\n{IM_END}\n{IM_START}assistant\n{text}{IM_END}"
                
                elif format_type == 'with_author':
                    # 30%: 标题+作者
                    author = sample.get('author', '佚名')
                    inputs = f"{IM_START}user\n{sample['title']}({author})\n{IM_END}\n{IM_START}assistant\n{text}{IM_END}"
                
                else:
                    # 10%: 通用prompt
                    user_input = random.choice(generic_prompts)
                    inputs = f"{IM_START}user\n{user_input}\n{IM_END}\n{IM_START}assistant\n{text}{IM_END}"
                encoded = tokenizer.encode(inputs)
                ids = encoded.ids

                if len(ids) > MAX_SEQ_LEN:
                    continue
                if len(self.data) < 3:  # 打印前3个样本
                    print("Sample input:")
                    print(repr(inputs))  # 查看是否含 \n 和 im_end
                    print("Decoded back:")
                    print(tokenizer.decode(ids))
                    print("-" * 50)
                self.data.append((ids, inputs))

            except Exception as e:
                # 可选：打印错误样本用于调试
                print(f"Error processing sample: {e}")
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
