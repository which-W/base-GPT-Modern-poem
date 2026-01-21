from torch.utils.data import Dataset
from config import *
import json
from tqdm import tqdm
from tokenizers import Tokenizer
import random
#更牛逼的数据集生成
# 数据集
class NalanDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open("all_poems.json", "r", encoding="utf-8") as fp:
            self.raw_ds = json.loads(fp.read())
        
        # 立即构建数据
        self.data = []
        self.build_train_data()

    def build_train_data(self):
        tokenizer = Tokenizer.from_file("tokenizer_hug.json")
        
        #获取特殊token的ID
        pad_id = tokenizer.token_to_id(PAD)
        im_start_id = tokenizer.token_to_id(IM_START)
        im_end_id = tokenizer.token_to_id(IM_END)
        
        print(f"特殊token IDs: PAD={pad_id}, IM_START={im_start_id}, IM_END={im_end_id}")
        print(f"注意: 特殊token解码为空字符串是正常现象，不影响训练\n")
        
        # 定义多样化的通用prompt
        generic_prompts = [
            "请创作一首现代爱情诗",
            "写一首现代诗",
            "来一首现代抽象诗",
        ]
        
        # 统计
        stats = {
            'total': len(self.raw_ds),
            'empty': 0,
            'too_short': 0,
            'too_long': 0,
            'has_garbled': 0,
            'success': 0
        }
        
        print("开始构建数据集...")
        
        for sample in tqdm(self.raw_ds, desc="处理数据"):
            try:
                # 提取诗歌内容
                text = sample['para'][0].strip() if sample.get('para') and len(sample['para']) > 0 else ""
                
                if not text:
                    stats['empty'] += 1
                    continue
                
                # 过滤太短的内容
                if len(text) < 10:
                    stats['too_short'] += 1
                    continue
                
                # 过滤乱码
                if '�' in text:
                    stats['has_garbled'] += 1
                    continue
                
                # 随机选择训练格式（增加数据多样性）
                rand = random.random()
                
                if rand < 0.6:  # 60%: 只用标题
                    title = sample.get('title', '').strip()
                    if not title:
                        title = random.choice(generic_prompts)
                    inputs = f"{IM_START}user\n{title}\n{IM_END}\n{IM_START}assistant\n{text}{IM_END}"
                
                elif rand < 0.9:  # 30%: 标题+作者
                    title = sample.get('title', '').strip()
                    author = sample.get('author', '佚名')
                    if not title:
                        title = random.choice(generic_prompts)
                    inputs = f"{IM_START}user\n{title}({author})\n{IM_END}\n{IM_START}assistant\n{text}{IM_END}"
                
                else:  # 10%: 通用prompt
                    user_input = random.choice(generic_prompts)
                    inputs = f"{IM_START}user\n{user_input}\n{IM_END}\n{IM_START}assistant\n{text}{IM_END}"
                
                # 编码
                encoded = tokenizer.encode(inputs)
                ids = encoded.ids

                # 检查长度
                if len(ids) > MAX_SEQ_LEN:
                    stats['too_long'] += 1
                    continue
                
                # 关键修复: 只验证诗歌内容，不验证特殊token
                # 提取诗歌内容部分（assistant后面的文本）
                try:
                    # 找到assistant部分的内容
                    parts = inputs.split(f"{IM_START}assistant\n")
                    if len(parts) >= 2:
                        content_part = parts[1].replace(f"{IM_END}", "").strip()
                        
                        # 只对内容部分编解码测试
                        test_encoded = tokenizer.encode(content_part)
                        test_decoded = tokenizer.decode(test_encoded.ids)
                        
                        # 只要诗歌内容能正确编解码就OK
                        if content_part != test_decoded:
                            stats['has_garbled'] += 1
                            if stats['has_garbled'] <= 3:
                                print(f"\n内容编解码不一致:")
                                print(f"  原文: {content_part[:100]}")
                                print(f"  解码: {test_decoded[:100]}")
                            continue
                except Exception as e:
                    # 如果提取失败，跳过验证，直接接受
                    pass
                
                # 验证特殊token ID是否存在
                if im_start_id not in ids or im_end_id not in ids:
                    if stats['has_garbled'] <= 3:
                        print(f"\n警告: 样本缺少特殊token ID")
                    continue
                
                # 打印前3个样本
                if len(self.data) < 3:
                    print("\n" + "="*60)
                    print(f"样本 {len(self.data) + 1}:")
                    print(f"ChatML原文:")
                    print(repr(inputs[:200]))
                    print(f"\nToken长度: {len(ids)}")
                    print(f"Token IDs前20个: {ids[:20]}")
                    print(f"包含IM_START(ID={im_start_id}): {im_start_id in ids}")
                    print(f"包含IM_END(ID={im_end_id}): {im_end_id in ids}")
                    print("="*60)
                
                self.data.append((ids, inputs))
                stats['success'] += 1

            except Exception as e:
                if stats['has_garbled'] <= 3:
                    print(f"\nError processing sample: {e}")
                    import traceback
                    traceback.print_exc()
                stats['has_garbled'] += 1
                continue
        
        # 打印统计
        print("\n" + "="*80)
        print("数据集构建统计:")
        print("="*80)
        print(f"总样本数:     {stats['total']:6,}")
        print(f"  成功:    {stats['success']:6,} ({stats['success']/stats['total']*100:5.1f}%)")
        print(f"  空内容:     {stats['empty']:6,}")
        print(f"  太短:       {stats['too_short']:6,}")
        print(f"  太长:       {stats['too_long']:6,}")
        print(f"  乱码/错误:  {stats['has_garbled']:6,}")
        print("="*80 + "\n")
        
        if stats['success'] == 0:
            raise ValueError("没有有效样本！请检查tokenizer和数据")
        
        if stats['success'] < stats['total'] * 0.5:
            print("警告: 有效样本少于50%，请检查数据质量")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_dataset():
    return NalanDataset()