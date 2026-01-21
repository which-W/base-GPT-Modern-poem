# diagnose.py - 完整诊断脚本
import json
import torch
from tokenizers import Tokenizer
from collections import Counter
from config import *
from dataset_train_one import load_dataset
from tqdm import tqdm

print("="*80)
print("诗歌GPT训练问题诊断工具")
print("="*80)

#第一部分：检查Tokenizer
print("\n【1】检查Tokenizer质量")
print("-"*80)

try:
    tokenizer = Tokenizer.from_file("tokenizer_hug.json")
    print(f"成功加载tokenizer")
    print(f"   词表大小: {tokenizer.get_vocab_size()}")
except Exception as e:
    print(f"加载tokenizer失败: {e}")
    exit(1)

# 测试编解码
test_texts = [
    "床前明月光，疑是地上霜。",
    "举头望明月，低头思故乡。",
    "我",
    "的",
    "一只鸟在想方向",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"
]

print("\n编解码测试:")
decode_errors = 0
for text in test_texts:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)
    match = text == decoded
    
    print(f"\n原文: {text}")
    print(f"Token数: {len(encoded.ids)}")
    print(f"前10个tokens: {encoded.tokens[:10]}")
    print(f"解码: {decoded}")
    print(f"匹配: {'ok' if match else '不匹配!'}")
    
    if not match:
        decode_errors += 1

if decode_errors > 0:
    print(f"\n发现 {decode_errors} 个解码错误! Tokenizer需要重新训练!")
else:
    print(f"\n所有测试通过，tokenizer编解码正常")

# 检查特殊token
print("\n特殊Token检查:")
try:
    pad_encoded = tokenizer.encode(PAD)
    im_start_encoded = tokenizer.encode(IM_START)
    im_end_encoded = tokenizer.encode(IM_END)
    
    print(f"PAD '{PAD}': {pad_encoded.ids} - tokens: {pad_encoded.tokens}")
    print(f"IM_START '{IM_START}': {im_start_encoded.ids} - tokens: {im_start_encoded.tokens}")
    print(f"IM_END '{IM_END}': {im_end_encoded.ids} - tokens: {im_end_encoded.tokens}")
    
    # 检查是否为单个token
    if len(pad_encoded.ids) != 1:
        print(f"PAD应该是单个token!")
    if len(im_start_encoded.ids) != 1:
        print(f"IM_START应该是单个token!")
    if len(im_end_encoded.ids) != 1:
        print(f"IM_END应该是单个token!")
        
except Exception as e:
    print(f"特殊token检查失败: {e}")

#第二部分：检查原始数据
print("\n" + "="*80)
print("【2】检查原始诗歌数据")
print("-"*80)

try:
    with open('all_poems.json', 'r', encoding='utf-8') as f:
        raw_poems = json.load(f)
    
    print(f" 成功加载诗歌数据")
    print(f"  总诗歌数: {len(raw_poems)}")
    
    # 统计信息
    has_title = sum(1 for p in raw_poems if p.get('title'))
    has_para = sum(1 for p in raw_poems if p.get('para') and len(p['para']) > 0)
    has_author = sum(1 for p in raw_poems if p.get('author'))
    
    print(f"   有标题: {has_title} ({has_title/len(raw_poems)*100:.1f}%)")
    print(f"   有内容: {has_para} ({has_para/len(raw_poems)*100:.1f}%)")
    print(f"   有作者: {has_author} ({has_author/len(raw_poems)*100:.1f}%)")
    
    # 检查前5首诗的数据质量
    print(f"\n前5首诗样本:")
    for i in range(min(5, len(raw_poems))):
        poem = raw_poems[i]
        title = poem.get('title', '无标题')
        para = poem.get('para', [])
        author = poem.get('author', '佚名')
        
        para_text = para[0] if para else '无内容'
        
        print(f"\n诗歌 {i+1}:")
        print(f"  标题: {title}")
        print(f"  作者: {author}")
        print(f"  内容长度: {len(para_text)} 字符")
        print(f"  内容预览: {para_text[:50]}...")
        
        # 检查异常字符
        if '�' in para_text:
            print(f"包含乱码字符!")
        
except Exception as e:
    print(f" 加载原始数据失败: {e}")

#第三部分：检查处理后的数据集
print("\n" + "="*80)
print("【3】检查处理后的训练数据")
print("-"*80)

try:
    dataset = load_dataset()
    print(f" 成功加载训练数据集")
    print(f" 样本数: {len(dataset)}")
    
    # 检查前10个样本
    print(f"\n前10个训练样本详情:")
    for i in range(min(10, len(dataset))):
        ids, chatml = dataset[i]
        
        print(f"\n样本 {i+1}:")
        print(f"  ChatML格式 (前200字符):")
        print(f"  {repr(chatml[:200])}")
        print(f"  Token长度: {len(ids)}")
        
        # 解码检查
        decoded = tokenizer.decode(ids)
        match = chatml == decoded
        
        if not match:
            print(f"  编码后解码不匹配!")
            print(f"  原始前100字符: {repr(chatml[:100])}")
            print(f"  解码前100字符: {repr(decoded[:100])}")
        else:
            print(f"  编解码一致")
        
        # 检查特殊token
        if IM_START in chatml and IM_END in chatml:
            print(f"  包含特殊token")
        else:
            print(f"  缺少特殊token!")
            
except Exception as e:
    print(f" 加载数据集失败: {e}")
    import traceback
    traceback.print_exc()

#第四部分：统计Token分布
print("\n" + "="*80)
print("【4】统计Token分布")
print("-"*80)

try:
    dataset = load_dataset()
    
    print("正在统计token分布...")
    all_tokens = []
    seq_lengths = []
    
    for ids, _ in tqdm(dataset, desc="统计中"):
        all_tokens.extend(ids)
        seq_lengths.append(len(ids))
    
    counter = Counter(all_tokens)
    
    print(f"\n总token数: {len(all_tokens):,}")
    print(f"唯一token数: {len(counter):,}")
    print(f"词表使用率: {len(counter)/tokenizer.get_vocab_size()*100:.1f}%")
    
    print(f"\n序列长度统计:")
    print(f"  平均长度: {sum(seq_lengths)/len(seq_lengths):.1f}")
    print(f"  最短: {min(seq_lengths)}")
    print(f"  最长: {max(seq_lengths)}")
    print(f"  中位数: {sorted(seq_lengths)[len(seq_lengths)//2]}")
    
    print(f"\n最高频20个token:")
    for token_id, count in counter.most_common(20):
        try:
            token_text = tokenizer.decode([token_id])
            freq = count / len(all_tokens) * 100
            print(f"  ID {token_id:5d}: {count:8,} 次 ({freq:5.2f}%) - '{token_text}'")
        except:
            print(f"  ID {token_id:5d}: {count:8,} 次 - [解码失败]")
    
    # 检查PAD token占比
    pad_id = tokenizer.encode(PAD).ids[0]
    if pad_id in counter:
        pad_ratio = counter[pad_id] / len(all_tokens) * 100
        print(f"\nPAD token占比: {pad_ratio:.2f}%")
        if pad_ratio > 30:
            print(f"  PAD占比过高，可能影响训练!")
    
    # 检查是否有某个token占比过高
    most_common_id, most_common_count = counter.most_common(1)[0]
    if most_common_count / len(all_tokens) > 0.1:
        token_text = tokenizer.decode([most_common_id])
        print(f"\n  最高频token '{token_text}' 占比超过10%，可能有问题!")
        
except Exception as e:
    print(f" 统计失败: {e}")
    import traceback
    traceback.print_exc()

#诊断总结
print("\n" + "="*80)
print("【诊断总结】")
print("="*80)
print("\n请检查以上输出，重点关注:")
print("1.  标记的错误和警告")
print("2. Tokenizer编解码是否一致")
print("3. 训练数据是否包含正确的特殊token")
print("4. Token分布是否合理（没有某个token占比过高）")
print("="*80)