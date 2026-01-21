from gpt import GPT
from config import *
import torch 
from tokenizer import BPETokenizer
import torch.nn.functional as F
import random
from tokenizers import Tokenizer
# 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

# 分词器
# tokenizer = BPETokenizer()  
# tokenizer.load('tokenizer.bin')

tokenizer = Tokenizer.from_file("tokenizer_hug.json")


# 特殊 token ID（提前转为整数）
pad_id = tokenizer.encode(PAD)
im_end_id = tokenizer.encode(IM_END)
STOP_IDS = set()
STOP_IDS.add(pad_id.ids[0])      # 1504
STOP_IDS.add(im_end_id.ids[0])   # 1501

# 加载模型
model = GPT(
    d_model=GPT_DIM,
    nhead=GPT_HEAD,
    feedforward=GPT_FF,
    vocab_size=tokenizer.get_vocab_size(),
    #vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=MAX_SEQ_LEN
).to(DEVICE)

try:  
    checkpoint = torch.load('final_checkpoint.bin', weights_only=True, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    print(f"模型加载成功")
    print(f"训练步数: {checkpoint.get('step', 'unknown')}")
    print(f"训练损失: {checkpoint.get('loss', 'unknown')}")
except Exception as e:
    print(f"模型加载失败: {e}")


print("model device:", next(model.parameters()).device)
model.eval()
def chat(query):
    global tokenizer, model

    # 构造输入
    inputs = f'{IM_START}user\n{query}\n{IM_END}\n{IM_START}assistant\n' if GPT_MODE == 'chat' else f'{query}'
    encoded= tokenizer.encode(inputs)
    ids = encoded.ids
    
    # 记录输入的长度，用于后续截取
    input_length = len(ids)
    
    while len(ids) < MAX_SEQ_LEN:
        batch_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        # batch_paddding_mask = torch.tensor([[0] * len(ids)], dtype=torch.bool).to(DEVICE)

        with torch.no_grad():
            logits = model(batch_ids, padding_mask=None)  # (batch, seq, vocab)
            logits = logits.logits if hasattr(logits, 'logits') else logits

            # 取最后一个 token 的 logits，并除以温度
            next_token_logits = logits[0, -1, :]   # shape: [vocab_size]
            
             # 重复惩罚：降低已生成 token 的概率
            for token_id in set(ids):
                if next_token_logits[token_id] < 0:
                    next_token_logits[token_id] *= REPETITION_PENALTY
                else:
                    next_token_logits[token_id] /= REPETITION_PENALTY
            next_token_logits = next_token_logits / TEMPERATURE
            
            # 在 GPU 上做 top-k
            topk_logits, topk_ids = torch.topk(next_token_logits, k=TOP_K, dim=-1)

            topk_logits_cpu = topk_logits.cpu()
            topk_ids_cpu = topk_ids.cpu()

            # 在 CPU 上采样
            topk_probs = F.softmax(topk_logits_cpu, dim=-1)
            rnd = random.random()
            cumsum = 0.0
            next_id = topk_ids_cpu[0].item()  # 默认选第一个（兜底）
            for i in range(TOP_K):
                prob = topk_probs[i].item()
                if rnd < cumsum + prob:
                    next_id = topk_ids_cpu[i].item()
                    break
                cumsum += prob

        # 终止判断
        if next_id in STOP_IDS:
            break

        ids = ids + [next_id]
    return tokenizer.decode(ids[input_length:])
if __name__ == '__main__':
    print("Chat with GPT (type 'exit' to quit)")
    while True:
        query=input('输入：')
        if query=='exit':
            break
        
        resp=chat(query)
        print('回答：',resp)
    
    