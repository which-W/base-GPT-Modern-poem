from torch import nn
import torch 
from embeding import EmbeddingWithPosition
from config import GPT_BLOCKS, MAX_SEQ_LEN
import math

class GPT(nn.Module):
    def __init__(self, d_model, nhead, feedforward, vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model  # 保存d_model用于缩放
        
        # Token + Positional Embedding
        self.emb = EmbeddingWithPosition(vocab_size=vocab_size, dim=d_model, max_seq_len=max_seq_len)
        # loss下不去的时候可以不做
        self.dropout = nn.Dropout(dropout)
        
        # 使用 TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward,
            batch_first=True,
            norm_first=True,
            activation='gelu',
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=GPT_BLOCKS)
        
        
        self.prob_linear = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定（embedding 和 output 共享）
        self.prob_linear.weight = self.emb.seq_emb.weight
        
        # 添加缩放因子
        # 由于使用了权重绑定，需要缩放logits以避免数值爆炸
        # 使用稍小的缩放因子以获得更接近理论的初始loss
        self.output_scale = 0.7 / math.sqrt(d_model)  

    def forward(self, x, padding_mask=None):
        # 注意力遮挡
        src_mask = torch.triu(
            torch.ones(x.size()[1], x.size()[1], dtype=torch.bool, device=x.device), 
            diagonal=1
        )
        
        # embedding
        x = self.emb(x)
        
        # transformer
        x = self.transformer(
            src=x,
            mask=src_mask,                      # causal mask
            src_key_padding_mask=padding_mask   # padding mask
        )
        
        
        logits = self.prob_linear(x)
        logits = logits * self.output_scale  # 缩放logits
        
        return logits


if __name__ == '__main__':
    from tokenizers import Tokenizer
    
    tokenizer = Tokenizer.from_file("tokenizer_hug.json")
    
    # 模拟输入
    x = torch.randint(0, tokenizer.get_vocab_size(), (5, 30))
    padding = torch.zeros(5, 30, dtype=torch.bool)
    
    # GPT模型
    from config import MAX_SEQ_LEN
    gpt = GPT(
        d_model=64, 
        nhead=2, 
        feedforward=128, 
        vocab_size=tokenizer.get_vocab_size(), 
        max_seq_len=MAX_SEQ_LEN
    )
    
    y = gpt(x, padding)
    print(f"Output shape: {y.shape}")
    print(f"Logits range: [{y.min().item():.2f}, {y.max().item():.2f}]")
    print(f"Logits std: {y.std().item():.2f}")
    
    # 测试loss
    import torch.nn.functional as F
    loss = F.cross_entropy(
        y[:, :-1].reshape(-1, tokenizer.get_vocab_size()),
        x[:, 1:].reshape(-1)
    )
    print(f"Test loss: {loss.item():.4f} (should be ~9.6 for random)")