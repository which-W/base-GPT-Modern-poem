import torch
from torch import nn
import math
from config import *
class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, dim, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len  # 仅用于初始化 buffer（可选）
        self.seq_emb = nn.Embedding(vocab_size, dim)

        # 可选：预注册一个足够大的 pos_encoding（如5000），避免重复计算
        self.register_buffer(
            'pos_encoding',
            self._build_position_encoding(max_seq_len),
            persistent=False  # 不保存到 state_dict（节省 checkpoint 大小）
        )

    def _build_position_encoding(self, seq_len):
        """生成长度为 seq_len 的位置编码"""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float) * 
            (-math.log(10000.0) / self.dim)
        )  # (dim//2,)
        pos_enc = torch.zeros(seq_len, self.dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)   # 偶数位
        pos_enc[:, 1::2] = torch.cos(position * div_term)   # 奇数位
        return pos_enc

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        seq_len = x.size(1)
        
        # 关键：如果输入超过预计算长度，动态扩展
        if seq_len > self.pos_encoding.size(0):
            # 动态生成更长的位置编码（只生成超出部分）
            new_pos_enc = self._build_position_encoding(seq_len)
            self.register_buffer('pos_encoding', new_pos_enc, persistent=False)
        
        x = self.seq_emb(x)  # (B, L, dim)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)  # 广播到 batch
        return x