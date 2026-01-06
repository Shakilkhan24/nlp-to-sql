# sequence to sequence task 
# that's why i used encoder-decoder transformer model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.pow(10000.0, i / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term) 
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def into_multiheads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.into_multiheads(self.w_q(query))
        K = self.into_multiheads(self.w_k(key))
        V = self.into_multiheads(self.w_v(value))
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).to(x.device)
        mask = causal_mask if mask is None else (mask & causal_mask)
        return self.attention(x, x, x, mask)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class AddAndNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.multi_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)

    def forward(self, x, mask=None):
        attn_output = self.multi_attention(x, x, x, mask)
        x = self.add_norm1(x, attn_output)
        ff_output = self.feed_forward(x)
        return self.add_norm2(x, ff_output)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_multi_attention = MaskedMultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)
        self.add_norm3 = AddAndNorm(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.add_norm1(x, self.masked_multi_attention(x, tgt_mask))
        x = self.add_norm2(x, self.cross_attention(x, encoder_output, encoder_output, src_mask))
        return self.add_norm3(x, self.feed_forward(x))

class OutputLinear(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)

class OutputSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):  
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.output_linear = OutputLinear(d_model, vocab_size)

    def make_src_mask(self, src_tokens, pad_idx=0):
        # src_tokens: (batch, src_len)
        mask = (src_tokens != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,src_len)
        return mask  # broadcastable to attention scores

    def make_tgt_mask(self, tgt_tokens, pad_idx=0):
        # tgt_tokens: (batch, tgt_len)
        seq_len = tgt_tokens.size(1)
        padding_mask = (tgt_tokens != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,tgt_len)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=tgt_tokens.device)).unsqueeze(0).unsqueeze(0)
        return padding_mask & causal_mask  # (batch,1,tgt_len,tgt_len)

    def forward(self, src_tokens, tgt_tokens, src_pad_idx=0, tgt_pad_idx=0):
        src_mask = self.make_src_mask(src_tokens, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt_tokens, tgt_pad_idx)

        src_emb = self.positional(self.embedding(src_tokens) * math.sqrt(self.embedding.d_model))
        enc_out = self.encoder(src_emb, src_mask)

        tgt_emb = self.positional(self.embedding(tgt_tokens) * math.sqrt(self.embedding.d_model))
        dec_out = self.decoder(tgt_emb, enc_out, src_mask, tgt_mask)

        logits = self.output_linear(dec_out)
        return logits

if __name__ == "__main__":
    print('EVERY THING IS OK')
