import torch
import torch.nn as nn

# Multi-Head Self-Attention mechanism definition


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        # in_channels 是输入的特征维度，out_channels 是投影后的维度
        # num_heads 是注意力头的数量
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(in_channels, out_channels)  # 最后输出调整维度

    def forward(self, x):
        # 这里 x 的维度是 (batch_size, seq_len, in_channels)
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        output = self.fc(attn_output)  # 调整输出的维度
        return output


class MultiHeadSelfAttentionModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, num_heads=4):
        super(MultiHeadSelfAttentionModel, self).__init__()
        # 使用多头自注意力机制
        self.multihead_self_attention = MultiHeadSelfAttention(in_channels, out_channels, num_heads)
        self.fc = nn.Linear(out_channels, num_classes)  # 最后输出类别

    def forward(self, x):
        attention_output = self.multihead_self_attention(x)
        output = self.fc(attention_output.mean(dim=1))  # 使用 mean(dim=1) 来聚合序列维度
        return output

