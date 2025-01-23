import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

class TemporalGraphNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalGraphNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 使用 GATConv 替代 GCNConv
        self.graph_conv1 = GATConv(in_channels, 128, heads=4, concat=True)
        self.graph_conv2 = GATConv(128 * 4, 128, heads=4, concat=True)
        self.graph_conv3 = GATConv(128 * 4, out_channels, heads=4, concat=False)

        # 更复杂的时间特征处理
        self.temporal_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

        # 使用跳跃连接和层归一化
        self.fc1 = nn.Linear(out_channels * 2, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.5)

        # 跳跃连接调整
        self.residual_transform = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_time):
        edge_time = edge_time.unsqueeze(-1).float()  # 确保形状为 (num_edges, 1)

        # 时间编码
        time_emb = self.temporal_embedding(edge_time).squeeze(1)  # 输出形状应为 (num_edges, out_channels)

        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 图卷积
        x_res = x  # 用于跳跃连接
        x_res = self.residual_transform(x_res)  # 调整维度
        x = self.graph_conv1(x, edge_index)
        x = F.relu(x)
        x = self.graph_conv2(x, edge_index)
        x = F.relu(x)
        x = self.graph_conv3(x, edge_index)
        x = F.relu(x)

        # 聚合
        row, col = edge_index
        x_aggregated = torch.zeros((x.size(0), x.size(1)), device=x.device)
        x_aggregated = x_aggregated.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(1)), x[col])

        # 时间特征聚合
        time_emb_mean = time_emb.mean(dim=0)
        time_emb_mean = time_emb_mean.unsqueeze(0).expand(x_aggregated.size(0), -1)

        # 融合图特征和时间特征
        x_aggregated = torch.cat([x_aggregated, time_emb_mean], dim=1)

        # 使用全连接层融合特征
        x = self.fc1(x_aggregated)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm(x)

        # 跳跃连接
        x = x + x_res

        return x
