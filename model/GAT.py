import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


class GAT(torch.nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channels=64, num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GATConv(in_channel, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels * heads))

        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channel, heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(out_channel))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
