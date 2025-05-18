import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

# consider using https://pytorch-geometric.readthedocs.io/
# Too many choices, we just randomly pick a one GATv2Conv

class CommBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, heads=8):
        super().__init__()
        
        assert hid_dim%heads==0 and out_dim%heads==0
        
        _hid_dim=hid_dim//heads
        _out_dim=out_dim//heads
        
        self.atten_conv1=gnn.GATv2Conv(
            in_channels=in_dim,
            out_channels=_hid_dim,
            heads=heads,
            add_self_loops=False
        )
        self.act1=nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(hid_dim)
        )
        self.atten_conv2=gnn.GATv2Conv(
            in_channels=hid_dim,
            out_channels=_out_dim,
            heads=heads,
            add_self_loops=False
        )
        self.act2=nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(hid_dim)
        )
        
    def forward(self, x, edge_idx):
        '''
        x: B*A, F
        edge_idx: B*A*Neighbors, 2
        '''
        
        h = x
        h = self.atten_conv1(h, edge_idx)
        h = self.act1(h)
        h = self.atten_conv2(h, edge_idx)
        h = self.act2(h)
        
        return h
        