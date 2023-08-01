import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv

from ..PyramidPooling import PyramidPooling


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 aggregator_type,
                 readout,
                 pyramid):
        super(GraphSAGE, self).__init__()
        self.readout = readout
        self.pyramid = pyramid
        self.conv1 = SAGEConv(in_feats, n_hidden, aggregator_type, activation=F.relu)
        self.conv2 = SAGEConv(n_hidden, n_hidden, aggregator_type, activation=F.relu)
        if self.readout != 'pplp':
            self.input_layer = nn.Linear(n_hidden, 512)
        else:
            self.nfpp_layer = PyramidPooling(self.pyramid)
            self.input_layer = nn.Linear(n_hidden * sum(pyramid), 512)
        self.classify = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        num_per_batch = g.batch_num_nodes()
        node_degrees = g.in_degrees()
        if self.readout == 'min':
            hg = dgl.readout_nodes(g, feat='h', op='min')
        if self.readout == 'mean':
            hg = dgl.readout_nodes(g, feat='h', op='mean')
        if self.readout == 'max':
            hg = dgl.readout_nodes(g, feat='h', op='max')
        if self.readout == 'sum':
            hg = dgl.readout_nodes(g, feat='h', op='sum')
        if self.readout == 'pplp':
            hg = self.nfpp_layer(h, num_per_batch, node_degrees)
        return self.classify(hg)
