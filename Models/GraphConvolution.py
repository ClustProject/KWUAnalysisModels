import torch
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels,
                 seed=2023,
                 bias=False
                 ):
        super(GraphConvolution, self).__init__()

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.torch.empty(out_channels))
        #         else:
        #             self.register_parameter('bias',None)
        self.reset_parameters()

    def forward(self, x, adj):
        out = torch.mm(x, self.weight)
        out = torch.mm(adj, out)

        if self.bias is not None:
            out += self.bias
        return out

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)