from torch import Tensor
import torch.nn as nn
import torch
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
import math
import copy
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
from torch_geometric.utils import softmax


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = .1, learnable=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.learnable = learnable
        # Compute the positional encodings in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe /= 10
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=self.learnable)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = .1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)  # Dim Extension
        self.l2 = nn.Linear(d_ff, d_model)  # Dim reduction
        self.relu = nn.CELU()

        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ffn = self.relu(self.l2(self.dropout(self.relu(self.l1(x)))))
        res = self.norm(x + self.dropout(ffn))
        return res


class GraphTransformer(MessagePassing):

    _alpha: OptTensor

    def __init__(
            self,
            n_samples: int,
            gcn_in_channels: int,  # Union[int, Tuple[int, int]],
            gcn_out_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            beta: bool = False,
            dropout: float = 0.,
            edge_dim: Optional[int] = None,
            bias: bool = True,
            root_weight: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = gcn_in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = nn.Dropout(p=dropout)
        self.edge_dim = edge_dim
        self.bias = bias
        self.gcn_norm = LayerNorm(gcn_out_channels)
        self.tfn_norm = None
        self.tfn_norm = LayerNorm(out_channels)
        self.relu = nn.CELU()
        self.softmax = nn.Softmax(dim=-1)

        #         self.positional_encoding = PositionalEncoding(gcn_out_channels, n_samples, dropout=dropout, learnable = False)

        #         self.conv1 = GCNConv(gcn_in_channels, gcn_out_channels*2, cached = True)
        #         self.conv2 = GCNConv(gcn_out_channels*2, gcn_out_channels, cached = True)

        self.conv1 = GraphConvolution(gcn_in_channels, gcn_out_channels * 2, bias=False)
        self.conv2 = GraphConvolution(gcn_out_channels * 2, gcn_out_channels, bias=False)

        if isinstance(gcn_out_channels, int):
            gcn_out_channels = (gcn_out_channels, gcn_out_channels)

        self.lin_key = Linear(gcn_out_channels[0], heads * out_channels)
        self.lin_query = Linear(gcn_out_channels[1], heads * out_channels)
        self.lin_value = Linear(gcn_out_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.tfn_norm = LayerNorm(out_channels * heads)

            self.lin_skip = Linear(gcn_out_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.tfn_norm = LayerNorm(out_channels)

            self.lin_skip = Linear(gcn_out_channels[1], out_channels, bias=bias)

            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    #     def forward(self, features: Union[Tensor, PairTensor], edge_index: Adj,
    #                 edge_attr: OptTensor = None, return_attention_weights=None):
    def forward(self, features: Union[Tensor, PairTensor], adj, edge_index: Adj = None,
                edge_attr: OptTensor = None, return_attention_weights=None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        # "Dropout" use of not. Dropout for convolutional layer doesn't work well?
        #         x1 = self.dropout(self.relu(self.conv1(features, edge_index, edge_attr)))
        #         x = self.dropout(self.relu(self.conv2(x1, edge_index, edge_attr)))

        x1 = self.dropout(self.relu(self.conv1(features, adj)))
        x = self.dropout(self.relu(self.conv2(x1, adj)))
        #         print(x)
        # "Residual connection (adding)" doesn't work because dimensions of x1 and x2 are different
        # x2 = self.conv2(x1, edge_index, edge_attr)
        # x = self.dropout(self.relu(self.gcn_norm(x1+x2)))

        # positional encoding
        #         x = self.positional_encoding(x)
        #         print(x)

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        #         if self.lin_edge is not None:
        #             assert adj is not None
        #             adj = self.lin_edge(adj).view(-1, self.heads,
        #                                                       self.out_channels)
        #             key = key + adj

        b = query.transpose(1, 0)
        c = key.transpose(1, 0).transpose(1, 2)
        alpha = torch.matmul(b, c) / math.sqrt(self.out_channels)
        print(torch.where(alpha < 0))
        alpha = self.softmax(alpha)
        alpha = self.dropout(alpha)

        #         print("alpha ", alpha)
        out = value
        #         if adj is not None:
        #             out = out + adj

        out = torch.matmul(alpha, out.transpose(1, 0)).transpose(1, 0)

        #         print("alpha * value: ", out)
        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        #         print("reshape H*C or mean : ", out)
        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        out = self.relu(self.tfn_norm(out))
        #         print("Gated residual connection : ", out)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, adj: Tensor, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert adj is not None
            adj = self.lin_edge(adj).view(-1, self.heads,
                                          self.out_channels)
            key_j = key_j + adj

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class Encoder_GT_Layer(nn.Module):
    def __init__(self,
                 n_samples: int,
                 gcn_in_channels: int,  # Union[int, Tuple[int, int]],
                 gcn_out_channels: int,
                 tfn_out_channels: int,
                 ffn_proj_hid_channels: int,
                 heads: int = 1,
                 concat: bool = True,
                 beta: bool = False,
                 edge_dim: Optional[int] = None,
                 bias: bool = True,
                 root_weight: bool = True,
                 dropout: float = .1):
        super(Encoder_GT_Layer, self).__init__()

        self.self_attn = GraphTransformer(n_samples, gcn_in_channels, gcn_out_channels, tfn_out_channels, heads, concat,
                                          beta, dropout, edge_dim, bias, root_weight)
        self.feed_forward = None
        if concat:
            self.feed_forward = FeedForward(heads * tfn_out_channels, ffn_proj_hid_channels, dropout)

    #     def forward(self, x, edge_index, edge_attr):
    #         x = self.self_attn(x,edge_index,edge_attr)
    #         return self.feed_forward(x, self.feed_forward)

    def forward(self, x, adj):
        x = self.self_attn(x, adj)
        if self.feed_forward is not None:
            self.feed_forward(x)
        return x


class Encoder_GT(nn.Module):
    def __init__(self,
                 hid_layer,
                 last_layer,
                 n: int = 3):
        super(Encoder_GT, self).__init__()
        self.hid_layers = nn.ModuleList([copy.deepcopy(hid_layer) for _ in range(n - 1)])
        self.last_layer = last_layer

    def forward(self, x, adj):
        for idx, layer in enumerate(self.hid_layers):
            x = layer(x, adj)
        x = self.last_layer(x, adj)
        return x