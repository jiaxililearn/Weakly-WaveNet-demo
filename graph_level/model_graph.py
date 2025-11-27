"""
Graph-level prediction model based on Graph WaveNet

Modified from the original node-level forecasting model to support graph-level
classification/regression tasks.

Key differences:
- Adds global pooling to aggregate node and temporal information
- Outputs single prediction per graph instead of per-node per-timestep
- Suitable for tasks like yearly predictions from daily sensor data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from model import nconv, linear, gcn


class gwnet_graph_level(nn.Module):
    """Graph WaveNet for graph-level prediction tasks"""

    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        out_dim=1,  # Graph-level output dimension
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
        pooling='mean',  # 'mean', 'max', 'attention'
        task='regression',  # 'regression' or 'classification'
    ):
        super(gwnet_graph_level, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.pooling = pooling
        self.task = task

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        # Initial feature transformation
        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        # Adaptive adjacency matrix learning
        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10).to(device), requires_grad=True
                ).to(device)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes).to(device), requires_grad=True
                ).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        # Build temporal convolution layers with GCN
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # Dilated convolutions for temporal modeling
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )

                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )

                # 1x1 convolution for residual connection
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )

                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                if self.gcn_bool:
                    self.gconv.append(
                        gcn(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len,
                        )
                    )

        # Processing before pooling
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.receptive_field = receptive_field

        # Graph-level readout layers
        if pooling == 'attention':
            # Attention-based pooling
            self.attention_weights = nn.Sequential(
                nn.Conv2d(end_channels, end_channels // 4, kernel_size=(1, 1)),
                nn.ReLU(),
                nn.Conv2d(end_channels // 4, 1, kernel_size=(1, 1))
            )

        # Final prediction head
        self.fc = nn.Sequential(
            nn.Linear(end_channels, end_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(end_channels // 2, out_dim)
        )

    def forward(self, input):
        """
        Forward pass for graph-level prediction

        Args:
            input: (batch, in_dim, num_nodes, seq_length)

        Returns:
            output: (batch, out_dim) - graph-level prediction
        """
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        # Calculate adaptive adjacency matrix
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers with skip connections
        for i in range(self.blocks * self.layers):
            residual = x

            # Gated activation
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # Skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # Graph convolution
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # Process skip connections
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        # x shape: (batch, end_channels, num_nodes, time)

        # Graph-level pooling
        if self.pooling == 'mean':
            # Average pooling over nodes and time
            x = torch.mean(x, dim=[2, 3])  # (batch, end_channels)
        elif self.pooling == 'max':
            # Max pooling over nodes and time
            x = torch.amax(x, dim=[2, 3])  # (batch, end_channels)
        elif self.pooling == 'attention':
            # Attention-weighted pooling
            attention = self.attention_weights(x)  # (batch, 1, num_nodes, time)
            attention = F.softmax(attention.view(attention.size(0), -1), dim=1)  # (batch, nodes*time)
            attention = attention.view(x.size(0), 1, x.size(2), x.size(3))
            x = torch.sum(x * attention, dim=[2, 3])  # (batch, end_channels)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Final prediction
        output = self.fc(x)  # (batch, out_dim)

        return output


class gwnet_encoder(nn.Module):
    """
    Encoder-only version of Graph WaveNet for feature extraction
    Can be used for transfer learning or as a feature extractor
    """

    def __init__(
        self,
        device,
        num_nodes,
        dropout=0.3,
        supports=None,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=2,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
    ):
        super(gwnet_encoder, self).__init__()
        # Same architecture as gwnet_graph_level but without the final FC layers
        # This allows you to extract features and add custom heads

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.supports = supports

        receptive_field = 1
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(num_nodes, 10).to(device), requires_grad=True
                ).to(device)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, num_nodes).to(device), requires_grad=True
                ).to(device)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )

                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )

                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )

                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

                if self.gcn_bool:
                    self.gconv.append(
                        gcn(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=self.supports_len,
                        )
                    )

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self.receptive_field = receptive_field

    def forward(self, input):
        """
        Returns encoded features without pooling
        Output: (batch, end_channels, num_nodes, time)
        """
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x

            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))

        return x
