from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA2IDX = {aa: index for index, aa in enumerate(AA_LIST)}
UNK_AA_IDX = 20
AA_VOCAB_SIZE = 21


def one_hot_encode_dna(seq: str, max_len: int = 101) -> np.ndarray:
    seq = seq.upper()
    if len(seq) != max_len:
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + ("N" * (max_len - len(seq)))

    arr = np.zeros((4, max_len), dtype=np.float32)
    for position, base in enumerate(seq):
        idx = BASE2IDX.get(base)
        if idx is not None:
            arr[idx, position] = 1.0
    return arr


def one_hot_encode_protein(seq: str, max_len: int = 800) -> np.ndarray:
    seq = seq.upper().replace(" ", "")
    if len(seq) > max_len:
        seq = seq[:max_len]

    arr = np.zeros((AA_VOCAB_SIZE, max_len), dtype=np.float32)
    for position in range(max_len):
        if position < len(seq):
            aa = seq[position]
            idx = AA2IDX.get(aa, UNK_AA_IDX)
        else:
            idx = UNK_AA_IDX
        arr[idx, position] = 1.0
    return arr


class MultiScaleConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 7, 11),
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for kernel in kernel_sizes:
            padding = (kernel // 2) * dilation
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    kernel_size=kernel,
                    padding=padding,
                    dilation=dilation,
                )
            )
            self.bns.append(nn.BatchNorm1d(branch_channels))
        self.out_channels = branch_channels * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for conv, bn in zip(self.convs, self.bns):
            y = conv(x)
            y = bn(y)
            y = F.relu(y)
            outputs.append(y)
        return torch.cat(outputs, dim=1)


class MultiModalMSTC_CrossAttn(nn.Module):
    def __init__(
        self,
        dna_channels: int = 4,
        prot_channels: int = AA_VOCAB_SIZE,
        dna_branch_channels: int = 64,
        prot_branch_channels: int = 64,
        dna_kernels: tuple[int, ...] = (5, 9, 13),
        prot_kernels: tuple[int, ...] = (9, 15, 21),
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.dna_mstc = MultiScaleConv1D(
            in_channels=dna_channels,
            branch_channels=dna_branch_channels,
            kernel_sizes=dna_kernels,
            dilation=1,
        )
        self.dna_conv1x1 = nn.Conv1d(self.dna_mstc.out_channels, 128, kernel_size=1)
        self.dna_bn = nn.BatchNorm1d(128)
        self.dna_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dna_max_pool = nn.AdaptiveMaxPool1d(1)

        self.prot_mstc = MultiScaleConv1D(
            in_channels=prot_channels,
            branch_channels=prot_branch_channels,
            kernel_sizes=prot_kernels,
            dilation=1,
        )
        self.prot_conv1x1 = nn.Conv1d(self.prot_mstc.out_channels, 128, kernel_size=1)
        self.prot_bn = nn.BatchNorm1d(128)
        self.prot_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.prot_max_pool = nn.AdaptiveMaxPool1d(1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=attn_heads,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        dna_x: torch.Tensor,
        prot_x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x_d = self.dna_mstc(dna_x)
        x_d = self.dna_conv1x1(x_d)
        x_d = self.dna_bn(x_d)
        x_d = F.relu(x_d)

        dna_tokens = x_d.transpose(1, 2)
        x1_avg = self.dna_avg_pool(x_d).squeeze(-1)
        x1_max = self.dna_max_pool(x_d).squeeze(-1)
        x1 = torch.cat([x1_avg, x1_max], dim=1)

        x_p = self.prot_mstc(prot_x)
        x_p = self.prot_conv1x1(x_p)
        x_p = self.prot_bn(x_p)
        x_p = F.relu(x_p)

        x2_avg = self.prot_avg_pool(x_p).squeeze(-1)
        x2_max = self.prot_max_pool(x_p).squeeze(-1)
        x2_global = torch.cat([x2_avg, x2_max], dim=1)

        query = x2_max.unsqueeze(1)
        attended, attn_weights = self.cross_attn(query, dna_tokens, dna_tokens, need_weights=True)
        attended = attended.squeeze(1)

        fused = torch.cat([attended, x1, x2_global], dim=1)
        logit = self.fc(fused).squeeze(-1)
        if return_attn:
            return logit, attn_weights
        return logit
