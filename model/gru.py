import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple

class PF_GRU(nn.Module):
    """
    GRU forecaster with optional metadata fusion.

    Args:
        input_size:     # features per time step
        hidden_size:    # GRU hidden size (per direction)
        num_layers:     # GRU layers
        pred_length:    # output horizon
        sequence_length:# input sequence length (needed to size the FC when flattening)
        meta_dim:       # dimension of metadata vector; if None, metadata is disabled
        meta_hidden:    # hidden sizes for metadata MLP (e.g., (64, 128))
        use_last_timestep: If True, use last hidden state instead of flattening all steps
        bidirectional:  # if you want to enable later; keeps code future-proof
        dropout:        # GRU dropout between layers (ignored if num_layers <= 1)
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 pred_length: int,
                 sequence_length: int,
                 meta_dim: Optional[int] = 40,
                 meta_hidden: Sequence[int] = (64, 128),
                 use_last_timestep: bool = False,
                 bidirectional: bool = False,
                 dropout: float = 0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.sequence_length = sequence_length
        self.meta_dim = meta_dim
        self.use_last_timestep = use_last_timestep
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        dir_mult = 2 if bidirectional else 1

        # Choose representation size from GRU
        if use_last_timestep:
            gru_feat_dim = hidden_size * dir_mult
        else:
            gru_feat_dim = hidden_size * dir_mult * sequence_length

        # Optional metadata MLP (like your ResNet meta_data_layer1/2)
        if meta_dim is not None:
            mlp_layers = []
            in_dim = meta_dim
            for h in meta_hidden:
                mlp_layers += [nn.Linear(in_dim, h), nn.LeakyReLU()]
                in_dim = h
            self.meta_mlp = nn.Sequential(*mlp_layers)
            meta_out_dim = meta_hidden[-1] if len(meta_hidden) > 0 else meta_dim
        else:
            self.meta_mlp = None
            meta_out_dim = 0

        # Final head: concatenate [gru_features, meta_features] → prediction
        self.fc = nn.Linear(gru_feat_dim + meta_out_dim, pred_length)

    def forward(self, x: torch.Tensor, meta_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:         [B, T, F]
        meta_data: [B, meta_dim] (if meta_dim is not None)
        """
        x = x.transpose(1,2)

        B = x.size(0)
        h0 = x.new_zeros(self.num_layers * (2 if self.bidirectional else 1),
                         B, self.hidden_size)

        out, hN = self.gru(x, h0)

        if self.use_last_timestep:
            # Use final hidden state from the last layer (concat directions if bi)
            if self.bidirectional:
                # hN shape: [num_layers*2, B, H] → take last layer's two directions and concat
                last_fw = hN[-2, :, :]  # [B, H]
                last_bw = hN[-1, :, :]  # [B, H]
                feat = torch.cat([last_fw, last_bw], dim=-1)  # [B, 2H]
            else:
                feat = hN[-1, :, :]  # [B, H]
        else:
            # Flatten all time steps (original behavior)
            feat = out.reshape(out.shape[0], -1)  # [B, T*H*(dir)]

        # Process metadata if provided/enabled
        if self.meta_mlp is not None:
            if meta_data is None:
                # If meta is enabled but not passed at call time, fall back to zeros of the right shape
                meta_data = x.new_zeros((B, self.meta_mlp[0].in_features))
            meta_feat = self.meta_mlp(meta_data)
            feat = torch.cat([feat, meta_feat], dim=1)

        y = self.fc(feat)
        return y
