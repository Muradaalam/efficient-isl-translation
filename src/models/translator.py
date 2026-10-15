
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class PoseGRUEncoder(nn.Module):
    """Encode variable-length pose sequences using a bidirectional GRU."""

    def __init__(
        self,
        num_landmarks=75,
        coord_dim=3,
        projection_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.output_dim = hidden_dim * 2

        self.projection = nn.Sequential(
            nn.Linear(num_landmarks * coord_dim, projection_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, pose, valid_mask, lengths):
        """
        Args:
            pose: [B, T, J, C]
            valid_mask: [B, T, J], True for valid landmarks
            lengths: [B], valid frame counts (must be >= 1)

        Returns:
            encoded: [B, T, 2 * hidden_dim]
            hidden: [2 * num_layers, B, hidden_dim]
        """
        if pose.ndim != 4:
            raise ValueError("pose must have shape [B, T, J, C]")

        batch, frames, joints, coords = pose.shape

        if (joints, coords) != (
            self.num_landmarks,
            self.coord_dim,
        ):
            raise ValueError("Unexpected landmark dimensions")

        if valid_mask.shape != (batch, frames, joints):
            raise ValueError("Unexpected valid_mask shape")

        if lengths.shape != (batch,):
            raise ValueError("lengths must have shape [B]")

        if torch.any(lengths < 1) or torch.any(lengths > frames):
            raise ValueError("Invalid sequence lengths")

        # Missing landmarks must not contribute to the encoder input.
        masked_pose = pose * valid_mask.unsqueeze(-1).to(pose.dtype)

        features = masked_pose.reshape(batch, frames, -1)
        projected = self.projection(features)

        packed = pack_padded_sequence(
            projected,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, hidden = self.gru(packed)

        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=frames,
        )

        return encoded, hidden
