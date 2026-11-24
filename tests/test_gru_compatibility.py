
import numpy as np
import torch

# Import your uniform sampling function
from src.sampling.temporal import sample

# Import Murad's GRU encoder
from src.models.translator import PoseGRUEncoder


def test_gru_compatibility():

    # Step 1: Simulate Shahjad's output (20 frames)
    pose = np.random.rand(20, 75, 3).astype(np.float32)
    mask = np.ones((20, 75), dtype=bool)

    # Step 2: Apply your sampler
    # 20 real frames + 44 padded frames
    sampled_pose, sampled_mask, indices = sample(
        pose, mask, target_frames=64
    )

    # Step 3: Convert NumPy arrays to PyTorch tensors
    # unsqueeze(0) adds the batch dimension
    pose_tensor = torch.from_numpy(sampled_pose).unsqueeze(0)
    mask_tensor = torch.from_numpy(sampled_mask).unsqueeze(0)

    # Step 4: Calculate actual sequence length
    # Ignore indices marked -1 (padding)
    actual_length = int(np.sum(indices >= 0))
    lengths = torch.tensor([actual_length], dtype=torch.long)

    # Step 5: Initialize Murad's GRU
    encoder = PoseGRUEncoder()
    encoder.eval()

    # Step 6: Run forward pass without computing gradients
    with torch.no_grad():
        encoded, hidden = encoder(
            pose_tensor, mask_tensor, lengths
        )

    # Step 7: Verify output dimensions
    assert encoded.shape == (1, 64, 256)
    assert hidden.shape == (2, 1, 128)

    # Step 8: Verify the correct sequence length
    assert actual_length == 20

    # Padded output positions should remain zero
    assert torch.all(encoded[:, 20:, :] == 0)
