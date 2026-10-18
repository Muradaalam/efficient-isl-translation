
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence,pad_packed_sequence)


class PoseGRUEncoder(nn.Module):
    """Encode variable-length pose sequences using a bidirectional GRU."""

    def __init__(self,num_landmarks=75,coord_dim=3,projection_dim=128,hidden_dim=128,num_layers=1,dropout=0.1):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.coord_dim = coord_dim
        self.output_dim = hidden_dim * 2

        self.projection = nn.Sequential(nn.Linear(num_landmarks * coord_dim, projection_dim),nn.ReLU())

        self.gru = nn.GRU(input_size=projection_dim,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True,bidirectional=True,dropout=dropout if num_layers > 1 else 0.0)

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

        if (joints, coords) != (self.num_landmarks,self.coord_dim):
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

        packed = pack_padded_sequence(projected,lengths.detach().cpu(),batch_first=True,enforce_sorted=False)

        packed_output, hidden = self.gru(packed)

        encoded, _ = pad_packed_sequence(packed_output,batch_first=True,total_length=frames)

        return encoded, hidden






class AttentionDecoder(nn.Module):
    """
    Attention-based GRU decoder.

    Compatible with PoseGRUEncoder using:
      encoder hidden_dim = 128
      bidirectional = True

    Expected inputs:
      encoded:        [B, T, 256]
      encoder_hidden: [2 * num_layers, B, 128]
      lengths:        [B]
      target_tokens:  [B, L]

    Returns:
      logits:         [B, L-1, vocab_size]
      attentions:     [B, L-1, T]
    """

    def __init__(self,vocab_size,encoder_dim=256,embedding_dim=128,hidden_dim=256,attention_dim=128,pad_token_id=0):
        super().__init__()

        # Convert English token IDs into learned vectors.
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=pad_token_id)

        # Additive attention layers.
        self.key_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.query_layer = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.score_layer = nn.Linear(attention_dim, 1, bias=False)

        # Initialize decoder state from encoder final states.
        self.init_hidden = nn.Sequential(nn.Linear(encoder_dim, hidden_dim),nn.Tanh())

        # Decoder GRU.
        self.gru = nn.GRU(input_size=embedding_dim + encoder_dim,hidden_size=hidden_dim,batch_first=True)

        # Predict vocabulary scores.
        self.output_layer = nn.Linear(hidden_dim + encoder_dim,vocab_size)

    def forward(self,encoded,encoder_hidden,lengths,target_tokens):
        batch, frames, _ = encoded.shape


        if (target_tokens.ndim != 2 or target_tokens.shape[0] != batch or target_tokens.shape[1] < 2):
            raise ValueError("target_tokens must have shape [B, L] with L >= 2")

        if (lengths.shape != (batch,) or torch.any(lengths < 1) or torch.any(lengths > frames)):
            raise ValueError("Invalid sequence lengths")

        if (encoder_hidden.ndim != 3 or encoder_hidden.shape[0] < 2 or encoder_hidden.shape[0] % 2 != 0 or encoder_hidden.shape[1] != batch):
            raise ValueError("Expected bidirectional encoder hidden states")

        # True for actual video frames, False for padding.
        source_mask = (torch.arange(frames, device=encoded.device).unsqueeze(0) < lengths.to(encoded.device).unsqueeze(1))

        # Combine the encoder's final forward/backward states.
        final_encoder_state = torch.cat((encoder_hidden[-2], encoder_hidden[-1]),dim=-1)

        # Convert into the initial decoder hidden state.
        hidden = self.init_hidden(final_encoder_state).unsqueeze(0)

        # Precompute attention keys.
        keys = self.key_layer(encoded)

        step_logits = []
        step_attentions = []

        # Teacher forcing: feed each correct previous token.
        for step in range(target_tokens.size(1) - 1):

            # Score video frames using the current hidden state.
            query = self.query_layer(hidden[-1]).unsqueeze(1)

            scores = self.score_layer(torch.tanh(keys + query)).squeeze(-1)

            # Padded video frames must receive no attention.
            scores = scores.masked_fill(~source_mask,torch.finfo(scores.dtype).min)

            attention = torch.softmax(scores, dim=1)

            # Weighted sum of encoder outputs.
            context = torch.bmm(attention.unsqueeze(1),encoded)

            # Embed the previous correct English token.
            word = self.embedding(target_tokens[:, step]).unsqueeze(1)

            # Combine word embedding and visual context.
            gru_input = torch.cat((word, context),dim=-1)

            # Update decoder hidden state.
            output, hidden = self.gru(gru_input,hidden)

            # Predict scores for the next English token.
            logits = self.output_layer(torch.cat((output, context), dim=-1)).squeeze(1)

            step_logits.append(logits)
            step_attentions.append(attention)

        return (torch.stack(step_logits, dim=1),torch.stack(step_attentions, dim=1))

class ISLTranslator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.encoder = PoseGRUEncoder()
        self.decoder = AttentionDecoder(vocab_size=vocab_size)

    def forward(self, pose, valid_mask, lengths, target_tokens):
        encoded, hidden = self.encoder(pose, valid_mask, lengths)

        logits, attention = self.decoder(encoded, hidden, lengths, target_tokens)

        return logits, attention
