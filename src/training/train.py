
import torch
import torch.nn.functional as F


def run_epoch(model, dataloader, optimizer=None, device="cpu", pad_id=0):
    training = optimizer is not None
    model.train(training)

    total_loss = 0
    total_tokens = 0

    for batch in dataloader:
        pose = batch["pose"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        lengths = batch["lengths"].to(device)
        tokens = batch["tokens"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits, _ = model(pose, valid_mask, lengths, tokens)

            targets = tokens[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id, reduction="sum")

            token_count = (targets != pad_id).sum().item()

            if token_count == 0:
                continue

            if training:
                (loss / token_count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("No valid target tokens found")

    return total_loss / total_tokens
