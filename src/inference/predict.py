def decode_tokens(token_ids, id_to_token, eos_id, special_ids):
    words = []

    for token_id in token_ids:
        token_id = int(token_id)

        if token_id == eos_id:
            break

        if token_id in special_ids:
            continue

        if token_id not in id_to_token:
            raise ValueError(f"Unknown token ID: {token_id}")

        words.append(id_to_token[token_id])

    return " ".join(words)
