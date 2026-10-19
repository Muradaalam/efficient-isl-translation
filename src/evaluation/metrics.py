def exact_match_accuracy(references, predictions):
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have the same length")

    if not references:
        return 0.0

    correct = 0

    for ref, pred in zip(references, predictions):
        if ref.strip().lower() == pred.strip().lower():
            correct += 1

    return correct / len(references)
