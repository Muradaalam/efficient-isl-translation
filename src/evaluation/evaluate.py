from src.evaluation.metrics import exact_match_accuracy, word_error_rate, bleu_scores


def evaluate_predictions(references, predictions):
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have the same length")

    if not references:
        raise ValueError("Cannot evaluate an empty dataset")

    results = {
        "num_samples": len(references),
        "exact_match": exact_match_accuracy(references, predictions),
        "wer": word_error_rate(references, predictions)
    }

    results.update(bleu_scores(references, predictions))

    return results