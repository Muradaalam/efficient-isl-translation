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

#using Levenshtein Distance Algorithm(DP)
def word_error_rate(references, predictions):
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have the same length")

    total_errors = 0
    total_words = 0

    for ref, pred in zip(references, predictions):
        ref_words = ref.strip().lower().split()
        pred_words = pred.strip().lower().split()

        rows = len(ref_words) + 1
        cols = len(pred_words) + 1

        dp = [[0] * cols for _ in range(rows)]

        for i in range(rows):
            dp[i][0] = i

        for j in range(cols):
            dp[0][j] = j

        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if ref_words[i - 1] == pred_words[j - 1] else 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )

        total_errors += dp[-1][-1]
        total_words += len(ref_words)

    if total_words == 0:
        return 0.0 if total_errors == 0 else float("inf")

    return total_errors / total_words

