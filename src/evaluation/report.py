
import csv
import json
from pathlib import Path

from src.evaluation.evaluate import evaluate_predictions


def save_evaluation_report(sample_ids, references, predictions, output_dir, experiment_id, split="test"):
    if not (len(sample_ids) == len(references) == len(predictions)):
        raise ValueError("Sample IDs, references and predictions must have the same length")

    if not sample_ids:
        raise ValueError("Cannot save an empty evaluation report")

    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs found")

    results = evaluate_predictions(references, predictions)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "predictions.csv"
    json_path = output_dir / "metrics.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "reference", "prediction"])

        for sample_id, ref, pred in zip(sample_ids, references, predictions):
            writer.writerow([sample_id, ref, pred])

    report = {
        "experiment_id": experiment_id,
        "split": split,
        **results
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report
