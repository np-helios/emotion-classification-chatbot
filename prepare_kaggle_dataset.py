import json
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "joy": "happy",
    "love": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "fear": "fear",
    "surprise": "surprise",
}
FINAL_CLASSES = ["happy", "sad", "angry", "fear", "surprise"]
SAMPLES_PER_CLASS = 2000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def sanitize_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").replace("\r", " ").split())


def save_split(file_path: Path, rows: list[tuple[str, str]]) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        for text, label in rows:
            handle.write(f"{text};{label}\n")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    source_path = project_dir / "kaggle_emotion" / "merged_training.pkl"
    output_dir = project_dir / "kaggle_emotion"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_pickle(source_path)
    df = df[["text", "emotions"]].copy()
    df["mapped_emotion"] = df["emotions"].astype(str).str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["mapped_emotion"])
    df["text"] = df["text"].map(sanitize_text)
    df = df[df["text"].str.len() > 0]

    balanced_parts = []
    for label in FINAL_CLASSES:
        class_df = df[df["mapped_emotion"] == label]
        sample_n = min(SAMPLES_PER_CLASS, len(class_df))
        balanced_parts.append(class_df.sample(n=sample_n, random_state=42))

    balanced_df = pd.concat(balanced_parts, ignore_index=True)

    splits = {"train": [], "val": [], "test": []}
    summary = {"source_rows": int(len(df)), "balanced_rows": int(len(balanced_df)), "per_class": {}}

    for label in FINAL_CLASSES:
        class_rows = balanced_df[balanced_df["mapped_emotion"] == label].sample(frac=1.0, random_state=42)
        records = list(zip(class_rows["text"].tolist(), class_rows["mapped_emotion"].tolist()))

        train_end = int(len(records) * TRAIN_RATIO)
        val_end = int(len(records) * (TRAIN_RATIO + VAL_RATIO))

        splits["train"].extend(records[:train_end])
        splits["val"].extend(records[train_end:val_end])
        splits["test"].extend(records[val_end:])

        summary["per_class"][label] = {
            "total": len(records),
            "train": len(records[:train_end]),
            "val": len(records[train_end:val_end]),
            "test": len(records[val_end:]),
        }

    save_split(output_dir / "train.txt", splits["train"])
    save_split(output_dir / "val.txt", splits["val"])
    save_split(output_dir / "test.txt", splits["test"])

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Created balanced dataset splits in:", output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
