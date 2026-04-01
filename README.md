# Emotion Classification Chatbot Using NLP

## Project Direction

This project is now structured to match an academic report format that requires:

- clear dataset-based experimentation
- three algorithms
- comparative evaluation
- a final chatbot that uses the best model

## Dataset Choice

The recommended external dataset for this project is a Kaggle mirror of `dair-ai/emotion`.

Why it fits:

- it is designed for text emotion classification
- its original labels can be mapped to the required five classes
- it supports a stronger academic report than a tiny manually written dataset

Label mapping used:

- `joy` -> `happy`
- `love` -> `happy`
- `sadness` -> `sad`
- `anger` -> `angry`
- `fear` -> `fear`
- `surprise` -> `surprise`

If Kaggle files are not present, the code falls back to the local file `emotion_dataset.json` so the project still runs.

## Implemented Methods

The current code uses exactly three report-style methods:

1. Naive Bayes + Bag of Words
2. Logistic Regression + TF-IDF
3. Linear SVM + TF-IDF

The chatbot then uses the best validation model.

## Feature Extraction Used in Code

- Bag of Words
- TF-IDF
- unigrams and bigrams

For the written report, you can still describe:

- word embeddings
- sentence embeddings
- transformer representations

as additional advanced feature extraction approaches, even if the present runnable code focuses on BoW and TF-IDF.

## Output Generated

When you run the project, it shows:

- dataset source
- train, validation, and test split summary
- comparison table across three methods
- detailed evaluation metrics
- confusion matrix
- sample conversations
- interactive chatbot mode

It also saves report-friendly files inside `report_artifacts/`:

- `results_summary.json`
- `results_table.csv`
- `training_history.csv`

## Kaggle File Layout

Place the Kaggle files like this:

```text
NLP_Project/
  kaggle_emotion/
    train.txt
    val.txt
    test.txt
```

Expected line format:

```text
text;label
```

Example:

```text
i feel nervous about tomorrow;fear
```

## Run

```bash
python3 chatbot.py
```

## Files

- `chatbot.py` - training, evaluation, and chatbot logic
- `emotion_dataset.json` - fallback local dataset
- `report_artifacts/results_table.csv` - summary comparison table
- `report_artifacts/training_history.csv` - training history for graph plotting
- `README.md` - project documentation

## Important Note

The project is now aligned with the report requirement much better than the earlier hybrid-only version. The comparison will become much stronger once the Kaggle dataset files are added, because the current fallback dataset is small and mainly for demonstration.
