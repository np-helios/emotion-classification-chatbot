import csv
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


EMOTIONS = ["happy", "sad", "angry", "fear", "surprise"]
NEGATIVE_EMOTIONS = {"sad", "angry", "fear"}
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
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being", "to", "for", "of", "in",
    "and", "or", "on", "with", "this", "that", "it", "me", "my", "we", "our", "you", "your", "they",
    "he", "she", "them", "their", "do", "does", "did", "have", "has", "had", "at", "from", "as", "if",
    "then", "there", "here", "about", "into", "onto", "while", "during"
}
NEGATIONS = {"not", "never", "no", "cannot", "can't", "dont", "don't", "didnt", "didn't", "won't", "wont", "isn't", "isnt"}
INTENSIFIERS = {"very", "really", "so", "too", "extremely", "quite", "super", "deeply"}
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "isn't": "is not",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
}
DOMAIN_KEYWORDS = {
    "work": {"office", "boss", "job", "work", "career", "meeting", "deadline", "project", "interview", "team", "manager", "exam", "viva", "presentation", "submission"},
    "relationships": {"friend", "partner", "boyfriend", "girlfriend", "family", "relationship", "marriage", "love", "message", "parents"},
    "health": {"health", "doctor", "pain", "sleep", "body", "medicine", "panic", "exercise", "diet", "report", "hospital"},
    "self_esteem": {"confidence", "worth", "myself", "failure", "insecure", "future", "motivation", "enough", "ability"},
}
DOMAIN_SUGGESTIONS = {
    "work": "Try reducing the issue into one next step, and if needed ask for very specific help.",
    "relationships": "A calm and honest message can help. Focus on one feeling and one request at a time.",
    "health": "Rest, hydrate, and if this feels serious or persistent, speaking with a qualified professional is the right next step.",
    "self_esteem": "Try separating the event from your identity. One difficult moment does not define your worth.",
    "general": "Take one slow breath, name the feeling, and focus on one practical next action you can control.",
}
CHATBOT_HINTS = {
    "happy": {"happy", "proud", "glad", "excited", "thrilled", "joy", "smiling", "grateful"},
    "sad": {"sad", "lonely", "empty", "cry", "hurt", "hopeless", "down", "awful"},
    "angry": {"angry", "mad", "furious", "annoyed", "frustrated", "rage", "irritated"},
    "fear": {"fear", "afraid", "scared", "worried", "anxious", "panic", "terrified", "nervous", "viva", "exam", "unprepared", "prepared"},
    "surprise": {"wow", "surprised", "unexpected", "shocked", "sudden", "astonished"},
}
CHATBOT_PHRASES = {
    "angry": {"fed up", "sick of", "makes me angry", "ignored my work", "ignored my effort"},
    "surprise": {"did not expect", "out of nowhere", "caught me off guard", "what a surprise"},
    "fear": {"scared about", "afraid of", "worried about", "panic attack", "have a viva", "viva in", "not prepared", "not ready", "exam in", "deadline in"},
    "sad": {"feel lonely", "want to cry", "not good enough", "feel empty"},
}


def normalize_text(text: str) -> str:
    lowered = text.lower()
    for source, target in CONTRACTIONS.items():
        lowered = lowered.replace(source, target)
    lowered = re.sub(r"(.)\1{2,}", r"\1\1", lowered)
    return lowered


def simple_stem(word: str) -> str:
    for suffix in ("ingly", "edly", "ation", "ments", "ment", "ing", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def preprocess(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z']+", normalize_text(text))
    cleaned = []
    negation_window = 0
    intensify_window = 0

    for token in raw_tokens:
        if token in NEGATIONS:
            negation_window = 3
            cleaned.append("NEG")
            continue
        if token in INTENSIFIERS:
            intensify_window = 2
            cleaned.append(token)
            continue

        stemmed = simple_stem(token)
        if stemmed in STOPWORDS:
            continue

        if negation_window > 0:
            stemmed = "not_" + stemmed
            negation_window -= 1
        if intensify_window > 0:
            stemmed = "int_" + stemmed
            intensify_window -= 1
        cleaned.append(stemmed)

    return cleaned


def parse_split_file(file_path: Path) -> list[dict]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ";" not in line:
                continue
            text, label = line.rsplit(";", 1)
            mapped = LABEL_MAP.get(label.strip().lower())
            if mapped:
                rows.append({"text": text.strip(), "emotion": mapped})
    return rows


def parse_csv_file(file_path: Path) -> list[dict]:
    rows = []
    with open(file_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get("text") or row.get("sentence") or row.get("content") or row.get("tweet") or row.get("Text")
            label = row.get("label") or row.get("emotion") or row.get("sentiment") or row.get("Emotion")
            if not text or not label:
                continue
            mapped = LABEL_MAP.get(str(label).strip().lower())
            if mapped:
                rows.append({"text": text.strip(), "emotion": mapped})
    return rows


def load_fallback_dataset(dataset_path: Path) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def stratified_split(records: list[dict], train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in records:
        grouped[row["emotion"]].append(row)

    splits = {"train": [], "val": [], "test": []}
    for emotion in EMOTIONS:
        items = grouped[emotion]
        train_end = max(1, int(len(items) * train_ratio))
        val_end = max(train_end + 1, int(len(items) * (train_ratio + val_ratio))) if len(items) > 2 else len(items)
        splits["train"].extend(items[:train_end])
        splits["val"].extend(items[train_end:val_end])
        splits["test"].extend(items[val_end:])
    return splits


def discover_dataset(project_dir: Path) -> tuple[dict[str, list[dict]], str]:
    kaggle_dir = project_dir / "kaggle_emotion"
    train_file = kaggle_dir / "train.txt"
    val_file = kaggle_dir / "val.txt"
    test_file = kaggle_dir / "test.txt"

    if train_file.exists() and val_file.exists() and test_file.exists():
        return {
            "train": parse_split_file(train_file),
            "val": parse_split_file(val_file),
            "test": parse_split_file(test_file),
        }, "Kaggle mirror of dair-ai/emotion"

    for candidate_name in ("kaggle_emotion.csv", "emotion.csv", "dataset.csv"):
        candidate = project_dir / candidate_name
        if candidate.exists():
            parsed = parse_csv_file(candidate)
            if parsed:
                return stratified_split(parsed), f"CSV dataset from {candidate_name}"

    fallback = load_fallback_dataset(project_dir / "emotion_dataset.json")
    return stratified_split(fallback), "Local curated fallback dataset"


def compute_binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    p = tp + fn
    n = tn + fp
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    sensitivity = tp / p if p else 0.0
    specificity = tn / n if n else 0.0
    fpr = fp / n if n else 0.0
    fnr = fn / p if p else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fdr = fp / (fp + tp) if (fp + tp) else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "npv": npv,
        "fdr": fdr,
        "mcc": mcc,
    }


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    matrix = {actual: {predicted: 0 for predicted in EMOTIONS} for actual in EMOTIONS}
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred)
    per_class = {}
    macro = defaultdict(float)

    for emotion in EMOTIONS:
        tp = matrix[emotion][emotion]
        fp = sum(matrix[other][emotion] for other in EMOTIONS if other != emotion)
        fn = sum(matrix[emotion][other] for other in EMOTIONS if other != emotion)
        tn = sum(
            matrix[actual][predicted]
            for actual in EMOTIONS for predicted in EMOTIONS
            if actual != emotion and predicted != emotion
        )
        metrics = compute_binary_metrics(tp, fp, tn, fn)
        per_class[emotion] = metrics
        for key, value in metrics.items():
            macro[key] += value

    macro_metrics = {key: value / len(EMOTIONS) for key, value in macro.items()}
    return {"matrix": matrix, "per_class": per_class, "macro": macro_metrics}


def make_vectorizer(kind: str, params: dict[str, object]):
    common = {
        "tokenizer": preprocess,
        "preprocessor": normalize_text,
        "token_pattern": None,
        "ngram_range": params["ngram_range"],
        "min_df": params["min_df"],
        "max_features": params["max_features"],
        "lowercase": False,
    }
    if kind == "bow":
        return CountVectorizer(**common)
    return TfidfVectorizer(**common, sublinear_tf=params.get("sublinear_tf", True), norm="l2")


def make_model(model_name: str, params: dict[str, object]):
    if model_name == "naive_bayes":
        return MultinomialNB(alpha=params["alpha"])
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=params["C"],
            solver="lbfgs",
            max_iter=params["max_iter"],
            class_weight="balanced",
        )
    if model_name == "linear_svm":
        return LinearSVC(
            C=params["C"],
            class_weight="balanced",
            max_iter=params["max_iter"],
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def run_experiment(config: dict[str, object], splits: dict[str, list[dict]]) -> dict[str, object]:
    train_texts = [row["text"] for row in splits["train"]]
    val_texts = [row["text"] for row in splits["val"]]
    test_texts = [row["text"] for row in splits["test"]]
    train_labels = [row["emotion"] for row in splits["train"]]
    val_labels = [row["emotion"] for row in splits["val"]]
    test_labels = [row["emotion"] for row in splits["test"]]

    best_val_f1 = -1.0
    best_model = None
    best_vectorizer = None
    best_params = None
    search_history = []

    for params in config["param_grid"]:
        vectorizer = make_vectorizer(config["feature_name_key"], params)
        x_train = vectorizer.fit_transform(train_texts)
        x_val = vectorizer.transform(val_texts)

        model = make_model(config["model_name"], params)
        model.fit(x_train, train_labels)

        val_pred = model.predict(x_val)
        val_metrics = evaluate_predictions(val_labels, list(val_pred))
        val_f1 = val_metrics["macro"]["f1"]
        search_history.append({"params": params, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_vectorizer = vectorizer
            best_params = params

    x_test = best_vectorizer.transform(test_texts)
    test_pred = list(best_model.predict(x_test))
    test_metrics = evaluate_predictions(test_labels, test_pred)

    return {
        "name": config["display_name"],
        "feature_name": config["feature_name"],
        "model_name": config["model_name"],
        "vectorizer": best_vectorizer,
        "model": best_model,
        "best_params": best_params,
        "val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "test_predictions": test_pred,
        "test_labels": test_labels,
        "search_history": search_history,
    }


class EmotionChatbot:
    def __init__(self, vectorizer, model, model_name: str) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.model_name = model_name
        self.memory = deque(maxlen=5)

    def detect_domain(self, text: str) -> str:
        lowered = normalize_text(text)
        token_set = set(preprocess(text))
        best_domain = "general"
        best_score = 0

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in lowered:
                    score += 2
                if token_set & set(preprocess(keyword)):
                    score += 1
            if score > best_score:
                best_score = score
                best_domain = domain
        return best_domain

    def predict_emotion(self, text: str) -> tuple[str, float]:
        vector = self.vectorizer.transform([text])
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vector)[0]
            indexed = {label: float(probabilities[idx]) for idx, label in enumerate(self.model.classes_)}
            return self._apply_chatbot_hints(text, indexed)

        scores = self.model.decision_function(vector)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()[0] if hasattr(scores[0], "__iter__") else scores.tolist()
        indexed_scores = {label: float(scores[idx]) for idx, label in enumerate(self.model.classes_)}
        max_score = max(indexed_scores.values())
        exp_scores = {label: math.exp(score - max_score) for label, score in indexed_scores.items()}
        total = sum(exp_scores.values()) or 1.0
        probabilities = {label: exp_scores[label] / total for label in indexed_scores}
        return self._apply_chatbot_hints(text, probabilities)

    def _apply_chatbot_hints(self, text: str, probabilities: dict[str, float]) -> tuple[str, float]:
        boosted = dict(probabilities)
        lowered = normalize_text(text)
        tokens = set(preprocess(text))

        academic_stress = {"viva", "exam", "interview", "deadline", "submission"}
        if tokens & academic_stress:
            boosted["fear"] += 0.18
            if any(phrase in lowered for phrase in {"not prepared", "not ready", "in 10 minute", "tomorrow"}):
                boosted["fear"] += 0.12

        for emotion, words in CHATBOT_HINTS.items():
            overlap = len(tokens & words)
            if overlap:
                boosted[emotion] += 0.10 * overlap

        for emotion, phrases in CHATBOT_PHRASES.items():
            if any(phrase in lowered for phrase in phrases):
                boosted[emotion] += 0.20

        total = sum(boosted.values()) or 1.0
        normalized = {label: value / total for label, value in boosted.items()}
        emotion = max(normalized, key=normalized.get)
        return emotion, normalized[emotion]

    def respond(self, user_input: str) -> dict[str, str]:
        emotion, confidence = self.predict_emotion(user_input)
        domain = self.detect_domain(user_input)

        memory_note = ""
        repeated = sum(1 for item in self.memory if item["emotion"] == emotion and emotion in NEGATIVE_EMOTIONS)
        if repeated >= 1:
            memory_note = f" I also notice this {emotion} feeling has come up again in our recent conversation."

        response_map = {
            "happy": "You sound happy. It is worth noticing what went well so you can build on it.",
            "sad": "You sound sad. Small and kind next steps matter when things feel heavy.",
            "angry": "You sound angry or deeply frustrated. A brief pause can help you choose the next step more clearly.",
            "fear": "You sound worried or afraid. It often helps to focus on one concrete action you can control.",
            "surprise": "You sound surprised. Taking a moment to process the change can help before reacting.",
        }

        response = response_map[emotion] + " " + DOMAIN_SUGGESTIONS.get(domain, DOMAIN_SUGGESTIONS["general"]) + memory_note
        self.memory.append({"text": user_input, "emotion": emotion, "domain": domain})
        return {
            "emotion": emotion,
            "domain": domain,
            "confidence": f"{confidence:.2f}",
            "response": response,
        }


def save_report_artifacts(project_dir: Path, dataset_source: str, splits: dict[str, list[dict]], experiments: list[dict[str, object]]) -> None:
    artifacts_dir = project_dir / "report_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    summary_rows = []
    search_rows = []
    for experiment in experiments:
        macro = experiment["test_metrics"]["macro"]
        summary_rows.append({
            "method": experiment["name"],
            "features": experiment["feature_name"],
            "accuracy": round(macro["accuracy"], 4),
            "precision_macro": round(macro["precision"], 4),
            "recall_macro": round(macro["recall"], 4),
            "f1_macro": round(macro["f1"], 4),
            "sensitivity": round(macro["sensitivity"], 4),
            "specificity": round(macro["specificity"], 4),
            "fpr": round(macro["fpr"], 4),
            "fnr": round(macro["fnr"], 4),
            "npv": round(macro["npv"], 4),
            "fdr": round(macro["fdr"], 4),
            "mcc": round(macro["mcc"], 4),
        })
        for idx, row in enumerate(experiment["search_history"], start=1):
            search_rows.append({
                "method": experiment["name"],
                "trial": idx,
                "params": json.dumps(row["params"], sort_keys=True),
                "val_f1": round(row["val_f1"], 6),
            })

    with open(artifacts_dir / "results_summary.json", "w", encoding="utf-8") as handle:
        json.dump({
            "dataset_source": dataset_source,
            "train_samples": len(splits["train"]),
            "val_samples": len(splits["val"]),
            "test_samples": len(splits["test"]),
            "results": summary_rows,
        }, handle, indent=2)

    with open(artifacts_dir / "results_table.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(artifacts_dir / "training_history.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "trial", "params", "val_f1"])
        writer.writeheader()
        writer.writerows(search_rows)


def print_header(dataset_source: str) -> None:
    print("=" * 88)
    print("EMOTION CLASSIFICATION CHATBOT - REPORT ALIGNED NLP PROJECT")
    print("=" * 88)
    print("Dataset source : " + dataset_source)
    print("Methods        : Naive Bayes + BoW | Logistic Regression + TF-IDF | Linear SVM + TF-IDF")
    print("Chatbot model  : best validation model")
    print()


def print_dataset_summary(splits: dict[str, list[dict]]) -> None:
    print("Dataset Summary")
    print("-" * 88)
    print(f"Train samples : {len(splits['train'])}")
    print(f"Val samples   : {len(splits['val'])}")
    print(f"Test samples  : {len(splits['test'])}")
    print("Class distribution:")
    for emotion in EMOTIONS:
        train_count = sum(1 for row in splits["train"] if row["emotion"] == emotion)
        val_count = sum(1 for row in splits["val"] if row["emotion"] == emotion)
        test_count = sum(1 for row in splits["test"] if row["emotion"] == emotion)
        print(f"  {emotion:<8} train={train_count:<4} val={val_count:<4} test={test_count:<4}")
    print()


def print_result_table(experiments: list[dict[str, object]]) -> None:
    print("Experimental Results")
    print("-" * 88)
    header = f"{'Method':<34}{'Acc':>7}{'Prec':>8}{'Rec':>8}{'F1':>8}{'MCC':>8}"
    print(header)
    for experiment in experiments:
        macro = experiment["test_metrics"]["macro"]
        print(
            f"{experiment['name']:<34}"
            f"{macro['accuracy']:>7.2f}"
            f"{macro['precision']:>8.2f}"
            f"{macro['recall']:>8.2f}"
            f"{macro['f1']:>8.2f}"
            f"{macro['mcc']:>8.2f}"
        )
    print()


def print_detailed_metrics(experiment: dict[str, object]) -> None:
    print(f"Detailed Metrics - {experiment['name']}")
    print("-" * 88)
    macro = experiment["test_metrics"]["macro"]
    print(
        "Accuracy={accuracy:.2f} Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f} "
        "Sensitivity={sensitivity:.2f} Specificity={specificity:.2f} FPR={fpr:.2f} "
        "FNR={fnr:.2f} NPV={npv:.2f} FDR={fdr:.2f} MCC={mcc:.2f}".format(**macro)
    )
    print("Best hyperparameters:", experiment["best_params"])
    print("Confusion Matrix (rows=actual, cols=predicted)")
    matrix = experiment["test_metrics"]["matrix"]
    header = "actual\\pred".ljust(12) + "".join(label.ljust(10) for label in EMOTIONS)
    print(header)
    for actual in EMOTIONS:
        row = actual.ljust(12) + "".join(str(matrix[actual][pred]).ljust(10) for pred in EMOTIONS)
        print(row)
    print()


def show_sample_conversations(bot: EmotionChatbot) -> None:
    print("Sample Conversations")
    print("-" * 88)
    samples = [
        "I am so proud because my project presentation went really well today",
        "I feel lonely because my friend stopped replying to my messages",
        "I am angry that my manager ignored my work again",
        "I am scared about my health report tomorrow",
        "Wow, I did not expect this sudden offer at all",
    ]
    for text in samples:
        result = bot.respond(text)
        print(f"User      : {text}")
        print(f"Emotion   : {result['emotion']} | domain: {result['domain']} | confidence: {result['confidence']}")
        print(f"Chatbot   : {result['response']}")
        print()


def interactive_chat(bot: EmotionChatbot) -> None:
    print("Interactive Mode")
    print("-" * 88)
    print("Type a message. Enter 'bye', 'exit', or 'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Bot: Please type a message.\n")
            continue
        if user_input.lower() in {"bye", "exit", "quit"}:
            print("Bot: Take care. Thank you for trying the chatbot.")
            break

        result = bot.respond(user_input)
        print(f"Bot emotion : {result['emotion']} | domain: {result['domain']} | confidence: {result['confidence']}")
        print(f"Bot reply   : {result['response']}\n")


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    splits, dataset_source = discover_dataset(project_dir)

    experiment_configs = [
        {
            "display_name": "Naive Bayes + BoW",
            "feature_name": "BoW",
            "feature_name_key": "bow",
            "model_name": "naive_bayes",
            "param_grid": [
                {"ngram_range": (1, 1), "min_df": 1, "max_features": 20000, "alpha": 0.3},
                {"ngram_range": (1, 2), "min_df": 2, "max_features": 30000, "alpha": 0.5},
                {"ngram_range": (1, 2), "min_df": 1, "max_features": 40000, "alpha": 0.8},
                {"ngram_range": (1, 3), "min_df": 2, "max_features": 50000, "alpha": 1.0},
            ],
        },
        {
            "display_name": "Logistic Regression + TF-IDF",
            "feature_name": "TF-IDF",
            "feature_name_key": "tfidf",
            "model_name": "logistic_regression",
            "param_grid": [
                {"ngram_range": (1, 2), "min_df": 2, "max_features": 40000, "sublinear_tf": True, "C": 2.0, "max_iter": 400},
                {"ngram_range": (1, 2), "min_df": 1, "max_features": 50000, "sublinear_tf": True, "C": 4.0, "max_iter": 500},
                {"ngram_range": (1, 3), "min_df": 2, "max_features": 60000, "sublinear_tf": True, "C": 3.0, "max_iter": 600},
                {"ngram_range": (1, 2), "min_df": 3, "max_features": 30000, "sublinear_tf": True, "C": 1.0, "max_iter": 400},
            ],
        },
        {
            "display_name": "Linear SVM + TF-IDF",
            "feature_name": "TF-IDF",
            "feature_name_key": "tfidf",
            "model_name": "linear_svm",
            "param_grid": [
                {"ngram_range": (1, 2), "min_df": 2, "max_features": 40000, "sublinear_tf": True, "C": 1.0, "max_iter": 4000},
                {"ngram_range": (1, 2), "min_df": 1, "max_features": 50000, "sublinear_tf": True, "C": 2.0, "max_iter": 5000},
                {"ngram_range": (1, 3), "min_df": 2, "max_features": 60000, "sublinear_tf": True, "C": 1.5, "max_iter": 5000},
                {"ngram_range": (1, 2), "min_df": 3, "max_features": 30000, "sublinear_tf": True, "C": 0.75, "max_iter": 4000},
            ],
        },
    ]

    experiments = [run_experiment(config, splits) for config in experiment_configs]
    experiments.sort(key=lambda item: item["val_macro_f1"], reverse=True)
    best_experiment = experiments[0]
    chatbot = EmotionChatbot(best_experiment["vectorizer"], best_experiment["model"], best_experiment["name"])

    save_report_artifacts(project_dir, dataset_source, splits, experiments)

    print_header(dataset_source)
    print_dataset_summary(splits)
    print_result_table(experiments)
    print_detailed_metrics(best_experiment)
    show_sample_conversations(chatbot)
    chatbot.memory.clear()
    interactive_chat(chatbot)


if __name__ == "__main__":
    main()
