import importlib.util
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class Tokenizer:
    def __init__(self, max_words=30000, max_length=80):
        self.max_words = max_words
        self.max_length = max_length
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts):
        counts = Counter()
        for text in texts:
            counts.update(str(text).lower().split())
        for token, _ in counts.most_common(self.max_words - 2):
            self.word_to_id.setdefault(token, len(self.word_to_id))

    def encode(self, text):
        ids = [self.word_to_id.get(token, 1) for token in str(text).lower().split()[:self.max_length]]
        return ids + [0] * (self.max_length - len(ids))


def run_bilstm(splits, seed=42, max_train=10000, epochs=3):
    """Train a small BiLSTM per dataset. Returns results and status tables."""
    if importlib.util.find_spec("torch") is None:
        return pd.DataFrame(), pd.DataFrame([{"Model": "BiLSTM", "Status": "N/A – not executed", "Detail": "PyTorch is not installed"}])

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, statuses = [], []

    for dataset, split in splits.items():
        train = split["train"].copy()
        if len(train) > max_train:
            train = train.sample(max_train, random_state=seed)
        labels = sorted(train["label"].unique())
        label_to_id = {label: index for index, label in enumerate(labels)}
        tokenizer = Tokenizer()
        tokenizer.fit(train["clean_text"])
        x = np.asarray([tokenizer.encode(text) for text in train["clean_text"]])
        y = np.asarray([label_to_id[label] for label in train["label"]])
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.15, random_state=seed, stratify=y)
        test = split["test"]
        x_test = np.asarray([tokenizer.encode(text) for text in test["clean_text"]])
        y_test = np.asarray([label_to_id[label] for label in test["label"]])

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(len(tokenizer.word_to_id), 64, padding_idx=0)
                self.lstm = nn.LSTM(64, 48, batch_first=True, bidirectional=True)
                self.dropout = nn.Dropout(.3)
                self.output = nn.Linear(96, len(labels))

            def forward(self, batch):
                _, (hidden, _) = self.lstm(self.embedding(batch))
                hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
                return self.output(self.dropout(hidden))

        model = Model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        train_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=128, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=256)
        started = time.perf_counter()
        best_state, best_val = None, -1
        for _ in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                loss_fn(model(batch_x.to(device)), batch_y.to(device)).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_pred = torch.cat([model(batch_x.to(device)).argmax(1).cpu() for batch_x, _ in val_loader]).numpy()
            val_f1 = f1_score(y_val, val_pred, average="macro")
            if val_f1 > best_val:
                best_val = val_f1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_pred = model(torch.tensor(x_test).to(device)).argmax(1).cpu().numpy()
        results.append({
            "Dataset": dataset, "Model": "BiLSTM", "Validation Macro F1": best_val,
            "Test Macro F1": f1_score(y_test, test_pred, average="macro"),
            "Test Weighted F1": f1_score(y_test, test_pred, average="weighted"),
            "Test Accuracy": accuracy_score(y_test, test_pred),
            "Training Time": time.perf_counter() - started,
            "Parameters": sum(parameter.numel() for parameter in model.parameters()),
        })
        statuses.append({"Dataset": dataset, "Model": "BiLSTM", "Status": "Executed", "Detail": f"Device: {device}"})
    return pd.DataFrame(results), pd.DataFrame(statuses)


def transformer_status(splits):
    available = all(importlib.util.find_spec(name) is not None for name in ("transformers", "datasets", "accelerate"))
    detail = "Dependencies available; enable Transformer training in the pipeline." if available else "Install requirements-advanced.txt to run Transformer models."
    status = "Ready" if available else "N/A – not executed"
    return pd.DataFrame([
        {"Dataset": dataset, "Model": model, "Status": status, "Detail": detail}
        for dataset in splits for model in ("DistilBERT", "Twitter-RoBERTa")
    ])


def run_transformers(splits, seed=42, max_train=2000, epochs=1, max_length=128):
    """Fine-tune DistilBERT and Twitter-RoBERTa for every supplied dataset."""
    required = ("transformers", "datasets", "accelerate")
    if not all(importlib.util.find_spec(name) is not None for name in required):
        return pd.DataFrame(), transformer_status(splits)

    import inspect
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(seed)
    model_specs = {
        "DistilBERT": "distilbert-base-uncased",
        "Twitter-RoBERTa": "cardiffnlp/twitter-roberta-base-sentiment",
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    argument_parameters = inspect.signature(TrainingArguments.__init__).parameters
    results, statuses = [], []

    for dataset, split in splits.items():
        train = split["train"].copy()
        if len(train) > max_train:
            train = train.sample(max_train, random_state=seed)
        train, validation = train_test_split(
            train, test_size=.15, random_state=seed, stratify=train["label"]
        )
        test = split["test"]
        labels = sorted(train["label"].unique())
        label_to_id = {label: index for index, label in enumerate(labels)}
        id_to_label = {index: label for label, index in label_to_id.items()}

        for model_label, checkpoint in model_specs.items():
            started = time.perf_counter()
            try:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

                def make_dataset(frame):
                    raw = Dataset.from_dict({
                        "text": frame["clean_text"].astype(str).tolist(),
                        "labels": [label_to_id[label] for label in frame["label"]],
                    })

                    def tokenize(batch):
                        return tokenizer(batch["text"], truncation=True, max_length=max_length)

                    return raw.map(tokenize, batched=True, remove_columns=["text"])

                train_dataset = make_dataset(train)
                validation_dataset = make_dataset(validation)
                test_dataset = make_dataset(test)
                model = AutoModelForSequenceClassification.from_pretrained(
                    checkpoint,
                    num_labels=len(labels),
                    label2id=label_to_id,
                    id2label=id_to_label,
                    ignore_mismatched_sizes=True,
                )
                output_dir = f"/tmp/sentiment_{model_label}_{len(results)}"
                arguments = {
                    "output_dir": output_dir,
                    "num_train_epochs": epochs,
                    "per_device_train_batch_size": 8,
                    "per_device_eval_batch_size": 16,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "logging_strategy": "no",
                    "save_strategy": "no",
                    "report_to": [],
                    "seed": seed,
                }
                if "eval_strategy" in argument_parameters:
                    arguments["eval_strategy"] = "epoch"
                elif "evaluation_strategy" in argument_parameters:
                    arguments["evaluation_strategy"] = "epoch"
                arguments = {key: value for key, value in arguments.items() if key in argument_parameters}
                training_args = TrainingArguments(**arguments)

                def compute_metrics(prediction):
                    logits = prediction.predictions
                    predictions = np.argmax(logits, axis=-1)
                    return {
                        "accuracy": accuracy_score(prediction.label_ids, predictions),
                        "macro_f1": f1_score(prediction.label_ids, predictions, average="macro", zero_division=0),
                        "weighted_f1": f1_score(prediction.label_ids, predictions, average="weighted", zero_division=0),
                    }

                trainer_kwargs = {
                    "model": model,
                    "args": training_args,
                    "train_dataset": train_dataset,
                    "eval_dataset": validation_dataset,
                    "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
                    "compute_metrics": compute_metrics,
                }
                if "processing_class" in trainer_parameters:
                    trainer_kwargs["processing_class"] = tokenizer
                elif "tokenizer" in trainer_parameters:
                    trainer_kwargs["tokenizer"] = tokenizer
                trainer = Trainer(**trainer_kwargs)
                trainer.train()
                validation_metrics = trainer.evaluate(validation_dataset)
                test_output = trainer.predict(test_dataset)
                test_pred = np.argmax(test_output.predictions, axis=-1)
                test_true = np.asarray(test["label"].map(label_to_id))
                results.append({
                    "Dataset": dataset,
                    "Model": model_label,
                    "Model Identifier": checkpoint,
                    "Validation Macro F1": validation_metrics.get("eval_macro_f1", np.nan),
                    "Test Macro F1": f1_score(test_true, test_pred, average="macro", zero_division=0),
                    "Test Weighted F1": f1_score(test_true, test_pred, average="weighted", zero_division=0),
                    "Test Accuracy": accuracy_score(test_true, test_pred),
                    "Training Time": time.perf_counter() - started,
                    "Parameters": sum(parameter.numel() for parameter in model.parameters()),
                })
                statuses.append({"Dataset": dataset, "Model": model_label, "Status": "Executed", "Detail": checkpoint})
            except Exception as error:
                statuses.append({
                    "Dataset": dataset, "Model": model_label, "Status": "N/A – not executed",
                    "Detail": f"{type(error).__name__}: {str(error)[:300]}",
                })
    return pd.DataFrame(results), pd.DataFrame(statuses)
