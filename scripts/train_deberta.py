#!/usr/bin/env python3
"""Fine-tune DeBERTa for balanced satisfiability classification."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


LABEL_TO_ID = {"unsatisfiable": 0, "satisfiable": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_text(row: dict[str, object]) -> str:
    return " ".join(str(sentence) for sentence in row["sentences"])


class SatDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, object]],
        tokenizer: object,
        max_length: int,
        split_name: str,
    ) -> None:
        self.rows = rows
        self.encodings = tokenizer(
            [row_text(row) for row in rows],
            truncation=False,
            padding=False,
        )
        lengths = [len(ids) for ids in self.encodings["input_ids"]]
        max_observed = max(lengths, default=0)
        over_limit = sum(length > max_length for length in lengths)
        print(
            json.dumps(
                {
                    "split": split_name,
                    "token_length_min": min(lengths, default=0),
                    "token_length_max": max_observed,
                    "examples_over_max_length": over_limit,
                    "configured_max_length": max_length,
                    "truncation": False,
                }
            ),
            flush=True,
        )
        if over_limit:
            raise RuntimeError(
                f"{split_name}: {over_limit} examples exceed {max_length} tokens; "
                "refusing to truncate because truncation can change satisfiability."
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = LABEL_TO_ID[str(self.rows[index]["label"])]
        item["row_index"] = index
        return item


class BatchCollator:
    def __init__(self, tokenizer: object) -> None:
        self.base = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    def __call__(self, features: list[dict[str, object]]) -> dict[str, object]:
        row_indices = [int(feature.pop("row_index")) for feature in features]
        batch = self.base(features)
        batch["row_indices"] = torch.tensor(row_indices, dtype=torch.long)
        return batch


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, object]:
    tp = sum(gold == 1 and pred == 1 for gold, pred in zip(labels, predictions))
    tn = sum(gold == 0 and pred == 0 for gold, pred in zip(labels, predictions))
    fp = sum(gold == 0 and pred == 1 for gold, pred in zip(labels, predictions))
    fn = sum(gold == 1 and pred == 0 for gold, pred in zip(labels, predictions))
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "total": len(labels),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "true_unsat_pred_unsat": tn,
            "true_unsat_pred_sat": fp,
            "true_sat_pred_unsat": fn,
            "true_sat_pred_sat": tp,
        },
        "predicted_labels": dict(Counter(ID_TO_LABEL[prediction] for prediction in predictions)),
        "gold_labels": dict(Counter(ID_TO_LABEL[label] for label in labels)),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    rows: list[dict[str, object]],
    device: torch.device,
    use_bf16: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    predictions_by_index: dict[int, int] = {}
    labels_by_index: dict[int, int] = {}

    for batch in loader:
        row_indices = batch.pop("row_indices").tolist()
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            output = model(**batch)
        total_loss += float(output.loss) * len(row_indices)
        predictions = output.logits.argmax(dim=-1).cpu().tolist()
        labels = batch["labels"].cpu().tolist()
        for index, prediction, label in zip(row_indices, predictions, labels):
            predictions_by_index[index] = prediction
            labels_by_index[index] = label

    ordered_indices = sorted(predictions_by_index)
    predictions = [predictions_by_index[index] for index in ordered_indices]
    labels = [labels_by_index[index] for index in ordered_indices]
    result = binary_metrics(labels, predictions)
    result["loss"] = round(total_loss / len(rows), 6)
    result["by_fragment"] = {}

    # Derive fragment names from the evaluated rows so this supports both
    # the original E0/E1/E2 dataset and the paper S/W/V dataset.
    fragments = sorted({str(row["fragment"]) for row in rows})
    for fragment in fragments:
        selected = [index for index in ordered_indices if rows[index]["fragment"] == fragment]
        fragment_labels = [labels_by_index[index] for index in selected]
        fragment_predictions = [predictions_by_index[index] for index in selected]
        result["by_fragment"][fragment] = binary_metrics(fragment_labels, fragment_predictions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", "--dev", dest="validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many consecutive epochs without validation accuracy improvement; 0 disables it.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this training job.")
    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(args.train))
    validation_rows = read_jsonl(Path(args.validation))
    test_rows = read_jsonl(Path(args.test))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_config = AutoConfig.from_pretrained(args.model)
    if args.dropout is not None:
        model_config.hidden_dropout_prob = args.dropout
        model_config.attention_probs_dropout_prob = args.dropout
        model_config.classifier_dropout = args.dropout
    model_config.num_labels = 2
    model_config.id2label = ID_TO_LABEL
    model_config.label2id = LABEL_TO_ID
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=model_config,
    ).to(device)

    train_dataset = SatDataset(train_rows, tokenizer, args.max_length, "train")
    validation_dataset = SatDataset(validation_rows, tokenizer, args.max_length, "validation")
    test_dataset = SatDataset(test_rows, tokenizer, args.max_length, "test")
    collator = BatchCollator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader_args = {
        "batch_size": args.batch_size * 2,
        "shuffle": False,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    validation_loader = DataLoader(validation_dataset, **eval_loader_args)
    test_loader = DataLoader(test_dataset, **eval_loader_args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimization_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = optimization_steps_per_epoch * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    history: list[dict[str, object]] = []
    best_validation_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "model": args.model,
                "train_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "test_examples": len(test_rows),
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": (
                    args.batch_size * args.gradient_accumulation_steps
                ),
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping_patience,
                "bf16": use_bf16,
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            batch.pop("row_indices")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(output.loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_loss={running_loss / step:.6f}",
                    flush=True,
                )

        validation_result = evaluate(
            model, validation_loader, validation_rows, device, use_bf16
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": round(running_loss / len(train_loader), 6),
            "validation": validation_result,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, indent=2), flush=True)

        if float(validation_result["accuracy"]) > best_validation_accuracy:
            best_validation_accuracy = float(validation_result["accuracy"])
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}: validation accuracy did not improve for "
                    f"{epochs_without_improvement} consecutive epoch(s).",
                    flush=True,
                )
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir / "best_model").to(device)
    final_validation = evaluate(
        best_model, validation_loader, validation_rows, device, use_bf16
    )
    final_test = evaluate(best_model, test_loader, test_rows, device, use_bf16)
    final_train = evaluate(best_model, train_loader, train_rows, device, use_bf16)
    results = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "hyperparameters": {
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "bf16": use_bf16,
            "seconds": round(time.time() - start_time, 2),
        },
        "history": history,
        "train": final_train,
        "validation": final_validation,
        "test": final_test,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "train": final_train,
                "validation": final_validation,
                "test": final_test,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
