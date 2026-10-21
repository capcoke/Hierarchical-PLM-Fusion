import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.autograd import Variable

from EC_contrastive import *


ROOT = Path(__file__).resolve().parent
MAIN_PROJECT_ROOT = ROOT.parent.parent
MODEL_PATH = ROOT / "Model"
MODEL_STEM = "SCREEN_fixed_best_combo_3_3_3-7_feature_fusion"
NUM_WORKERS = int(os.getenv("SCREEN_NUM_WORKERS", "0" if os.name == "nt" else "2"))
TORCH_NUM_THREADS = int(os.getenv("SCREEN_TORCH_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
SCREEN_TEST_SEED = os.getenv("SCREEN_TEST_SEED", "11")
SCREEN_TEST_SUMMARY_PATH = os.getenv("SCREEN_TEST_SUMMARY_PATH", "").strip()

MODEL_PATH.mkdir(parents=True, exist_ok=True)


def build_checkpoint_names():
    suffix = f"_seed{SCREEN_TEST_SEED}" if SCREEN_TEST_SEED else ""
    model_stem = MODEL_STEM + suffix
    return [
        (model_stem + "_best_auprc.pkl", "best_auprc"),
        (model_stem + "_best_f1.pkl", "best_f1"),
    ]


class EnzDataset(Dataset):
    def __init__(self, dataframe, data_path):
        self.names = dataframe["ID"].values
        self.sequences = dataframe["sequence"].values
        self.labels = dataframe["label"].values
        self.data_path = data_path

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = apply_sequence_truncation(self.sequences[index])
        label = np.array(apply_sequence_truncation(self.labels[index]), dtype=np.int64)
        prot5_feature = load_residue_embedding(sequence_name, self.data_path, "prot5", sequence_length=len(label))
        esm2_layer28_feature = load_residue_embedding(
            sequence_name, self.data_path, "esm2_layer28", sequence_length=len(label)
        )
        esm2_layer33_feature = load_residue_embedding(
            sequence_name, self.data_path, "esm2_layer33", sequence_length=len(label)
        )
        return sequence_name, sequence, label, prot5_feature, esm2_layer28_feature, esm2_layer33_feature

    def __len__(self):
        return len(self.labels)


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []

    for data in data_loader:
        with torch.no_grad():
            _, _, labels, prot5_feature, esm2_layer28_feature, esm2_layer33_feature = data

            if torch.cuda.is_available():
                prot5_feature = Variable(prot5_feature.cuda(non_blocking=True))
                esm2_layer28_feature = Variable(esm2_layer28_feature.cuda(non_blocking=True))
                esm2_layer33_feature = Variable(esm2_layer33_feature.cuda(non_blocking=True))
                y_true = Variable(labels.cuda(non_blocking=True)).long()
            else:
                prot5_feature = Variable(prot5_feature)
                esm2_layer28_feature = Variable(esm2_layer28_feature)
                esm2_layer33_feature = Variable(esm2_layer33_feature)
                y_true = Variable(labels).long()

            prot5_feature = torch.squeeze(prot5_feature)
            esm2_layer28_feature = torch.squeeze(esm2_layer28_feature)
            esm2_layer33_feature = torch.squeeze(esm2_layer33_feature)
            y_true = torch.squeeze(y_true)

            y_pred, _ = model(prot5_feature, esm2_layer28_feature, esm2_layer33_feature)
            loss = model.criterion(y_pred, y_true)
            y_prob = torch.softmax(y_pred / 8, dim=1).cpu().numpy()
            y_true_np = y_true.cpu().numpy().tolist()

            valid_pred += [pred[1] for pred in y_prob]
            valid_true += list(y_true_np)
            epoch_loss += loss.item()
            n += 1

    return epoch_loss / max(n, 1), valid_true, valid_pred


def analysis(y_true, y_pred, best_threshold=None):
    if best_threshold is None:
        best_f1 = 0.0
        best_threshold = 0.0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            f1 = metrics.f1_score(y_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_pred)
    return {
        "binary_acc": metrics.accuracy_score(y_true, binary_pred),
        "precision": metrics.precision_score(y_true, binary_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, binary_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, binary_pred, zero_division=0),
        "AUC": metrics.roc_auc_score(y_true, y_pred),
        "AUPRC": metrics.auc(recalls, precisions),
        "mcc": metrics.matthews_corrcoef(y_true, binary_pred),
        "threshold": best_threshold,
    }


def run_single_checkpoint(test_dataframe, data_path, checkpoint_name, tag):
    test_loader = DataLoader(
        dataset=EnzDataset(test_dataframe, data_path),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )
    model = SCREEN(NLAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
    if torch.cuda.is_available():
        model.cuda()

    checkpoint_path = MODEL_PATH / checkpoint_name
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    _, test_true, test_pred = evaluate(model, test_loader)
    result_test = analysis(test_true, test_pred)

    print(f"[{tag}] checkpoint: {checkpoint_name}")
    print("Test recall: ", result_test["recall"])
    print("Test precision:", result_test["precision"])
    print("Test f1: ", result_test["f1"])
    print("Test mcc: ", result_test["mcc"])
    print("Test AUC: ", result_test["AUC"])
    print("Test AUPRC: ", result_test["AUPRC"])
    print("Threshold: ", result_test["threshold"])
    return result_test


def main():
    torch.set_num_threads(TORCH_NUM_THREADS)
    print("DataLoader workers:", NUM_WORKERS)
    print("Torch CPU threads:", TORCH_NUM_THREADS)
    print("Fusion mode: hierarchical_adaptive_logit_fusion")
    print("ESM2 init weights:", [0.5, 0.5])
    print("Final init weights:", [PROT5_LOGIT_WEIGHT, ESM2_LOGIT_WEIGHT])
    print("Conv stage preset:", CONV_STAGE_PRESET or "custom/default")
    print("Conv count tag:", CURRENT_CONV_COUNT_TAG)
    print("Conv counts:", CURRENT_CONV_COUNTS)
    print("Kernel configs:", DEFAULT_BRANCH_KERNELS)
    print("SCREEN_TEST_SEED:", SCREEN_TEST_SEED or "default")

    datasets = ["NN", "HA_superfamily", "EF_superfamily", "PC", "EF_fold"]
    checkpoints = build_checkpoint_names()

    summary = {
        "model_stem": MODEL_STEM,
        "screen_test_seed": SCREEN_TEST_SEED or None,
        "conv_stage_preset": CONV_STAGE_PRESET or None,
        "conv_count_tag": CURRENT_CONV_COUNT_TAG,
        "conv_counts": CURRENT_CONV_COUNTS,
        "kernel_configs": {key: list(value) for key, value in DEFAULT_BRANCH_KERNELS.items()},
        "checkpoints": [],
    }

    checkpoint_results = {}
    for checkpoint_name, tag in checkpoints:
        checkpoint_results[tag] = {
            "checkpoint_name": checkpoint_name,
            "tag": tag,
            "datasets": {},
        }

    for dataset in datasets:
        data_dir = str(MAIN_PROJECT_ROOT / "Dataset" / dataset) + "/"
        with open(data_dir + "test-" + dataset + "_id.txt", "r") as f:
            protein_list = [line.strip() for line in f.readlines()]

        prot_seq = {}
        prot_anno = {}
        with open(data_dir + dataset + "_enzyme_label.txt", "r") as f:
            data = f.readlines()
        for line in range(0, len(data)):
            if data[line].startswith(">"):
                protein = data[line].lstrip(">").strip()
                pdbid, chain = protein.split("-") if "-" in protein else protein.split("_")
                pro = pdbid.lower() + "-" + chain.upper()
                prot_seq[pro] = data[line + 1].strip()
                prot_anno[pro] = data[line + 2].strip()

        sequences = []
        labels = []
        for prot in protein_list:
            sequences.append(prot_seq[prot])
            label = prot_anno[prot]
            labels.append([int(label[i]) for i in range(len(label))])

        test_dataframe = pd.DataFrame({"ID": protein_list, "sequence": sequences, "label": labels})
        print("========== Dataset:", dataset, "==========")
        for checkpoint_name, tag in checkpoints:
            result = run_single_checkpoint(test_dataframe, data_dir, checkpoint_name, tag)
            checkpoint_results[tag]["datasets"][dataset] = result

    for _, tag in checkpoints:
        dataset_results = checkpoint_results[tag]["datasets"]
        avg_f1 = sum(item["f1"] for item in dataset_results.values()) / max(len(dataset_results), 1)
        checkpoint_results[tag]["avg_f1"] = avg_f1
        summary["checkpoints"].append(checkpoint_results[tag])

    summary["best_checkpoint_by_avg_f1"] = max(summary["checkpoints"], key=lambda item: item["avg_f1"])

    if SCREEN_TEST_SUMMARY_PATH:
        summary_path = Path(SCREEN_TEST_SUMMARY_PATH)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print("Test summary saved to:", summary_path)


if __name__ == "__main__":
    main()
