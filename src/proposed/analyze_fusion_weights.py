import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from EC_contrastive import *


ROOT = Path(__file__).resolve().parent
MAIN_PROJECT_ROOT = ROOT.parent.parent
MODEL_PATH = ROOT / "Model"
MODEL_STEM = "SCREEN_fixed_best_combo_3_3_3-7_feature_fusion"
NUM_WORKERS = int(os.getenv("SCREEN_NUM_WORKERS", "0" if os.name == "nt" else "2"))
TORCH_NUM_THREADS = int(os.getenv("SCREEN_TORCH_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
SCREEN_TEST_SEED = os.getenv("SCREEN_TEST_SEED", "").strip()
SCREEN_CHECKPOINT_TAG = os.getenv("SCREEN_CHECKPOINT_TAG", "").strip().lower()
FUSION_WEIGHT_OUTPUT_PATH = os.getenv("SCREEN_FUSION_WEIGHT_OUTPUT_PATH", "").strip()

MODEL_PATH.mkdir(parents=True, exist_ok=True)

WEIGHT_NAME_MAP = {
    "esm2_feature_weights": ["esm2_layer28", "esm2_layer33"],
    "final_feature_weights": ["prot5", "esm2_fused"],
    "flat_feature_weights": ["prot5", "esm2_layer28", "esm2_layer33"],
}


def build_checkpoint_names():
    suffix = f"_seed{SCREEN_TEST_SEED}" if SCREEN_TEST_SEED else ""
    model_stem = MODEL_STEM + suffix
    return {
        "best_auprc": model_stem + "_best_auprc.pkl",
        "best_f1": model_stem + "_best_f1.pkl",
    }


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


def load_test_dataframe(dataset):
    data_dir = str(MAIN_PROJECT_ROOT / "Dataset" / dataset) + "/"
    with open(data_dir + "test-" + dataset + "_id.txt", "r") as handle:
        protein_list = [line.strip() for line in handle.readlines()]

    prot_seq = {}
    prot_anno = {}
    with open(data_dir + dataset + "_enzyme_label.txt", "r") as handle:
        data = handle.readlines()
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

    dataframe = pd.DataFrame({"ID": protein_list, "sequence": sequences, "label": labels})
    return dataframe, data_dir


def _init_weight_store():
    return {}


def _update_weight_store(store, aux_outputs):
    for key, labels in WEIGHT_NAME_MAP.items():
        if key not in aux_outputs:
            continue
        weights = aux_outputs[key].detach().cpu().numpy()
        if weights.ndim != 2:
            continue
        if key not in store:
            store[key] = {
                "labels": labels,
                "residue_sum": np.zeros(weights.shape[1], dtype=np.float64),
                "residue_count": 0,
                "protein_means": [],
            }
        store[key]["residue_sum"] += weights.sum(axis=0)
        store[key]["residue_count"] += weights.shape[0]
        store[key]["protein_means"].append(weights.mean(axis=0))


def _finalize_weight_store(store):
    summary = {}
    for key, value in store.items():
        protein_means = np.stack(value["protein_means"], axis=0) if value["protein_means"] else np.zeros((0, len(value["labels"])))
        residue_mean = (value["residue_sum"] / max(value["residue_count"], 1)).tolist()
        protein_mean = protein_means.mean(axis=0).tolist() if len(protein_means) else [0.0] * len(value["labels"])
        protein_std = protein_means.std(axis=0).tolist() if len(protein_means) else [0.0] * len(value["labels"])
        summary[key] = {
            "labels": value["labels"],
            "num_proteins": int(len(value["protein_means"])),
            "num_residues": int(value["residue_count"]),
            "residue_mean": residue_mean,
            "protein_mean": protein_mean,
            "protein_std": protein_std,
        }
    return summary


def collect_dataset_weight_store(model, test_dataframe, data_path):
    loader = DataLoader(
        dataset=EnzDataset(test_dataframe, data_path),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )
    subset_store = _init_weight_store()
    for data in loader:
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

            del y_true
            prot5_feature = torch.squeeze(prot5_feature)
            esm2_layer28_feature = torch.squeeze(esm2_layer28_feature)
            esm2_layer33_feature = torch.squeeze(esm2_layer33_feature)
            _, aux_outputs = model(prot5_feature, esm2_layer28_feature, esm2_layer33_feature)
            _update_weight_store(subset_store, aux_outputs)
    return subset_store


def analyze_checkpoint(model, dataset_name, test_dataframe, data_path):
    subset_store = collect_dataset_weight_store(model, test_dataframe, data_path)
    subset_summary = _finalize_weight_store(subset_store)
    print(f"========== Dataset: {dataset_name} ==========")
    for key, item in subset_summary.items():
        readable = ", ".join(
            f"{label}={mean:.4f}" for label, mean in zip(item["labels"], item["residue_mean"])
        )
        print(f"{key}: {readable}")
    return subset_summary, subset_store


def main():
    torch.set_num_threads(TORCH_NUM_THREADS)
    print("DataLoader workers:", NUM_WORKERS)
    print("Torch CPU threads:", TORCH_NUM_THREADS)
    print("MODEL_STEM:", MODEL_STEM)
    print("SCREEN_TEST_SEED:", SCREEN_TEST_SEED or "default")

    checkpoints = build_checkpoint_names()
    if SCREEN_CHECKPOINT_TAG:
        if SCREEN_CHECKPOINT_TAG not in checkpoints:
            raise ValueError(f"Unknown SCREEN_CHECKPOINT_TAG: {SCREEN_CHECKPOINT_TAG}")
        selected = {SCREEN_CHECKPOINT_TAG: checkpoints[SCREEN_CHECKPOINT_TAG]}
    else:
        selected = checkpoints

    datasets = ["NN", "HA_superfamily", "EF_superfamily", "PC", "EF_fold"]
    summary = {
        "model_stem": MODEL_STEM,
        "screen_test_seed": SCREEN_TEST_SEED or None,
        "selected_checkpoints": list(selected.keys()),
        "datasets": datasets,
        "results": {},
    }

    for tag, checkpoint_name in selected.items():
        model = SCREEN(NLAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
        if torch.cuda.is_available():
            model.cuda()
        checkpoint_path = MODEL_PATH / checkpoint_name
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
        model.eval()

        print(f"\n===== Analyze checkpoint [{tag}] {checkpoint_name} =====")
        checkpoint_summary = {
            "checkpoint_name": checkpoint_name,
            "datasets": {},
        }
        overall_store = _init_weight_store()

        for dataset in datasets:
            dataframe, data_dir = load_test_dataframe(dataset)
            subset_summary, subset_store = analyze_checkpoint(model, dataset, dataframe, data_dir)
            checkpoint_summary["datasets"][dataset] = subset_summary
            for key, item in subset_store.items():
                if key not in overall_store:
                    overall_store[key] = {
                        "labels": item["labels"],
                        "residue_sum": np.zeros(len(item["labels"]), dtype=np.float64),
                        "residue_count": 0,
                        "protein_means": [],
                    }
                overall_store[key]["residue_sum"] += item["residue_sum"]
                overall_store[key]["residue_count"] += item["residue_count"]
                overall_store[key]["protein_means"].extend(item["protein_means"])

        checkpoint_summary["overall"] = _finalize_weight_store(overall_store)
        summary["results"][tag] = checkpoint_summary

    output_path = Path(FUSION_WEIGHT_OUTPUT_PATH) if FUSION_WEIGHT_OUTPUT_PATH else ROOT / "fusion_weight_summary.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("\nFusion weight summary saved to:", output_path)


if __name__ == "__main__":
    main()
