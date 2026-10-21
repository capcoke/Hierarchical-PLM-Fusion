import os
import random
import time
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.autograd import Variable

from EC_contrastive import *


ROOT = Path(__file__).resolve().parent
MAIN_PROJECT_ROOT = ROOT.parent.parent
DATA_PATH = str(MAIN_PROJECT_ROOT / "Dataset" / "training_data") + "/"
MODEL_DIR = ROOT / "Model"
MODEL_STEM = "SCREEN_fixed_best_combo_3_3_3-7_feature_fusion"
VALIDATION_START_EPOCH = int(os.getenv("SCREEN_VALIDATION_START_EPOCH", "15"))
EARLY_STOP_PATIENCE = int(os.getenv("SCREEN_EARLY_STOP_PATIENCE", "10"))
EARLY_STOP_MIN_EPOCHS = int(os.getenv("SCREEN_EARLY_STOP_MIN_EPOCHS", str(VALIDATION_START_EPOCH + 8)))
NUM_WORKERS = int(os.getenv("SCREEN_NUM_WORKERS", "0" if os.name == "nt" else "2"))
TORCH_NUM_THREADS = int(os.getenv("SCREEN_TORCH_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
BATCH_LOG_INTERVAL = int(os.getenv("SCREEN_BATCH_LOG_INTERVAL", "100"))
SCREEN_SEED = int(os.getenv("SCREEN_SEED", "11"))
SCREEN_SEEDS = os.getenv("SCREEN_SEEDS", "")
THRESHOLD_STEPS = int(os.getenv("SCREEN_THRESHOLD_STEPS", "201"))
EMA_ENABLED = os.getenv("SCREEN_EMA", "1") == "1"
EMA_DECAY = float(os.getenv("SCREEN_EMA_DECAY", "0.995"))
EMA_START_EPOCH = int(os.getenv("SCREEN_EMA_START_EPOCH", "0"))
SUMMARY_PATH = ROOT / "training_summary.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_model_names(seed):
    suffix = f"_seed{seed}"
    model_stem = MODEL_STEM + suffix
    return {
        "stem": model_stem,
        "best_auprc": model_stem + "_best_auprc.pkl",
        "best_f1": model_stem + "_best_f1.pkl",
        "legacy": model_stem + ".pkl",
    }


class ModelEma:
    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(model)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            ema_state = self.module.state_dict()
            model_state = model.state_dict()
            for key, ema_value in ema_state.items():
                model_value = model_state[key].detach()
                if not torch.is_floating_point(ema_value):
                    ema_value.copy_(model_value)
                else:
                    ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)


class DualMetricEarlyStopping:
    def __init__(self, model_names, patience=10, verbose=True, min_delta=1e-6, min_epochs=0):
        self.model_names = model_names
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_auprc = None
        self.best_f1 = None
        self.early_stop = False

    def __call__(self, result_f1, result_auprc, model, epoch):
        improved = False

        if self.best_auprc is None or result_auprc > self.best_auprc + self.min_delta:
            old_value = 0.0 if self.best_auprc is None else self.best_auprc
            self.best_auprc = result_auprc
            self.save_checkpoint(model, self.model_names["best_auprc"], "AUPRC", old_value, result_auprc)
            torch.save(model.state_dict(), str(MODEL_DIR / self.model_names["legacy"]))
            improved = True

        if self.best_f1 is None or result_f1 > self.best_f1 + self.min_delta:
            old_value = 0.0 if self.best_f1 is None else self.best_f1
            self.best_f1 = result_f1
            self.save_checkpoint(model, self.model_names["best_f1"], "F1", old_value, result_f1)
            improved = True

        if improved:
            self.counter = 0
        else:
            if epoch + 1 < self.min_epochs:
                return
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, filename, metric_name, old_value, new_value):
        if self.verbose:
            print(f"Validation {metric_name} improved ({old_value:.4f} --> {new_value:.4f}). Saving model ...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(MODEL_DIR / filename))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, data_loader, ema_model=None, epoch=0):
    epoch_loss_train = 0.0
    n = 0
    for batch_idx, data in enumerate(data_loader, start=1):
        model.optimizer.zero_grad()
        _, _, labels, _, prot5_feature, esm2_layer28_feature, esm2_layer33_feature = data

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
        loss.backward()
        model.optimizer.step()
        if ema_model is not None and epoch >= EMA_START_EPOCH:
            ema_model.update(model)

        epoch_loss_train += loss.item()
        n += 1

        if BATCH_LOG_INTERVAL > 0 and batch_idx % BATCH_LOG_INTERVAL == 0:
            print(f"  batch {batch_idx}/{len(data_loader)} | avg_loss_so_far: {epoch_loss_train / n:.7f}", flush=True)

    return epoch_loss_train / max(n, 1)


def evaluate(model, data_loader):
    model.eval()
    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []

    for data in data_loader:
        with torch.no_grad():
            _, _, labels, _, prot5_feature, esm2_layer28_feature, esm2_layer33_feature = data

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
            y_true_np = y_true.cpu().numpy()
            valid_pred += [pred[1] for pred in y_prob]
            valid_true += list(y_true_np)

            epoch_loss += loss.item()
            n += 1

    return epoch_loss / max(n, 1), valid_true, valid_pred


def analysis(y_true, y_pred, best_threshold=None):
    if best_threshold is None:
        best_f1 = 0.0
        best_threshold = 0.0
        for step_idx in range(THRESHOLD_STEPS):
            threshold = step_idx / max(THRESHOLD_STEPS - 1, 1)
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


def train_model(train_dataframe, valid_dataframe, data_path, seed):
    print("Train on", str(train_dataframe.shape[0]), "samples, validate on", str(valid_dataframe.shape[0]), "samples")
    print("DataLoader workers:", NUM_WORKERS)
    print("Fusion mode: hierarchical_adaptive_logit_fusion")
    print("ESM2 init weights:", [0.5, 0.5])
    print("Final init weights:", [PROT5_LOGIT_WEIGHT, ESM2_LOGIT_WEIGHT])
    print("Conv stage preset:", CONV_STAGE_PRESET or "custom/default")
    print("Conv count tag:", CURRENT_CONV_COUNT_TAG)
    print("Conv counts:", CURRENT_CONV_COUNTS)
    print("Kernel configs:", DEFAULT_BRANCH_KERNELS)
    print("EMA enabled:", EMA_ENABLED)
    print("EMA decay:", EMA_DECAY)
    print("Threshold steps:", THRESHOLD_STEPS)
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }
    train_loader = DataLoader(dataset=ProDataset(train_dataframe, data_path), shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(dataset=ProDataset(valid_dataframe, data_path), shuffle=False, **loader_kwargs)

    model = SCREEN(NLAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT)
    if torch.cuda.is_available():
        model.cuda()
    ema_model = ModelEma(model, EMA_DECAY) if EMA_ENABLED else None
    model_names = build_model_names(seed)

    print("Training from scratch without warm start.")
    early_stopping = DualMetricEarlyStopping(
        model_names=model_names,
        patience=EARLY_STOP_PATIENCE,
        verbose=True,
        min_epochs=EARLY_STOP_MIN_EPOCHS,
    )
    history = []

    for epoch in range(NUMBER_EPOCHS):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()
        epoch_time = time.time()

        epoch_loss_train_avg = train_one_epoch(model, train_loader, ema_model=ema_model, epoch=epoch)
        print("Epoch: {} cost time: {}, train_loss: {}".format(epoch + 1, time.time() - epoch_time, epoch_loss_train_avg))

        if epoch >= VALIDATION_START_EPOCH:
            print("========== Evaluate Valid set ==========")
            eval_model = ema_model.module if ema_model is not None and epoch >= EMA_START_EPOCH else model
            epoch_loss_valid_avg, valid_true, valid_pred = evaluate(eval_model, valid_loader)
            result_valid = analysis(valid_true, valid_pred)

            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(epoch + 1, epoch_loss_train_avg, epoch_loss_valid_avg))
            print("Valid binary acc: ", result_valid["binary_acc"])
            print("Valid precision: ", result_valid["precision"])
            print("Valid recall: ", result_valid["recall"])
            print("Valid f1: ", result_valid["f1"])
            print("Valid AUC: ", result_valid["AUC"])
            print("Valid AUPRC: ", result_valid["AUPRC"])
            print("Valid mcc: ", result_valid["mcc"])
            print("Valid threshold: ", result_valid["threshold"])
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss_train_avg,
                    "valid_loss": epoch_loss_valid_avg,
                    "f1": result_valid["f1"],
                    "auprc": result_valid["AUPRC"],
                    "threshold": result_valid["threshold"],
                }
            )

            early_stopping(result_valid["f1"], result_valid["AUPRC"], eval_model, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return {
        "seed": seed,
        "model_names": model_names,
        "best_f1": early_stopping.best_f1,
        "best_auprc": early_stopping.best_auprc,
        "history": history,
    }


def parse_seed_list():
    if not SCREEN_SEEDS.strip():
        return [SCREEN_SEED]
    return [int(item.strip()) for item in SCREEN_SEEDS.split(",") if item.strip()]


def main():
    ids, sequences, labels, ec1_list = [], [], [], []
    data_path = DATA_PATH

    with open(data_path + "training_id_withEC.txt", "r") as f:
        for line in f.readlines():
            protein = line.strip()
            pdb_id = protein[:4].lower() + "-" + protein[5].upper()
            ids.append(pdb_id)
    prot_num = len(ids)

    prot_seq = {}
    prot_anno = {}
    with open(data_path + "training_label.txt", "r") as f:
        data = f.readlines()
    for line in range(0, len(data)):
        if data[line].startswith(">"):
            protein = data[line].lstrip(">").strip()
            pro, chain = protein.split("-") if "-" in protein else protein.split("_")
            pdb_id = pro.lower() + "-" + chain.upper()
            prot_seq[pdb_id] = data[line + 1].strip()
            prot_anno[pdb_id] = data[line + 2].strip()

    for prot in ids:
        seq = prot_seq[prot]
        label = prot_anno[prot]
        sequences.append(seq)
        labels.append([int(label[i]) for i in range(len(label))])

    pdb2ec1 = np.load(data_path + "training_PDB2EC1.npy", allow_pickle=True).item()
    for prot in ids:
        ec1_list.append(get_EC1_num(pdb2ec1[prot]))

    data_dic = {"ID": ids, "sequence": sequences, "label": labels, "EC1": ec1_list}
    data_dataframe = pd.DataFrame(data_dic)

    train_dataframe = data_dataframe.iloc[:int(prot_num * 0.9)]
    valid_dataframe = data_dataframe.iloc[int(prot_num * 0.9):]
    torch.set_num_threads(TORCH_NUM_THREADS)
    print("Torch CPU threads:", TORCH_NUM_THREADS)
    print("Batch log interval:", BATCH_LOG_INTERVAL)
    print("Training data path:", data_path)
    seed_list = parse_seed_list()
    print("Seed list:", seed_list)
    print("Validation start epoch:", VALIDATION_START_EPOCH)
    print("Early stop patience:", EARLY_STOP_PATIENCE)
    print("Early stop min epochs:", EARLY_STOP_MIN_EPOCHS)

    summaries = []
    for seed in seed_list:
        print("\n##############################")
        print("Starting training with seed:", seed)
        print("##############################")
        set_seed(seed)
        summaries.append(train_model(train_dataframe, valid_dataframe, data_path, seed))

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print("Training summary saved to:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
