import os
import itertools

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

INPUT_DIM = 64
HIDDEN_DIM = 512
NLAYER = 3
DROPOUT = 0.1
LEARNING_RATE = 5e-5
BATCH_SIZE = 1
NUM_CLASSES = 2
NUMBER_EPOCHS = int(os.getenv("SCREEN_EPOCHS", "120"))
PROT5_INPUT_DIM = int(os.getenv("SCREEN_PROT5_INPUT_DIM", "1024"))
ESM2_INPUT_DIM = int(os.getenv("SCREEN_ESM2_INPUT_DIM", "1280"))
PROT5_LOGIT_WEIGHT = float(os.getenv("SCREEN_PROT5_LOGIT_WEIGHT", "0.4"))
ESM2_LOGIT_WEIGHT = float(os.getenv("SCREEN_ESM2_LOGIT_WEIGHT", str(1.0 - PROT5_LOGIT_WEIGHT)))
MAX_SEQUENCE_LENGTH = int(os.getenv("SCREEN_MAX_SEQUENCE_LENGTH", "1000"))
FUSION_HIDDEN_DIM = int(os.getenv("SCREEN_FUSION_HIDDEN_DIM", "32"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BRANCH_CONFIGS = {
    "prot5": {
        "env_subdir": "SCREEN_PROT5_SUBDIR",
        "env_h5": "SCREEN_PROT5_H5",
        "default_h5": "train_per_residue_embeddings.h5",
        "candidates": ["Prot5"],
    },
    "esm2_layer28": {
        "env_subdir": "SCREEN_ESM2_LAYER28_SUBDIR",
        "env_h5": "SCREEN_ESM2_LAYER28_H5",
        "default_h5": "train_per_residue_embeddings.h5",
        "candidates": [
            "ESM2_layer28",
            "ESM2_layer_28",
            "esm2_layer28",
            os.path.join("ESM2", "layer28"),
        ],
    },
    "esm2_layer33": {
        "env_subdir": "SCREEN_ESM2_LAYER33_SUBDIR",
        "env_h5": "SCREEN_ESM2_LAYER33_H5",
        "default_h5": "train_per_residue_embeddings.h5",
        "candidates": [
            "ESM2_layer33",
            "ESM2_layer_33",
            "ESM2_final",
            "esm2_layer33",
            os.path.join("ESM2", "layer33"),
            os.path.join("ESM2", "final"),
        ],
    },
}


CONV_STAGE_PRESET = os.getenv("SCREEN_CONV_STAGE_PRESET", "").strip().lower()
BASE_KERNEL_SET = (3, 7, 11)
FIXED_BRANCH_KERNELS = {
    "prot5": (3,),
    "esm2_layer28": (3,),
    "esm2_layer33": (3, 7),
}


def _parse_kernel_list(raw_value, fallback):
    value = (raw_value or "").strip()
    if not value:
        return tuple(fallback)
    kernels = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        kernel = int(item)
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError(f"Kernel sizes must be positive odd integers, got: {kernel}")
        kernels.append(kernel)
    if not kernels:
        raise ValueError("At least one kernel size must remain enabled.")
    return tuple(kernels)


def _parse_conv_count(raw_value, fallback):
    value = (raw_value or "").strip()
    if not value:
        return fallback
    count = int(value)
    if count not in (1, 2, 3):
        raise ValueError("Conv count mode must be one of 1, 2, 3.")
    return count


def _filter_default_kernels(enable3, enable7, enable11):
    kernels = []
    if enable3 == "1":
        kernels.append(3)
    if enable7 == "1":
        kernels.append(7)
    if enable11 == "1":
        kernels.append(11)
    if not kernels:
        raise ValueError("All default kernels 3/7/11 are disabled. Enable at least one kernel.")
    return tuple(kernels)


def _build_stage_preset():
    # stage1: Prot5 1 conv, ESM28 1 conv, ESM33 1 conv
    if CONV_STAGE_PRESET == "stage1_single_all":
        return {
            "prot5": (7,),
            "esm2_layer28": (7,),
            "esm2_layer33": (7,),
        }
    # stage2: Prot5 1 conv, ESM28/33 3 convs
    if CONV_STAGE_PRESET == "stage2_esm_three_only":
        return {
            "prot5": (7,),
            "esm2_layer28": (3, 7, 11),
            "esm2_layer33": (3, 7, 11),
        }
    # stage3: Prot5 3 convs, ESM28/33 3 convs
    if CONV_STAGE_PRESET == "stage3_all_three":
        return {
            "prot5": (3, 7, 11),
            "esm2_layer28": (3, 7, 11),
            "esm2_layer33": (3, 7, 11),
        }
    return None


def _parse_combo_index(raw_value, fallback):
    value = (raw_value or "").strip()
    if not value:
        return fallback
    index = int(value)
    if index < 1:
        raise ValueError("Combination index must be >= 1.")
    return index


def _resolve_kernel_combo(count_mode, combo_index):
    combos = list(itertools.combinations(BASE_KERNEL_SET, count_mode))
    if combo_index > len(combos):
        raise ValueError(
            f"Combination index {combo_index} exceeds available {count_mode}-kernel combinations ({len(combos)})."
        )
    return combos[combo_index - 1]


def _build_count_preset():
    prot5_count = _parse_conv_count(os.getenv("SCREEN_PROT5_CONV_COUNT", ""), 3)
    esm28_count = _parse_conv_count(os.getenv("SCREEN_ESM2_LAYER28_CONV_COUNT", ""), 3)
    esm33_count = _parse_conv_count(os.getenv("SCREEN_ESM2_LAYER33_CONV_COUNT", ""), 3)
    prot5_combo = _parse_combo_index(os.getenv("SCREEN_PROT5_COMBO_INDEX", ""), 1)
    esm28_combo = _parse_combo_index(os.getenv("SCREEN_ESM2_LAYER28_COMBO_INDEX", ""), 1)
    esm33_combo = _parse_combo_index(os.getenv("SCREEN_ESM2_LAYER33_COMBO_INDEX", ""), 1)
    return {
        "prot5": _resolve_kernel_combo(prot5_count, prot5_combo),
        "esm2_layer28": _resolve_kernel_combo(esm28_count, esm28_combo),
        "esm2_layer33": _resolve_kernel_combo(esm33_count, esm33_combo),
    }


def _build_branch_kernels():
    preset_kernels = _build_stage_preset()
    count_preset = _build_count_preset()
    default_kernels = _filter_default_kernels(
        os.getenv("SCREEN_ENABLE_KERNEL3", "1"),
        os.getenv("SCREEN_ENABLE_KERNEL7", "1"),
        os.getenv("SCREEN_ENABLE_KERNEL11", "1"),
    )
    branch_kernels = {
        "prot5": _parse_kernel_list(
            os.getenv("SCREEN_PROT5_KERNELS", ""),
            preset_kernels["prot5"] if preset_kernels else FIXED_BRANCH_KERNELS["prot5"],
        ),
        "esm2_layer28": _parse_kernel_list(
            os.getenv("SCREEN_ESM2_LAYER28_KERNELS", ""),
            preset_kernels["esm2_layer28"] if preset_kernels else FIXED_BRANCH_KERNELS["esm2_layer28"],
        ),
        "esm2_layer33": _parse_kernel_list(
            os.getenv("SCREEN_ESM2_LAYER33_KERNELS", ""),
            preset_kernels["esm2_layer33"] if preset_kernels else FIXED_BRANCH_KERNELS["esm2_layer33"],
        ),
    }
    if not preset_kernels and not any(
        os.getenv(name, "").strip()
        for name in ("SCREEN_PROT5_KERNELS", "SCREEN_ESM2_LAYER28_KERNELS", "SCREEN_ESM2_LAYER33_KERNELS")
    ):
        return FIXED_BRANCH_KERNELS
    if not preset_kernels and not any(
        os.getenv(name, "").strip()
        for name in (
            "SCREEN_PROT5_CONV_COUNT",
            "SCREEN_PROT5_COMBO_INDEX",
            "SCREEN_ESM2_LAYER28_CONV_COUNT",
            "SCREEN_ESM2_LAYER28_COMBO_INDEX",
            "SCREEN_ESM2_LAYER33_CONV_COUNT",
            "SCREEN_ESM2_LAYER33_COMBO_INDEX",
        )
    ):
        return branch_kernels
    return branch_kernels


DEFAULT_BRANCH_KERNELS = _build_branch_kernels()
CURRENT_CONV_COUNTS = {
    branch_name: len(kernel_sizes) for branch_name, kernel_sizes in DEFAULT_BRANCH_KERNELS.items()
}
CURRENT_CONV_COUNT_TAG = "p{0}_{1}_e28{2}_{3}_e33{4}_{5}".format(
    CURRENT_CONV_COUNTS["prot5"],
    "-".join(str(k) for k in DEFAULT_BRANCH_KERNELS["prot5"]),
    CURRENT_CONV_COUNTS["esm2_layer28"],
    "-".join(str(k) for k in DEFAULT_BRANCH_KERNELS["esm2_layer28"]),
    CURRENT_CONV_COUNTS["esm2_layer33"],
    "-".join(str(k) for k in DEFAULT_BRANCH_KERNELS["esm2_layer33"]),
)


def get_EC1_num(ec_onehot):
    ec1 = list(int(i) for i in ec_onehot)
    ec_index = ec1.index(1.0)
    if ec_index == 0:
        return 7
    if ec_index == 1:
        return 3
    if ec_index == 2:
        return 4
    if ec_index == 3:
        return 5
    if ec_index == 4:
        return 6
    if ec_index == 5:
        return 1
    return 2


def apply_sequence_truncation(array_like):
    if len(array_like) <= MAX_SEQUENCE_LENGTH:
        return array_like
    head = MAX_SEQUENCE_LENGTH // 2
    tail = MAX_SEQUENCE_LENGTH - head
    return array_like[:head] + array_like[-tail:]


def apply_feature_truncation(feature):
    if feature.shape[0] <= MAX_SEQUENCE_LENGTH:
        return feature
    head = MAX_SEQUENCE_LENGTH // 2
    tail = MAX_SEQUENCE_LENGTH - head
    return np.concatenate([feature[:head], feature[-tail:]], axis=0)


def align_feature_length(feature, target_len):
    current_len = feature.shape[0]
    if current_len == target_len:
        return feature
    if current_len > target_len:
        return feature[:target_len]
    padding = np.zeros((target_len - current_len, feature.shape[1]), dtype=feature.dtype)
    return np.concatenate([feature, padding], axis=0)


def resolve_branch_h5_path(data_path, branch_name):
    config = BRANCH_CONFIGS[branch_name]
    data_root = data_path.rstrip("/\\")
    override_subdir = os.getenv(config["env_subdir"], "").strip()
    subdir_candidates = [override_subdir] if override_subdir else config["candidates"]
    h5_name = os.getenv(config["env_h5"], config["default_h5"]).strip() or config["default_h5"]

    checked_paths = []
    for subdir in subdir_candidates:
        candidate_path = os.path.join(data_root, subdir, h5_name)
        checked_paths.append(candidate_path)
        if os.path.exists(candidate_path):
            return candidate_path

    checked = "\n".join(checked_paths)
    raise FileNotFoundError(
        "Missing {} residue embedding file. Checked:\n{}".format(branch_name, checked)
    )


def load_residue_embedding(sequence_name, data_path, branch_name, sequence_length=None):
    h5_path = resolve_branch_h5_path(data_path, branch_name)
    with h5py.File(h5_path, "r") as handle:
        feature = handle[sequence_name][:].astype(np.float32)
    feature = apply_feature_truncation(feature)
    if sequence_length is not None:
        feature = align_feature_length(feature, sequence_length)
    return feature


class ProDataset(Dataset):
    def __init__(self, dataframe, data_path):
        self.names = dataframe["ID"].values
        self.sequences = dataframe["sequence"].values
        self.labels = dataframe["label"].values
        self.ec1 = dataframe["EC1"].values if "EC1" in dataframe.columns else None
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
        ec1 = -1 if self.ec1 is None else int(self.ec1[index])
        return sequence_name, sequence, label, np.int64(ec1), prot5_feature, esm2_layer28_feature, esm2_layer33_feature

    def __len__(self):
        return len(self.labels)


class MultiScaleBranchEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, kernel_sizes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.kernel_sizes = tuple(kernel_sizes)
        self.importance_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, residue_feature):
        base = F.relu(self.input_proj(F.dropout(residue_feature, self.dropout, training=self.training)))
        seq = base.transpose(0, 1).unsqueeze(0)
        multi_scale = sum(F.relu(conv(seq)) for conv in self.convs) / float(len(self.convs))
        multi_scale = multi_scale.squeeze(0).transpose(0, 1)
        residue_importance = self.importance_gate(base)
        gated_multi_scale = multi_scale * residue_importance
        fusion_weight = self.fusion_gate(torch.cat([base, gated_multi_scale], dim=1))
        fused = base + fusion_weight * gated_multi_scale
        return self.out_norm(fused)


class ResidueClassifier(nn.Module):
    def __init__(self, hidden_dim, nfeat, nclass, dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, nfeat),
            nn.ReLU(),
            nn.Linear(nfeat, nclass),
        )

    def forward(self, residue_features):
        return self.classifier(residue_features)


class AdaptiveTwoBranchFusion(nn.Module):
    def __init__(self, input_dim, init_probs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(FUSION_HIDDEN_DIM, 2),
        )
        self._init_gate_bias(init_probs)

    def _init_gate_bias(self, init_probs):
        final_linear = self.gate[-1]
        nn.init.zeros_(final_linear.weight)
        base_probs = torch.tensor(init_probs, dtype=torch.float32)
        final_linear.bias.data.copy_(torch.log(base_probs))

    def forward(self, left_logits, right_logits):
        branch_logits = torch.stack([left_logits, right_logits], dim=1)
        gate_input = torch.cat([left_logits, right_logits], dim=-1)
        fusion_scores = self.gate(gate_input)
        fusion_weights = torch.softmax(fusion_scores, dim=-1)
        fused_logits = (branch_logits * fusion_weights.unsqueeze(-1)).sum(dim=1)
        return fused_logits, fusion_weights


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, hidden_dim, init_probs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self._init_gate_bias(init_probs)

    def _init_gate_bias(self, init_probs):
        final_linear = self.gate[-1]
        nn.init.zeros_(final_linear.weight)
        base_probs = torch.tensor(init_probs, dtype=torch.float32)
        final_linear.bias.data.copy_(torch.log(base_probs))

    def forward(self, left_feature, right_feature):
        pair_feature = torch.cat([left_feature, right_feature], dim=-1)
        fusion_scores = self.gate(pair_feature)
        fusion_weights = torch.softmax(fusion_scores, dim=-1)
        branch_features = torch.stack([left_feature, right_feature], dim=1)
        fused_feature = (branch_features * fusion_weights.unsqueeze(-1)).sum(dim=1)
        return self.out_norm(fused_feature), fusion_weights


class SCREEN(nn.Module):
    def __init__(self, nlayers, nfeat, hidden_dim, nclass, dropout):
        super().__init__()
        del nlayers
        self.prot5_encoder = MultiScaleBranchEncoder(
            PROT5_INPUT_DIM, hidden_dim, dropout, kernel_sizes=DEFAULT_BRANCH_KERNELS["prot5"]
        )
        self.esm2_layer28_encoder = MultiScaleBranchEncoder(
            ESM2_INPUT_DIM, hidden_dim, dropout, kernel_sizes=DEFAULT_BRANCH_KERNELS["esm2_layer28"]
        )
        self.esm2_layer33_encoder = MultiScaleBranchEncoder(
            ESM2_INPUT_DIM, hidden_dim, dropout, kernel_sizes=DEFAULT_BRANCH_KERNELS["esm2_layer33"]
        )

        self.prot5_classifier = ResidueClassifier(hidden_dim, nfeat, nclass, dropout)
        self.esm2_layer28_classifier = ResidueClassifier(hidden_dim, nfeat, nclass, dropout)
        self.esm2_layer33_classifier = ResidueClassifier(hidden_dim, nfeat, nclass, dropout)
        self.esm2_feature_fusion = AdaptiveFeatureFusion(hidden_dim, init_probs=[0.5, 0.5])
        self.final_feature_fusion = AdaptiveFeatureFusion(
            hidden_dim, init_probs=[PROT5_LOGIT_WEIGHT, ESM2_LOGIT_WEIGHT]
        )
        self.esm2_feature_classifier = ResidueClassifier(hidden_dim, nfeat, nclass, dropout)
        self.feature_fused_classifier = ResidueClassifier(hidden_dim, nfeat, nclass, dropout)
        self.esm2_fusion = AdaptiveTwoBranchFusion(
            input_dim=NUM_CLASSES * 2,
            init_probs=[0.5, 0.5],
        )
        self.final_fusion = AdaptiveTwoBranchFusion(
            input_dim=NUM_CLASSES * 2,
            init_probs=[PROT5_LOGIT_WEIGHT, ESM2_LOGIT_WEIGHT],
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

    def forward(self, prot5_feature, esm2_layer28_feature, esm2_layer33_feature):
        prot5_residue = self.prot5_encoder(prot5_feature.float())
        esm2_layer28_residue = self.esm2_layer28_encoder(esm2_layer28_feature.float())
        esm2_layer33_residue = self.esm2_layer33_encoder(esm2_layer33_feature.float())
        esm2_feature_residue, esm2_feature_weights = self.esm2_feature_fusion(
            esm2_layer28_residue,
            esm2_layer33_residue,
        )
        final_feature_residue, final_feature_weights = self.final_feature_fusion(
            prot5_residue,
            esm2_feature_residue,
        )

        prot5_logits = self.prot5_classifier(prot5_residue)
        esm2_layer28_logits = self.esm2_layer28_classifier(esm2_layer28_residue)
        esm2_layer33_logits = self.esm2_layer33_classifier(esm2_layer33_residue)
        esm2_feature_logits = self.esm2_feature_classifier(esm2_feature_residue)
        feature_fused_logits = self.feature_fused_classifier(final_feature_residue)

        esm2_fused_logits, esm2_fusion_weights = self.esm2_fusion(
            esm2_layer28_logits,
            esm2_layer33_logits,
        )
        logit_fused_logits, fusion_weights = self.final_fusion(
            prot5_logits,
            esm2_fused_logits,
        )
        final_logits = feature_fused_logits

        aux_outputs = {
            "prot5_logits": prot5_logits,
            "esm2_layer28_logits": esm2_layer28_logits,
            "esm2_layer33_logits": esm2_layer33_logits,
            "esm2_feature_logits": esm2_feature_logits,
            "feature_fused_logits": feature_fused_logits,
            "esm2_fused_logits": esm2_fused_logits,
            "logit_fused_logits": logit_fused_logits,
            "esm2_fusion_weights": esm2_fusion_weights,
            "final_fusion_weights": fusion_weights,
            "esm2_feature_weights": esm2_feature_weights,
            "final_feature_weights": final_feature_weights,
            "prot5_residue": prot5_residue,
            "esm2_layer28_residue": esm2_layer28_residue,
            "esm2_layer33_residue": esm2_layer33_residue,
            "esm2_feature_residue": esm2_feature_residue,
            "final_feature_residue": final_feature_residue,
        }
        return final_logits, aux_outputs
