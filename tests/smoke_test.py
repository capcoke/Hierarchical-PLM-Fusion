"""Synthetic model check; no dataset, checkpoint, optimizer step, or downloads."""

import ast
import importlib.util
import json
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    path = root / "src" / "proposed" / "EC_contrastive.py"
    spec = importlib.util.spec_from_file_location("proposed_release_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = module.SCREEN(
        module.NLAYER, module.INPUT_DIM, module.HIDDEN_DIM,
        module.NUM_CLASSES, module.DROPOUT,
    ).cpu().eval()
    inputs = (torch.randn(8, 1024), torch.randn(8, 1280), torch.randn(8, 1280))
    logits, aux = model(*inputs)
    assert logits.shape == (8, 2), logits.shape
    assert torch.isfinite(logits).all()
    probabilities = torch.softmax(logits / 8, dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(8))
    for key in ("esm2_feature_weights", "final_feature_weights"):
        weights = aux[key]
        assert weights.shape == (8, 2), (key, weights.shape)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(8))
    parameters = tuple(model.parameters())
    loss = model.criterion(logits, torch.arange(8) % 2)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    total = sum(p.numel() for p in parameters)
    active = sum(p.numel() for p, g in zip(parameters, gradients) if g is not None)
    assert total == 9800344, total
    assert active == 9208268, active
    assert all(torch.isfinite(g).all() for g in gradients if g is not None)
    print(json.dumps({"status": "PASS", "output_shape": list(logits.shape),
                      "total_parameters": total, "loss_path_parameters": active,
                      "optimizer_step_executed": False}, indent=2))


if __name__ == "__main__":
    main()
