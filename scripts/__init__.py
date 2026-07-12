"""QS-Net PyTorch modules extracted from training-qs-net-pytorch_v2.0.ipynb."""

from scripts.circuit import build_forward_circuit, create_quantum_device, initialize_weights
from scripts.conformal import calibrate_threshold, nonconformity_score
from scripts.constants import DEFAULT_CF, DEFAULT_NOISE_RATE, ZERO_DAY
from scripts.data import (
    balanced_sample,
    class_balance_table,
    load_split,
    plot_class_balance_pie,
    stratified_head,
)
from scripts.inference import estimate_lipschitz, predict_batch, qsnet_infer
from scripts.logging import to_jsonable, write_history_log
from scripts.loss import (
    ce_loss_term,
    compute_l_ce,
    compute_l_inter,
    compute_l_intra,
    gradient_variance,
    inter_loss_term,
    intra_loss_term,
    maqt_loss,
)
from scripts.prototypes import PrototypeBank, compute_prototypes, prototype_summary
from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import expectations_to_tensor, get_torch_device, to_np_x, to_torch_x

__all__ = [
    "DEFAULT_CF",
    "DEFAULT_NOISE_RATE",
    "PrototypeBank",
    "ZERO_DAY",
    "balanced_sample",
    "build_forward_circuit",
    "calibrate_threshold",
    "ce_loss_term",
    "class_balance_table",
    "compute_l_ce",
    "compute_l_inter",
    "compute_l_intra",
    "compute_prototypes",
    "create_quantum_device",
    "estimate_lipschitz",
    "expectations_to_tensor",
    "fidelity",
    "get_torch_device",
    "gradient_variance",
    "initialize_weights",
    "inter_loss_term",
    "intra_loss_term",
    "load_split",
    "maqt_loss",
    "nonconformity_score",
    "plot_class_balance_pie",
    "predict_batch",
    "prototype_summary",
    "qsnet_infer",
    "stratified_head",
    "to_jsonable",
    "to_np_x",
    "to_torch_x",
    "trace_distance",
    "write_history_log",
]
