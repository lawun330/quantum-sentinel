"""Shared constants for QS-Net training and inference."""

# training
DEFAULT_SEED = 42                       # (data.py, hilbert.py)
DEFAULT_BATCH_SIZE = 64                 # (inference.py, hilbert.py, conformal.py, prototypes.py, attacks.py)
DEFAULT_LR = 0.05                       # learning rate # (train.py)
DEFAULT_EPOCHS = 10                     # epochs # (train.py)
DEFAULT_GRAD_CLIP_NORM = 1.0            # gradient clip norm # (train.py)

# plotting
DEFAULT_MIN_PCT = 5.0                   # pie chart label cutoff % # (data.py)

# algorithm 1: circuit
DEFAULT_NOISE_RATE = 0.01               # depolarizing noise # (circuit.py, inference.py)
DEFAULT_CF = 1.0                        # confidence factor # (inference.py)
DEFAULT_WEIGHT_INIT_EPS = 1e-2          # random-init epsilon # (circuit.py, gradient.py)
DEFAULT_REUPLOAD = True                 # data re-uploading flag # (circuit.py)

# algorithm 1: curriculum / MAQT loss
DEFAULT_FOCAL = False                   # focal loss flag # (loss.py)
DEFAULT_FOCAL_GAMMA = 2.0               # focal loss gamma # (loss.py)
DEFAULT_LAMBDA1 = 0.5                   # L_intra weight # (loss.py, gradient.py)
DEFAULT_LAMBDA2 = 0.3                   # L_inter weight # (loss.py, gradient.py)
DEFAULT_WARMUP_FRAC = 0.2               # curriculum lambda warmup fraction # (loss.py)

# algorithm 1: prototypes / gradients
BARREN_PLATEAU_VAR_THRESHOLD = 1e-6     # barren-plateau threshold # (sweep.py)
DEFAULT_EMA_MOMENTUM = 0.9              # EMA momentum # (prototypes.py)
DEFAULT_N_TRIALS = 30                   # barren-plateau probe trials # (gradient.py)

# algorithm 2
DEFAULT_ALPHA = 0.05                    # conformal miscoverage # (conformal.py)

# algorithm 3
ZERO_DAY = -1                           # zero-day / reject label # (inference.py)
DEFAULT_N_PROBE = 30                    # Lipschitz probe count # (inference.py)
DEFAULT_DELTA = 0.05                    # Lipschitz perturbation # (inference.py)

# H1 diagnostics
DEFAULT_EPS = 1e-7                      # fidelity clamp epsilon # (quantum_metrics.py)

# attacks / robustness ablation
DEFAULT_CONTROL_BATCH_SIZE = 16         # plain-CE baseline batch size # (attacks.py)

# logging
DEFAULT_LOG_DIR = "logs"                # (logging.py)
DEFAULT_SWEEP_LOG_DIR = DEFAULT_LOG_DIR + "/sweeps"  # (logging.py)

# theory for Propositions
INPUT_DIM_D = 8                         # nominal feature dimension d (Section 1 conventions) # (theory.py)

DEFAULT_MIX = 0.5                       # random density matrix mixing factor (Section 2, Proposition 1) # (theory.py)
DEFAULT_N_PAIRS = 64                    # Proposition 1 numerical pairs (Section 2, Proposition 1) # (theory.py)
DEFAULT_P_VALS = (0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0) # Proposition 1 p values (Section 2, Proposition 1) # (theory.py)
PROP1_RESIDUAL_TOL = 1e-10              # Proposition 1 exact-identity tolerance (Section 2, Proposition 1) # (theory.py)

DEFAULT_L_PHI = 0.5                     # fallback L_phi constant value (Section 4) # (theory.py)
# DEFAULT_L_PHI = num_layers/2.0        # analytic L_phi bound (Section 4) # (theory.py)

DEFAULT_BETA = 0.05                     # quantile-relaxed zero-day fraction (Section 3.5, Corollary 1) # (theory.py)

DEFAULT_LOWER_PERCENTILE = 5.0          # lower percentile for robust F_in # (theory.py)
DEFAULT_UPPER_PERCENTILE = 97.0         # upper percentile for robust F_out # (theory.py)

DEFAULT_CV = 5                          # cross-validation fold count # (theory.py)
DEFAULT_MAX_ITER = 1000                 # maximum number of iterations for logistic regression # (theory.py)