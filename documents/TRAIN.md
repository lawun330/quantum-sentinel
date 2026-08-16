# Training Settings

One-place reference for MAQT and VQC training configs from the final notebooks.

- MAQT loss: $L = L_{CE} + \lambda_1 L_{intra} + \lambda_2 L_{inter}$
    - $L_{inter} = -\mathrm{mean}\,D_{\mathrm{tr}}$ to other-class prototypes
    - **Inter-margin** and **hardest-only** are currently unused
- VQC loss: $L = L_{CE}$

---

| FROZEN Dataset | Arch | Target | Known classes | Features | Train data | Val data | PCA | Qubits | Layers | $p$ | Loss | $\lambda_1$ | $\lambda_2$ | Inter-margin | Hardest-only | $\lambda$ warmup | LR start | LR schedule | Epochs max | Batch | Weighted sampler | Early stop | Patience | `min_delta` | ES metric | Seed | Prototypes | Notebook | Notes |
|---------|------|--------|---------------|----------|------------|----------|-----|--------|--------|-----|------|-------------|-------------|--------------|--------------|------------------|----------|-------------|---------|-------|------------------|------------|----------|-------------|-----------|------|------------|----------|-------|
| BoT-IoT | MAQT | `label_multiclass` | 4 (DDoS, DoS, Normal, Reconnaissance) | Team C FROZEN | full | stratified 10% from train | yes (0.95) | 5 | 2 | 0.01 | MAQT | 0.5 | 0.1 | — | — | curriculum `warmup_frac=0.2` | 0.01 | cosine | 50 | 64 | yes | yes | 5 | 0.001 | val MAQT loss | 42 | EMA + exact at end | `final-bot-iot-maqt-train.ipynb` | ~68% CE full-test (no reject) |
| BoT-IoT | VQC | `label_multiclass` | 4 (same) | Team C FROZEN | full | stratified 10% from train | yes (0.95) | 5 | 2 | 0.01 | CE only | — | — | — | — | — | 0.01 | cosine | 50 | 64 | yes | yes | 5 | 0.001 | val CE loss | 42 | post-hoc `PrototypeBank` | `final-bot-iot-vqc-train.ipynb` | fair ablation twin |
| UNSW-NB15 | MAQT | `label_multiclass` | 8 (Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance) | Team C FROZEN | full | stratified 10% from train | yes (0.95) | 4 | 2 | 0.01 | MAQT | 0.66 | 0.33 | — | — | curriculum `warmup_frac=0.2` | 0.01 | cosine | 50 | 64 | yes | yes | 5 | 0.001 | val MAQT loss | 42 | EMA + exact at end | `final-unsw-nb15-maqt-train.ipynb` | $L_{inter}$ fights with $L_{CE}$ |
| UNSW-NB15 | VQC | `label_multiclass` | 8 (same) | Team C FROZEN | full | stratified 10% from train | yes (0.95) | 4 | 2 | 0.01 | CE only | — | — | — | — | — | 0.01 | cosine | 50 | 64 | yes | yes | 5 | 0.001 | val CE loss | 42 | post-hoc `PrototypeBank` | `final-unsw-nb15-vqc-train.ipynb` | fair ablation twin |
| CICIoT2023 | MAQT | `label_multiclass` | 31 (Backdoor_Malware, ..., XSS) | Team C FROZEN | DPP subset, 1000/class | — | yes (0.95) | 6 | 3 | 0.01 | MAQT | 0.5 | 0.66 | — | — | curriculum `warmup_frac=0.2` | 0.005 | cosine | 15 (resume) | 8 | yes | yes | 5 | 0.001 | train MAQT loss | 42 | EMA + exact at end | `cicio-training.ipynb` | HP-tuned resume from `maqt-latest.pt` |
| CICIoT2023 | VQC | — | — | — | — | — | — | — | — | — | CE only | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | not shipped yet |
