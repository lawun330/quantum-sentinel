# Expanding README.md File Structure

## Legacy Notebooks

```
/
└── legacy_notebooks/
    │
    ├── prelegacy_q8_pipeline/  # PRE-LEGACY NOTEBOOKS
    │   │
    │   ├── e2e-qs-net_v1.0.ipynb       # PennyLane             ## TON-IoT
    │   │
    │   ├── ./                          # PennyLane + PyTorch   ## TON IoT
    │   │   ├── e2e-qs-net_v2.0.ipynb   ## eps curve uses LAST minibatch
    │   │   │                           ## measure grad DURING weight update, averages over eps
    │   │   │
    │   │   ├── e2e-qs-net_v2.1.ipynb   ## eps curve uses LAST minibatch 
    │   │   │                           ## measure grad DURING weight update, averages over eps
    │   │   │
    │   │   └── e2e-qs-net_v2.2.ipynb   ## eps curve averages ALL minibatches
    │   │                               ## measure grad AFTER weight update, averages over eps
    │   │                               ## added logging ## utilize "legacy_scripts/"
    │   │
    │   └── e2e-qs-net_v3.0.ipynb       # class-weighted 200 CICIoT2023                        ## 30 eps, 64 b_sz
    │
    └── legacy_q8_pipeline/     # day (11-14, 16-19, 22, 24) deliverables ## LEGACY NOTEBOOKS
        │
        ├── 11.hilbert-geometry-diagnostics_v1.ipynb    # REAL: H1 Diagnostics         ## CICIoT2023
        │
        ├── 12.fgsm-pgd-dummy.ipynb                     # class-weighted 200 CICIoT2023        ## 10 eps, 32 b_sz
        ├── 12.fgsm-pgd-kaggle.ipynb                    # class-weighted all CICIoT2023        ## 10 eps, 128 b_sz ## T/O
        │
        ├── 13.tune-lambdas-demo.ipynb                  # DEMO: Lambdas Tuning
        ├── ./                                          # REAL: Lambdas Tuning
        │   13.tune-lambdas-bot-iot.ipynb               ## class-weighted 0.5% all BoT-IoT     ## 10 eps, 64 b_sz
        │   13.tune-lambdas-ciciot2023.ipynb            ## class-weighted 0.5% all CICIoT2023  ## 10 eps, 64 b_sz
        │   13.tune-lambdas-unsw-nb15.ipynb             ## class-weighted 0.5% all UNSW-NB15   ## 10 eps, 32 b_sz
        │
        ├── 14.maqt-prototypes_bot-iot.ipynb            # DEMO: MAQT Training          ## BoT-IoT
        ├── 14.maqt-prototypes_ciciot2023.ipynb         # DEMO: MAQT Training          ## CICIoT2023
        ├── 14.maqt-prototypes_unsw-nb15.ipynb          # DEMO: MAQT Training          ## UNSW-NB15
        │
        ├── ./                                          # REAL: MAQT Training
        │   16.estimate-lipschitz_bot-iot.ipynb         ## capped(700), ~2800 all BoT-IoT      ## 10 eps, 64 b_sz
        │   16.estimate-lipschitz_ciciot2023.ipynb      ## capped(100), ~2800 all CICIoT2023   ## 10 eps, 64 b_sz
        │   16.estimate-lipschitz_unsw-nb15.ipynb       ## capped(400), ~2800 all UNSW-NB15    ## 10 eps, 64 b_sz
        │
        ├── 17.certified-radius_ciciot2023.ipynb        # REAL: Certified Radius       ## CICIoT2023
        │
        ├── 18.fin-fout-epsstar_bot-iot.ipynb           # USE day 16 MAQT artifacts    ## BoT-IoT
        ├── 18.fin-fout-epsstar_ciciot2023.ipynb        # USE day 16 MAQT artifacts    ## CICIoT2023
        ├── 18.fin-fout-epsstar_unsw-nb15.ipynb         # USE day 16 MAQT artifacts    ## UNSW-NB15
        │
        ├── 19.fgsm-pgd-robustness_bot-iot.ipynb        # USE day 16 MAQT artifacts    ## BoT-IoT
        ├── 19.fgsm-pgd-robustness_ciciot2023.ipynb     # USE day 16 MAQT artifacts    ## CICIoT2023
        ├── 19.fgsm-pgd-robustness_unsw-nb15.ipynb      # USE day 16 MAQT artifacts    ## UNSW-NB15
        │
        ├── ./                                          # REAL: Full Algo.3 Pipeline
        │   22.e2e-algo3_bot-iot.ipynb                  ## USE day 16 MAQT artifacts   ## BoT-IoT
        │   22.e2e-algo3_ciciot2023.ipynb               ## USE day 16 MAQT artifacts   ## CICIoT2023
        │   22.e2e-algo3_unsw-nb15.ipynb                ## USE day 16 MAQT artifacts   ## UNSW-NB15
        │
        └── ./                                          # REAL: Trust Region Visualized
            24.trust-region-2d_bot-iot.ipynb            ## USE day 16 MAQT artifacts   ## BoT-IoT
            24.trust-region-2d_ciciot2023.ipynb         ## USE day 16 MAQT artifacts   ## CICIoT2023
            24.trust-region-2d_unsw-nb15.ipynb          ## USE day 16 MAQT artifacts   ## UNSW-NB15
```

## `e2e-qs-net` (MAQT + CQ-ZDR + Inference) Architecture Changelog

- **v1:** first working version; simpler quantum backend; noise bolted on afterward; fixed readout; PennyLane trains everything.
- **v2.x:** same algorithms, but rebuilt so noise and mixed states are native; classification uses a learnable head; training rides PyTorch; (later: balancing, better epoch logging, file logs).
- **v3.x:** early stopping, richer logging, class-weighted MAQT train, val monitoring, dual known-test eval (head + pipeline); more script modularization.
- **v4.x or final-<>-<>-train:** major redesign for optimization; utilize new optimized "scripts/"; backward incompatible with "legacy_scripts/".