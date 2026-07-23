# quantum-sentinel

This project applies quantum machine learning to Cyber IoT datasets using the QS-Net architecture. QS-Net combines three stages:

1. **Algorithm 1 (MAQT)**: Trains the variational circuit by minimizing cross-entropy plus intra- and inter-class prototype losses, shaping class geometry in Hilbert space.
2. **Algorithm 2 (CQ-ZDR)**: Calibrates a conformal zero-day rejection threshold from known-class calibration data.
3. **Algorithm 3 (Inference)**: Classifies known traffic within a certified radius, or flags samples as `ZERO_DAY` when they fall outside that bound.

Together, these stages yield a hybrid quantum–classical pipeline that learns known attack patterns and detects unseen (zero-day) threats with statistical guarantees.

## Datasets

Modified versions of the following datasets:

- [CIC IoT 2023 (Canadian Institute for Cybersecurity)](https://www.kaggle.com/datasets/himadri07/ciciot2023)
- [BoT IoT](https://research.unsw.edu.au/projects/bot-iot-dataset)
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [Edge-IIoTset (Ferrag et al., IEEE Access 2022)](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) (discarded later)
- [TON IoT (UNSW Canberra)](https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset) (discarded later)

## File Structure

```
/
...
├── data/
├── img/
├── documents/
├── legacy_scripts/
├── scripts/
│
├── 01.setup-guide.md                      # day 1 deliverable
├── 02.encoding-data-iris.ipynb            # day 2 deliverable
├── 03.training-iris.ipynb                 # day 3 deliverable
│
├── # day 4, 5, 6, 7 deliverables ## use PennyLane only ## TON IoT
│   └── e2e-qs-net_v1.0.ipynb
│
├── # day 8, 9 deliverables     ## use PennyLane + PyTorch ## TON IoT
│   ├── e2e-qs-net_v2.0.ipynb   ## epoch curve uses last minibatch only
│   │                           ## measure grad during weight update, averages over epoch
│   │
│   ├── e2e-qs-net_v2.1.ipynb   ## epoch curve uses last minibatch only 
│   │                           ## measure grad during weight update, averages over epoch
│   │
│   └── e2e-qs-net_v2.2.ipynb   ## added logging ## utilize "legacy_scripts/"
│                               ## epoch curve averages all minibatches
│                               ## measure grad after weight update, averages over epoch
│
├── 10.maqt-loss-unit-test-iris.ipynb          # day 10 deliverable
├── 11.hilbert-geometry-diagnostics_v1.ipynb   # day 11 deliverable ## CICIoT2023
│
├── # day 12 deliverable           ## CICIoT2023
│   ├── 12.fgsm-pgd-dummy.ipynb    ## class-weighted 200 data, 10 epochs, 32 batch size
│   └── 12.fgsm-pgd-kaggle.ipynb   ## class-weighted all data, 10 epochs, 128 batch size (TIMEOUT)
│
├── # day 13 deliverable
│   ├── 13.tune-lambdas-demo.ipynb
│   ├── 13.tune-lambdas-ciciot2023.ipynb   ## class-weighted 0.5% of all data, 10 epochs, 64 batch size
│   ├── 13.tune-lambdas-unsw-nb15.ipynb    ## class-weighted 0.5% of all data, 10 epochs, 32 batch size
│   └── 13.tune-lambdas-bot-iot.ipynb      ## class-weighted 0.5% of all data, 10 epochs, 64 batch size
│
├── # day 14 deliverable demo
│   ├── 14.maqt-prototypes_ciciot2023.ipynb
│   ├── 14.maqt-prototypes_unsw-nb15.ipynb
│   └── 14.maqt-prototypes_bot-iot.ipynb
│
├── # day 15 deliverable
│
├── # day 16 deliverable
│   ├── 16.estimate-lipschitz_ciciot2023.ipynb     ## capped sample(100), ~2800 data, 10 epochs
│   ├── 16.estimate-lipschitz_unsw-nb15.ipynb      ## capped sample(400), ~2800 data, 10 epochs
│   └── 16.estimate-lipschitz_bot-iot.ipynb        ## capped sample(700), ~2800 data, 10 epochs
│
├── # day 17 deliverable
├── # day 18 deliverable
│
├── e2e-qs-net_v3.0.ipynb   ## CICIoT2023 class-weighted 200 data, 30 epochs, 64 batch size
├── e2e-qs-net_v4.0.ipynb   ##
└──
```

## `e2e-qs-net` (MAQT + CQ-ZDR + Inference) Architecture Changelog

- **v1:** first working version; simpler quantum backend; noise bolted on afterward; fixed readout; PennyLane trains everything.
- **v2.x:** same algorithms, but rebuilt so noise and mixed states are native; classification uses a learnable head; training rides PyTorch; (later: balancing, better epoch logging, file logs).
- **v3.x:** early stopping, richer logging, class-weighted MAQT train, val monitoring, dual known-test eval (head + pipeline); more script modularization.
- **v4.x:** major redesign for optimization; utilize new optimized "scripts/"; backward incompatible with "legacy_scripts/".

## Tech Stack

`PennyLane`, `PyTorch`