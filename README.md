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
├── setup-guide.md                      # day 1 deliverable
├── encoding-data-iris.ipynb            # day 2 deliverable
├── training-iris.ipynb                 # day 3 deliverable
│
├── # day 4, 5, 6, 7 deliverables ## TON IoT
├── training-qs-net_v1.ipynb            ## FULL END-2-END PIPELINE
│
├── # day 8, 9 deliverables ## use PennyLane + PyTorch ## TON IoT
├── training-qs-net-pytorch_v2.0.ipynb  ## FULL END-2-END PIPELINE
│                                       ## epoch curve uses last minibatch only
│                                       ## measure grad during weight update, averages over epoch
│
├── training-qs-net-pytorch_v2.1.ipynb  ## FULL END-2-END PIPELINE
│                                       ## epoch curve uses last minibatch only 
│                                       ## measure grad during weight update, averages over epoch
│
├── training-qs-net-pytorch_v2.2.ipynb  ## FULL END-2-END PIPELINE
│                                       ## added logging ## utilize "legacy_scripts/"
│                                       ## epoch curve averages all minibatches
│                                       ## measure grad after weight update, averages over epoch
│
├── maqt-loss-unit-test-iris.ipynb          # day 10 deliverable
├── Hilbert-geometry-diagnostics_v1.ipynb   # day 11 deliverable ## CICIoT2023
│
├── # day 12 deliverable ## CICIoT2023
├── fgsm-pgd-dummy.ipynb    ## class-weighted 200 data, 10 epochs, 32 batch size
├── fgsm-pgd-kaggle.ipynb   ## class-weighted all data, 10 epochs, 128 batch size (TIMEOUT)
│
├── # day 13 deliverable
├── tune-lambdas-demo.ipynb
├── tune-lambdas-ciciot2023.ipynb   ## class-weighted 0.5% of all data, 10 epochs, 64 batch size
├── tune-lambdas-unsw-nb15.ipynb    ## class-weighted 0.5% of all data, 10 epochs, 32 batch size
├── tune-lambdas-bot-iot.ipynb      ## class-weighted 0.5% of all data, 10 epochs, 64 batch size
│
├── # day 14 deliverable demo
├── qs-net-prototypes_ciciot2023.ipynb
├── qs-net-prototypes_unsw-nb15.ipynb
├── qs-net-prototypes_bot-iot.ipynb
│
├── training-qs-net_v3.0.ipynb  ## FULL END-2-END PIPELINE
│                               ## CICIoT2023 class-weighted 200 data, 30 epochs, 64 batch size
└──
```

## Changes with Versions for QS-Net

- **v1:** first working version; simpler quantum backend; noise bolted on afterward; fixed readout; PennyLane trains everything.
- **v2.x:** same algorithms, but rebuilt so noise and mixed states are native; classification uses a learnable head; training rides PyTorch; (later: balancing, better epoch logging, file logs).
- **v3.x:** early stopping, richer logging, class-weighted MAQT train, val monitoring, dual known-test eval (head + pipeline); more script modularization.
- **v4.x:** major redesign for optimization; utilize new optimized "scripts/"; backward incompatible with "legacy_scripts/".

## Tech Stack

- PennyLane
- Torch