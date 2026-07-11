# quantum-sentinel

This project applies quantum machine learning to Cyber IoT datasets using the QS-Net architecture. QS-Net combines three stages:

1. **Algorithm 1 (MAQT)**: Trains the variational circuit by minimizing cross-entropy plus intra- and inter-class prototype losses, shaping class geometry in Hilbert space.
2. **Algorithm 2 (CQ-ZDR)**: Calibrates a conformal zero-day rejection threshold from known-class calibration data.
3. **Algorithm 3 (Inference)**: Classifies known traffic within a certified radius, or flags samples as `ZERO_DAY` when they fall outside that bound.

Together, these stages yield a hybrid quantum–classical pipeline that learns known attack patterns and detects unseen (zero-day) threats with statistical guarantees.

## File Structure

```
/
...
├── data/
├── img/
├── documents/
├── scripts/
│
├── setup-guide.md                      # day 1 deliverable
├── encoding-data-iris.ipynb            # day 2 deliverable
├── training-iris.ipynb                 # day 3 deliverable
├── training-qs-net_v1.ipynb            # day 4, 5, 6, 7 deliverables
├── training-qs-net-pytorch_v2.0.ipynb  # day 8, 9 deliverables # more data, less epochs
├── training-qs-net-pytorch_v2.1.ipynb  # day 8, 9 deliverables # less data, more epochs
├── training-qs-net-pytorch_v2.2.ipynb  # day 8, 9 deliverables # class balanced on train
├── maqt-loss-unit-test-iris.ipynb      # day 10 deliverable
└── 
```

## Tech Stack

- PennyLane
- Torch