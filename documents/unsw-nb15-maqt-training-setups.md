# UNSW-NB15 MAQT Training Setups

This note summarizes how lawun chose MAQT training settings on FROZEN UNSW-NB15, what the loss curves mean, and what is still open. UNSW is harder: **8 known classes**, heavy Normal imbalance, and a smaller quantum width (PCA → 4 qubits vs 5 on BoT-IoT).

**Latest (full known train):** old inter loss (no margin / no hardest-only); cosine LR start $0.01$; $\lambda_1=0.66$, $\lambda_2=0.33$; weighted sampler; patience $5$. CE full-test still **~46%**. See §5.

Unless noted, “~46%” = CE-head on full known test, **no** conformal reject mask.

---

## 0. Loss (two inter variants)

$
L = L_{CE} + \lambda_1 L_{intra} + \lambda_2 L_{inter}
$

| Term | Role |
|------|------|
| $L_{CE}$ | classify from circuit expectations |
| $L_{intra}$ | pull $\rho(x)$ toward own prototype |
| $L_{inter}$ | push away from wrong-class prototypes |

**Old inter** (`scripts/loss.py`, current train): mean negative TD to other-class prototypes. No margin, no hardest-only. $L_{inter}$ is typically **negative**.

**Hinge inter** (v2.4 suggestion, tried then dropped): `relu(m - d)` on nearest wrong prototype (`hardest_only=True`). Margin $m$ unpaid while $d < m$.

---

## 1. Why BoT knobs don’t copy

| | BoT-IoT | UNSW-NB15 |
|--|---------|-----------|
| Known classes | 4 | 8 |
| Full train | ~148k | ~81k |
| Qubits (PCA) | 5 | 4 |
| Selected CE full-test | ~68% | stuck ~46% |

---

## 2. First full run: weak $\lambda_2$, hinge margin $0.6$ → ~46%

**Setup:** full train; cosine $0.01$; $\lambda_1=0.5$; $\lambda_2=0.01$; hinge margin $0.6$; patience $5$.

| Metric | ~value |
|--------|--------|
| CE full-test acc | ~46% ($n \approx 10086$) |
| CE full-test macro-F1 | ~0.26 |
| Algo-3 accepted-known acc | ~34% |

$L_{CE}$ fell (LR $0.01$ fine). Mean inter TD ~0.23–0.27 ≪ $0.6$ → hinge unpaid; $\lambda_2$ too small; ES cut ~epoch 9.

---

## 3. Subset smoke: hinge $\lambda_2=0.1$, margin $0.3$

DPP cap 1000/class. $L_{inter}$ ~0.05; TD near $0.3$. Looked promising → promoted to full-data hinge trials.

---

## 4. Hinge on full data (margins $0.2$–$0.6$) — no CE gain

Tried full train with hinge + hardest-only across margins **$0.2$–$0.6$** (incl. planned $\lambda_2=0.1$, margin $0.3$, patience $15$).

**Result:** CE full-test stayed **~46%**. Geometry knobs moved; accuracy did not.

→ Dropped hinge. Reverted to **old inter loss**.

---

## 5. Latest: old inter loss (still ~46%)

**Setup:**

| Knob | Choice |
|------|--------|
| Inter loss | **old** (no margin, no hardest-only) |
| Train | full FROZEN known |
| LR | cosine, start $0.01$ |
| $\lambda_1$, $\lambda_2$ | $0.66$, $0.33$ |
| Sampler | weighted (keep Normal; don’t hard-balance first) |
| Early stop | patience $5$ |
| Notebooks | `final-unsw-nb15-maqt-train` → `final-unsw-nb15-algo3-cached (maqt)` |

Same ~46% CE full-test. Open: lift accuracy (retrain / other knobs), not “flip hinge back on” without a new signal.

---

## Decision trail

```text
full / hinge / λ2=0.01 / margin=0.6 / patience=5
  → ~46% CE; hinge unpaid; ES short

subset / hinge / λ2=0.1 / margin=0.3
  → geometry OK on smoke; not a CE win on full data

full / hinge / margin ∈ {0.2…0.6}
  → still ~46% → abandon hinge + hardest-only

latest: full / old inter / λ1=0.66 / λ2=0.33 / cosine 0.01
  → still ~46% CE full-test
```
