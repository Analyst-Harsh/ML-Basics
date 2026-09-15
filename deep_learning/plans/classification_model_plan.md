# PyTorch Learning Plan — Binary Classification (`classification_model.ipynb`)

## Context

Direct follow-up to `deep_learning/linear_model.ipynb` (linear regression,
already completed), which explicitly named "a classification model" as
its next-notebook topic. This notebook teaches binary classification with
PyTorch using `sklearn.datasets.make_circles` — a synthetic, deliberately
non-linearly-separable dataset (concentric circles) — so a plain logistic
regression visibly *fails* first, motivating hidden layers + nonlinear
activation as the fix, rather than introducing them as an unmotivated fact.

Same `uv` project, same environment: `torch`, `numpy`, `scikit-learn`,
`matplotlib`, `seaborn` are already installed in `deep_learning/pyproject.toml`
(confirmed) — no new dependencies, no Phase-0-style install needed this
time, just a one-line "reusing the same environment" note.

Mechanics already thoroughly taught in `linear_model.ipynb` — the manual
gradient-descent loop, the `nn.Module`/`nn.Sequential`/loss/optimizer
mapping, and the production-practices checklist — are **not re-taught
from scratch** here, only applied with a light recap. New content unique
to this notebook: sigmoid, binary cross-entropy, `nn.BCEWithLogitsLoss`'s
logits-not-probabilities gotcha, and plotting a 2D decision boundary
(mesh grid + `contourf`) — no existing convention for this in the repo,
so it gets established fresh here.

Style to match (same as the linear notebook): markdown-heavy "Block N —
Title" sections, rationale before code, interpretation of *actual*
printed/plotted output after code — every number quoted in an
interpretation cell must come from actually running that code first, not
a guess — closing with a results table + one bolded takeaway.

## Approach

The central pedagogical arc is Block 4: show logistic regression fail on
the circles data first (straight decision boundary, ~50% accuracy), then
add one hidden layer + ReLU and show it succeed (~95%+ accuracy),
directly contrasting the two.

Reuse existing repo conventions and already-installed libraries — don't
reinvent:
- `sklearn.datasets.make_circles` for the dataset
- `sklearn.model_selection.train_test_split` for the train/val split
- `matplotlib` for scatter plots, loss curves, and decision boundaries
  (`contourf` over a mesh grid — not a hand-rolled boundary plot)

### Setup note (no Phase-0 rebuild)
One short markdown cell: same `uv` project/environment as
`linear_model.ipynb`, same `torch`/`device` resolution
(`torch.backends.mps.is_available()`), no new dependencies. One code cell
reusing the existing `import torch; device = ...` pattern — no
re-explanation.

### Block 1 — Data: Circles
- `make_circles(n_samples=..., noise=..., factor=..., random_state=...)`
  → `X` (2 features), `y` (binary label).
- Scatter plot `X` colored by `y` — markdown interpretation calling out
  visually that the two classes are concentric rings, i.e. **not
  linearly separable** (sets up Block 4a's planned failure).
- `train_test_split` into train/val; convert to `float32` tensors, `X`
  shaped `(N, 2)`, `y` shaped `(N, 1)`.
- Self-check: printed shapes/dtypes match expectations; scatter plot
  visibly shows two interleaved rings.

### Block 2 — Mechanics Recap (light): Sigmoid & BCE
- Markdown: sigmoid formula `σ(z) = 1 / (1 + e^-z)`, why it maps logits
  to `(0, 1)` probabilities. Quick plot of its S-shape.
- Markdown: binary cross-entropy formula
  `-[y·log(p) + (1-y)·log(1-p)]`, one line on why squared error isn't
  used for classification.
- ONE scalar autograd-vs-hand-derivative check (mirrors the linear
  notebook's Block 2 scalar example, not the full loop): a single
  `requires_grad=True` weight, one point's BCE loss, `.backward()`,
  compare `w.grad` to the hand-computed derivative
  `dL/dw = (σ(wx) - y) * x`.
- Explicitly skip a full manual training loop — one line noting this
  mechanic was already proven in `linear_model.ipynb` Block 2.
- Self-check: sigmoid plot has the expected S-shape; `w.grad` matches
  the hand-computed value.

### Block 3 — `nn.Module` for Classification (brief)
- Short mapping note (briefer than the linear notebook's — the concept
  was already taught, this just adapts it): `nn.Linear(2, 1)` paired
  with BCE instead of MSE for a classification head.
- **Explicit gotcha callout**: `nn.BCEWithLogitsLoss()` expects raw
  logits (pre-sigmoid), combining sigmoid + BCE internally for numerical
  stability — not the same as `nn.BCELoss()` fed post-sigmoid
  probabilities. This is the loss used in both 4a and 4b, so the model's
  final layer should NOT include a `Sigmoid()`.
- Build `nn.Sequential(nn.Linear(2, 1))`, print it, instantiate
  `nn.BCEWithLogitsLoss()` + an optimizer, one manual
  `zero_grad → backward → step` cycle, print loss before/after.
- Self-check: one step reduces loss; printed model shows
  `in_features=2, out_features=1`.

### Block 4a — Project Part 1: Logistic Regression FAILS
- Fresh seeded `nn.Sequential(nn.Linear(2, 1))` + `BCEWithLogitsLoss` +
  optimizer, moved to `device`.
- Training loop on the circles train/val split (mirrors the linear
  notebook's loop: `model.train()`/`model.eval()`, `torch.no_grad()` for
  val, loss history recorded).
- Plot train/val loss curve.
- **Decision boundary plot** (new technique, introduced in detail here):
  build a 2D mesh grid over the padded feature space
  (`np.meshgrid` + `np.linspace`), convert grid points to a `float32`
  tensor, run the model in `eval()` + `no_grad()`, `torch.sigmoid(logits)`
  thresholded at `0.5` for predicted class, reshape back to the grid,
  `plt.contourf(...)` for filled regions with the real data scattered on
  top.
- Compute validation accuracy.
- Markdown interpretation with real numbers: expect accuracy ≈ 50%
  (coin-flip), boundary is a straight line — explain this is the
  *expected*, correct result: one linear layer can only produce a linear
  boundary, which can't separate concentric circles.
- Self-check: accuracy ≈ 50%, boundary plot is a straight line.

### Block 4b — Project Part 2: Add a Hidden Layer + ReLU — It Works
- Markdown: motivate the fix directly from 4a's failure — a linear model
  can't bend, so `nn.ReLU()` is the nonlinearity that lets the network
  compose linear pieces into a curved boundary.
- Fresh seeded `nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))`
  + same `BCEWithLogitsLoss` + optimizer, moved to `device`.
- Same training loop structure as 4a (reused, not re-explained); same
  decision-boundary technique reused from 4a.
- Compute validation accuracy.
- Markdown interpretation with real numbers: expect accuracy ≥ ~90%
  (target ~95%+), boundary visibly curves/rings around one class —
  directly contrasted against 4a's straight line and ~50% accuracy.
- Self-check: accuracy ≥ ~90%, boundary visibly curved, not a straight
  line.

### Block 5 — Production Practices (lighter — apply, don't re-teach)
- Short checklist pointing back at where each practice already appeared
  in 4a/4b (seeding, `.to(device)`, `train()`/`eval()`, `no_grad()` for
  validation) — confirmation, not re-explanation.
- `state_dict()` save/load round-trip for the winning MLP (Block 4b's):
  save → build a fresh identical architecture → `load_state_dict` →
  compare predictions/boundary before and after with `torch.allclose`.
- Self-check: reloaded model's predictions match the original's exactly.

### Block 6 — Wrap-up
- Markdown results table: model (logistic regression vs MLP), final
  train/val loss, val accuracy, boundary shape (line vs curve).
- One bolded one-sentence takeaway (e.g. "a linear model can only draw a
  straight decision boundary — adding a hidden layer with a nonlinear
  activation lets the network learn curved boundaries and solve problems
  linear models fundamentally cannot").
- "Next steps / deferred" bullets: multi-class classification
  (`nn.CrossEntropyLoss`, softmax), confusion matrix/precision/recall
  beyond raw accuracy, `Dataset`/`DataLoader` classes, deeper/wider MLPs
  and regularization (dropout, weight decay), LR schedulers, and CNNs
  once image data is introduced.

## Files touched
- `deep_learning/classification_model.ipynb` — new notebook, built out
  per the blocks above
- `deep_learning/classification_model_state.pt` — saved in Block 5
  (already covered by `.gitignore`'s `*.pt` rule, added earlier)

## Verification
- Run every notebook cell top-to-bottom after each block is built —
  each block's self-check (printed accuracy, `torch.allclose`, visual
  boundary shape) should pass before moving to the next.
- Final check: Block 4a's accuracy ≈ 50% with a straight-line boundary
  (expected failure), Block 4b's accuracy ≥ ~90% with a curved boundary,
  Block 5's reloaded model's predictions match the original exactly.
