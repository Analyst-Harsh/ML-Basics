# PyTorch Learning Plan — First Model (`linear_model.ipynb`)

## Context

This is the user's first-ever PyTorch model. The target notebook,
`deep_learning/linear_model.ipynb`, started as a blank slate — just a
title cell ("Linear model usign pytorch") and one empty markdown cell.

The goal is to learn PyTorch basics *while* building a linear regression
model with `nn.Sequential`, trained on synthetic linear data
(`y = wx + b + noise`) so the true weight/bias are known and recoverable —
this makes gradient descent mechanics concretely verifiable rather than
abstract. Core production-grade practices (seeding, device-agnostic code,
train/eval modes, state_dict save/load) are folded into this same
notebook, with heavier production topics (DataLoader/Dataset, LR
schedulers, checkpointing) explicitly deferred to a later notebook.

Style to match: `Week-2/day-4.ipynb` — markdown-heavy "Block N — Title"
sections, rationale before code, interpretation of actual printed numbers
after code, closing results table + one bolded takeaway. NOT the
code-first `Week-2/capstone.ipynb` style.

## Approach

Build out `linear_model.ipynb` in the phased structure below. Each phase
maps to one or more notebook blocks. The ordering is deliberate: every
`nn`/`torch.optim` API in Block 3 has its manual, hand-rolled counterpart
shown first in Block 2, so the abstraction never feels like magic.

Reuse existing repo conventions throughout — don't reinvent:
- `numpy` for synthetic data generation
- `sklearn.model_selection.train_test_split` for the train/val split
- `matplotlib`/`seaborn` for the data scatter and loss curve plots

### Phase 0 — Environment Setup
- Terminal (outside the notebook): `uv add torch` in `deep_learning/`.
  Skip `torchvision`/`torchaudio` — not needed for tensors/tabular data.
- **Known risk**: `deep_learning/.python-version` is pinned to `3.14`.
  PyTorch's Python 3.14 wheel support is very recent and there are
  reports of install failures specifically on macOS. If `uv add torch`
  fails to resolve, fall back to pinning this project to Python 3.12 or
  3.13 (`uv python pin 3.12`) before retrying — `deep_learning` is
  already its own isolated `uv` project, so this doesn't affect
  `Week-1`/`Week-2`.
- Notebook: import torch, print `torch.__version__`, check
  `torch.backends.mps.is_available()`, set
  `device = torch.device("mps" if available else "cpu")`.
- Self-check: import succeeds, device prints as expected (Mac → `mps`,
  falls back to `cpu` with a one-line note if sandboxed).

### Block 1 — Tensors
- Tensor creation (from list, from numpy, `zeros`/`ones`/`randn`), `.shape`,
  `.dtype`, elementwise ops, `@`/`matmul`, `torch.from_numpy`/`.numpy()`
  round-trip, `.to(device)`.
- Markdown interpretation of real printed shapes/dtypes; note `float32`
  is the default and why it matters for `nn.Linear` later.
- Self-check: round-trip tensor equals original numpy array
  (`np.allclose`).

### Block 2 — Autograd, hand-rolled
- `requires_grad=True`, `.backward()`, `.grad`, `torch.no_grad()`,
  zeroing gradients, why gradients accumulate.
- Scalar toy example: compute a simple loss, call `.backward()`, verify
  `w.grad` against the hand-computed derivative.
- **Manual gradient descent loop** on a tiny synthetic 1D toy dataset
  (known `w_true`, `b_true`) — plain Python loop, no `nn.Module`, no
  optimizer. Print learned vs true `w`/`b`.
- Self-check: learned `w`, `b` land within ~0.05–0.1 of true values —
  explicit printed numeric comparison.

### Block 3 — `nn.Module`, `nn.Sequential`, loss, optimizer
- Map each manual-loop piece from Block 2 onto its `nn` equivalent:
  weights/bias → `nn.Linear` params, manual squared error →
  `nn.MSELoss`, manual update → `optimizer.step()`.
- Build `model = nn.Sequential(nn.Linear(1, 1))`. Print `model` and
  `list(model.parameters())` (random init).
- Instantiate `nn.MSELoss()` and `torch.optim.SGD(model.parameters(), lr=...)`.
  Run one manual step by hand: `optimizer.zero_grad()` →
  `loss.backward()` → `optimizer.step()`. Print loss before/after.
- Self-check: one step reduces loss.

### Block 4 — The project: linear regression on synthetic data
- Generate `X`, `y = w_true * X + b_true + noise` with numpy; scatter plot.
- `train_test_split` (sklearn) for train/val; convert to `float32`
  tensors shaped `(N, 1)`.
- `torch.manual_seed(...)` before model init (first explicit production
  practice, needed here for reproducible results).
- Fresh `nn.Sequential(nn.Linear(1, 1))` + loss + optimizer.
- Training loop: `model.train()` → forward/backward/step on train set;
  periodic `model.eval()` + `torch.no_grad()` forward on val set; record
  both loss histories.
- Plot train vs val loss curves.
- Print learned `model[0].weight`/`model[0].bias` next to `w_true`/`b_true`
  — the pedagogical payoff of using synthetic data with known ground truth.
- Self-check: learned weight/bias within a small tolerance of true values;
  val loss doesn't diverge from train loss.

### Block 5 — Production practices review
- Markdown checklist naming each practice already used in Block 4 (seed,
  `.to(device)`, `train()`/`eval()`) with a pointer back to where it
  appeared — explain *why*, not just re-demo.
- Explicitly move model + tensors to `device`, rerun briefly to confirm
  it still works identically.
- `torch.save(model.state_dict(), "linear_model_state.pt")` → build a
  fresh model → `load_state_dict` → compare predictions before/after
  reload with `torch.allclose`.
- Self-check: reloaded model's predictions match original's.

### Block 6 — Wrap-up
- Markdown results table: true/learned `w`, true/learned `b`, final
  train/val loss.
- One bolded one-sentence takeaway.
- "Next steps / deferred" bullets: `Dataset`/`DataLoader` classes, LR
  schedulers, experiment tracking/checkpointing, GPU deep-dive, and a
  classification model (logistic regression / small MLP) as the natural
  next notebook.

## Files touched
- `deep_learning/linear_model.ipynb` — built out per phases above
- `deep_learning/pyproject.toml` — add `torch` dependency (`uv add torch`)

## Verification
- Run every notebook cell top-to-bottom (`Run All`) after each block is
  built — each block's self-check (printed comparisons, `np.allclose`/
  `torch.allclose`) should pass before moving to the next block.
- Final check: Block 4's learned `w`/`b` should be close to the synthetic
  dataset's true `w_true`/`b_true`, and Block 5's reload predictions should
  match the original model's predictions exactly.
