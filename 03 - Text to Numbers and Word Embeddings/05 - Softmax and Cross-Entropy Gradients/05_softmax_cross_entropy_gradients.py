# -*- coding: utf-8 -*-

"""
∂ 05 · Softmax and Cross-Entropy Gradients for Word2Vec

Python-script version of 05_softmax_cross_entropy_gradients.ipynb.
Run:  python 05_softmax_cross_entropy_gradients.py
(each figure opens in a window; close it to continue)
"""

# Module 03 – Text to Numbers and Word Embeddings · Notebook 5 of 9
# Course notes: pages 11–14 – E=-log(y_o), (∂E)/(∂u_k)=(∂E)/(∂y_o)(Σ_i
# (∂y_o)/(∂z_i)(∂z_i)/(∂u_k)), the quotient-rule derivation of the softmax, Δu_k=η(δ_ok-
# y_k)h^ᵀ, Δv=η(u_o-Σ_i y_iu_i)
#
# The model from notebook 04 predicts neighbours badly (E≈log n). To train it we need ∂E/∂U
# and ∂E/∂V. This notebook derives them step by step, following the notes, checks them
# numerically, and then reads what they mean: the updates have a beautifully simple
# interpretation.
#
# --- What you will learn ---
#   #  |  Result
#   1  |  The softmax derivative (∂y_o)/(∂z_k)=y_o(δ_ok-y_k), via the quotient rule
#   2  |  (∂E)/(∂z_k)=y_k-δ_ok – "prediction minus truth"
#   3  |  Output vectors: Δu_k=η (δ_ok-y_k) h – every row of U changes
#   4  |  Input vectors: Δv_c=η (u_o-Σ_i y_iu_i) – one column of V changes
#   5  |  Gradient check, and a picture of what one update does

# ==============================================================================
# 0 · Setup
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------- one consistent, clean plotting style for the whole notebook ----------
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948",
         ink="#0b0b0b", muted="#52514e", grid="#e6e5e1")
plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"], "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
    "xtick.color": C["muted"], "ytick.color": C["muted"], "legend.frameon": False,
    "lines.linewidth": 2.0, "font.size": 10.5,
    "axes.prop_cycle": plt.cycler(color=[C[k] for k in
                       ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]]),
})
np.set_printoptions(precision=4, suppress=True)

# ==============================================================================
# 1 · The variables
# ==============================================================================
#
# --- Notation for the word2vec notebooks (same as the course notes, pages 10–14) ---
#
#   Symbol  |  Shape  |  Meaning
#   n  |  number  |  vocabulary size: how many different words we know
#   d  |  number  |  embedding size: how many numbers describe one word (notes: typically 300)
#   w_k  |  –  |  the k-th word of the vocabulary, k=1..n
#   x  |  n×1  |  one-hot vector of the input word: all zeros except a 1 at its index
#   V  |  d×n  |  input matrix. Column k, written v_k, is the word vector of w_k
#   h=Vx  |  d×1  |  hidden vector. Because x is one-hot, h is simply the column v_c of the input word c
#   U  |  n×d  |  output matrix. Row k, written u_k, is the "output vector" of w_k
#   z=Uh  |  n×1  |  one score per vocabulary word: z_k=u_k·h
#   y=softmax(z)  |  n×1  |  probabilities: y_k=e^(z_k)/Σ[m] e^(z_m) = predicted chance that w_k is the context word
#   o  |  index  |  the true output (context) word of this training example
#   E=-log y_o  |  number  |  error: minus the log of the probability given to the right word
#   δ_ok  |  number  |  Kronecker delta: 1 if o=k, else 0
#   η  |  number  |  learning rate
#
# No biases and no activation function except the final softmax, so the model has exactly 2 d
# n parameters (notes, page 10).
# New here: t = the one-hot vector of the true output word (t_k=δ_ok), and e=y-t, the
# prediction error vector.
#
# ==============================================================================
# 2 · The forward pass for one training pair, as code
# ==============================================================================
#
# For a pair (c,o) – input word c, true neighbour o:
#
#     h=v_c, z=Uh, y=softmax(z), E=-log y_o
#
# We use a tiny model (n=6 words, d=3) so every number can be printed.

def softmax(z):
    e = np.exp(z - z.max())                 # subtract the max: same result, no overflow
    return e / e.sum()


def forward(V, U, c, o):
    """One skip-gram pair (c → o). Returns h, z, y and the error E."""
    h = V[:, c]                             # h = V x = column c
    z = U @ h                               # z_k = u_k · h
    y = softmax(z)
    return h, z, y, -np.log(y[o])

# A tiny random model and one pair:

rng = np.random.default_rng(0)
n_toy, d_toy = 6, 3
V = rng.normal(0, 0.5, (d_toy, n_toy))
U = rng.normal(0, 0.5, (n_toy, d_toy))
c, o = 2, 4                                 # input word 2, true neighbour 4
h, z, y, E = forward(V, U, c, o)
print("h =", h.round(3))
print("z =", z.round(3))
print("y =", y.round(3), " (sums to", y.sum().round(6), ")")
print(f"E = -log y_o = -log {y[o]:.3f} = {E:.3f}")

# ==============================================================================
# 3 · The derivation, following the notes
# ==============================================================================
#
# We need (∂E)/(∂u_k) for every row of U, and (∂E)/(∂v_c) for the input column of V.
# Everything goes through the scores z, so we first find ∂E/∂z_k.
#
# --- Step 1 – derivative of the error with respect to y_o ---
#
#     E=-log y_o  (∂E)/(∂y_o)=-1/y_o
#
# --- Step 2 – derivative of the softmax (the quotient rule, notes page 12) ---
#
# Write S=Σ[m=1..n] e^(z_m), so y_o=e^(z_o)·dfrac1S. Differentiate the product with respect to
# z_k:
#
#     (∂y_o)/(∂z_k)=frac1S (∂e^(z_o))/(∂z_k)+e^(z_o) (∂)/(∂z_k)(frac1S)=frac1S e^(z_o)(∂z_o)/(∂z_k)+e^(z_o)(-1/(S^2))(∂S)/(∂z_k)_e^(z_k)
#
# Now (∂z_o)/(∂z_k)=δ_ok (1 if o=k, else 0), and (e^(z_o))/S=y_o, (e^(z_k))/S=y_k:
#
#     (∂y_o)/(∂z_k)=y_o δ_ok-y_o y_k=y_o (δ_ok-y_k)
#
# --- Step 3 – combine: the error signal at the scores ---
#
#     (∂E)/(∂z_k)=(∂E)/(∂y_o)·(∂y_o)/(∂z_k)=-frac1y_o y_o (δ_ok-y_k)= y_k-δ_ok
#
# In vector form ∂E/∂z=y-t=e: predicted probability minus the truth, for every word. (The same
# result as softmax + cross-entropy in module 02.)
#
# --- Step 4 – rows of U (output vectors) ---
#
# z_k=u_k·h depends on row u_k and on no other row, and (∂z_k)/(∂u_k)=h^ᵀ. So (notes: the sum
# over i keeps only the term i=k)
#
#     (∂E)/(∂u_k)=Σ_i (∂E)/(∂z_i)(∂z_i)/(∂u_k)=(y_k-δ_ok) h^ᵀ  Δu_k=-η(∂E)/(∂u_k)=η (δ_ok-y_k) h^ᵀ
#
#   case  |  update  |  meaning (notes page 13)
#   k=o (the true word)  |  η (1-y_o) h  |  pull u_o towards h – more so the less sure we were
#   k≠ o  |  -η y_k h  |  push u_k away from h, in proportion to how much we wrongly predicted it. "If 'my' has probability 0, no need to change anything."
#
# --- Step 5 – the input column of V (the word vector) ---
#
# h affects E through every score: (∂z_k)/(∂h)=u_k, so
#
#     (∂E)/(∂h)=Σ[k=1] ^n(y_k-δ_ok) u_k=U^ᵀ(y-t)
#
# Since h=Vx=v_c, only column c of V influences h ("k = input → I; k ≠ input → 0", notes page
# 14):
#
#     Δv_c=-η U^ᵀ(y-t)=η(u_o-Σ[i=1..n] y_i u_i), Δv_k=0 for k≠ c
#
# Notes' interpretation: u_o is "the word I should have predicted", and Σ_i y_iu_i is "each
# word i predicted with probability y_i" – the average prediction. The word vector moves from
# what the model expected towards what actually happened.
#
# ==============================================================================
# 4 · The gradients in code
# ==============================================================================
#
# --- 4.1 · The error signal e=y-t ---

def error_signal(y, o):
    """dE/dz = y - t, where t is the one-hot vector of the true word o."""
    e = y.copy()
    e[o] -= 1.0
    return e

e = error_signal(y, o)
print("y     =", y.round(3))
print("t     =", np.eye(n_toy)[o])
print("e=y-t =", e.round(3), "  (sums to", e.sum().round(10), ")")

# The entries of e always sum to 0 (probabilities sum to 1, and so does t): whatever is added
# to the true word's score is taken from the others.
#
#  Plot: y, t and e for this pair. Only the true word has a negative error (its score must go
#  up); every other word has a positive error equal to its probability (its score must go
#  down).

fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
words_toy = [f"w{k}" for k in range(n_toy)]
for ax, vals, title, col in [(axs[0], y, "prediction y", C["blue"]), (axs[1], np.eye(n_toy)[o], f"truth t (o = w{o})", C["aqua"]),
                             (axs[2], e, "error signal e = y − t = ∂E/∂z", C["violet"])]:
    bars = ax.bar(words_toy, vals, color=col, width=0.6)
    ax.bar_label(bars, fmt="%.2f", fontsize=8.5); ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_title(title); ax.grid(axis="x", visible=False)
axs[2].set_ylim(e.min() - 0.15, max(e.max(), 0) + 0.15)
plt.tight_layout(); plt.show()

# --- 4.2 · All gradients for one pair ---
# * `dU = np.outer(e, h)` → row k is e_k h^ᵀ=(y_k-δ_ok)h^ᵀ (Step 4) – an n×d matrix, all rows
#   non-zero.
# * `dh = U.T @ e` → U^ᵀ(y-t) (Step 5); it goes into column c of a zero matrix shaped like V.

def gradients(V, U, c, o):
    """dE/dV and dE/dU for one skip-gram pair (c → o), full softmax."""
    h, z, y, E = forward(V, U, c, o)
    e = error_signal(y, o)                  # dE/dz
    dU = np.outer(e, h)                     # dE/du_k = e_k h^T   (every row)
    dV = np.zeros_like(V)
    dV[:, c] = U.T @ e                      # dE/dv_c = U^T e     (one column only)
    return dV, dU, E

dV, dU, E = gradients(V, U, c, o)
print("non-zero columns of dV:", np.flatnonzero(np.abs(dV).sum(axis=0)), " (only the input word c =", c, ")")
print("non-zero rows of dU   :", np.flatnonzero(np.abs(dU).sum(axis=1)), " (every word)")

# --- 4.3 · Gradient check ---
# Same brute-force check as module 02: nudge each parameter by ±ε and measure the change in E.

def numerical_grad(f, theta, eps=1e-6):
    g = np.zeros_like(theta)
    for idx in np.ndindex(theta.shape):
        old = theta[idx]
        theta[idx] = old + eps; fp = f()
        theta[idx] = old - eps; fm = f()
        theta[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def rel_err(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)

E_fn = lambda: forward(V, U, c, o)[3]
print(f"dU relative error = {rel_err(dU, numerical_grad(E_fn, U)):.2e}")
print(f"dV relative error = {rel_err(dV, numerical_grad(E_fn, V)):.2e}")

# Both around 10^(-10): the derivation is right.
#
#  Plot: where the gradient lives. ∂E/∂V (left) is zero except for the input word's column;
#  ∂E/∂U (right) touches every row.

fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
vmax = max(np.abs(dV).max(), np.abs(dU).max())
axs[0].imshow(dV, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axs[0].set_title(f"∂E/∂V  (d × n): only column c = w{c}"); axs[0].set_xticks(range(n_toy), words_toy); axs[0].set_yticks(range(d_toy))
im = axs[1].imshow(dU, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axs[1].set_title("∂E/∂U  (n × d): every row"); axs[1].set_yticks(range(n_toy), words_toy); axs[1].set_xticks(range(d_toy))
for a in axs: a.grid(False)
plt.colorbar(im, ax=axs, shrink=0.8)
plt.show()

# ==============================================================================
# 5 · One update, and what it means geometrically
# ==============================================================================
#
# --- 5.1 · Apply one SGD step ---
# V← V-η ∂E/∂V, U← U-η ∂E/∂U. The true word's probability should rise and E should fall.

eta = 0.5
V_new, U_new = V - eta * dV, U - eta * dU
_, _, y_new, E_new = forward(V_new, U_new, c, o)
print(f"y_o: {y[o]:.3f} → {y_new[o]:.3f}      E: {E:.3f} → {E_new:.3f}")

# --- 5.2 · Check the notes' formula for Δv_c ---
# Δv_c=η(u_o-Σ_i y_iu_i) – "the word I should have predicted minus the average prediction":

expected_u = y @ U                                  # Σ_i y_i u_i  (a d-vector)
delta_vc_notes = eta * (U[o] - expected_u)
print("Δv_c from the gradient :", (V_new[:, c] - V[:, c]).round(4))
print("Δv_c from the notes    :", delta_vc_notes.round(4))

# --- 5.3 · A 2-D picture ---
# With d=2 we can draw everything: the output vectors u_k (one per word), the input vector
# h=v_c, and the update. v_c moves along u_o-bar u, from the average prediction bar u=Σ_i
# y_iu_i towards the true word u_o; the true word's u_o moves towards h, and the others move
# away from h in proportion to y_k.

rng2 = np.random.default_rng(3)
V2 = rng2.normal(0, 0.6, (2, 6)); U2 = rng2.normal(0, 1.0, (6, 2))
c2, o2 = 0, 3
dV2, dU2, _ = gradients(V2, U2, c2, o2)
_, _, y2, _ = forward(V2, U2, c2, o2)
h2 = V2[:, c2]
u_bar = y2 @ U2
print("y =", y2.round(3), "  true word o =", o2)

#  Plot: grey = output vectors u_k (true word in green), blue = input vector h=v_c and its
#  update (arrow), orange = the average prediction bar u. Small arrows on each u_k show their
#  updates.

fig, ax = plt.subplots(figsize=(6.5, 6))
step = 1.5
for k in range(6):
    col = C["green"] if k == o2 else C["muted"]
    ax.scatter(*U2[k], color=col, s=60, zorder=3)
    ax.annotate(f"u{k}" + (" (true)" if k == o2 else f"  y={y2[k]:.2f}"), U2[k], textcoords="offset points", xytext=(6, 4), color=col, fontsize=9)
    ax.annotate("", xy=U2[k] - step * dU2[k], xytext=U2[k], arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
ax.scatter(*h2, color=C["blue"], s=90, zorder=4); ax.annotate("h = v_c", h2, textcoords="offset points", xytext=(6, -12), color=C["blue"])
ax.annotate("", xy=h2 - step * dV2[:, c2], xytext=h2, arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2.2))
ax.scatter(*u_bar, color=C["orange"], s=70, marker="D", zorder=4)
ax.annotate("ū = Σ y_i u_i\n(average prediction)", u_bar, textcoords="offset points", xytext=(-20, -60), color=C["orange"], fontsize=9)
ax.annotate("", xy=U2[o2], xytext=u_bar, arrowprops=dict(arrowstyle="-|>", color=C["orange"], lw=1.2, ls="--"))
ax.text(h2[0] - 0.9, h2[1] + 0.12, "Δv_c ∥ u_o − ū", color=C["blue"], fontsize=9)
pts = np.vstack([U2, U2 - step * dU2, h2, h2 - step * dV2[:, c2], u_bar])
ax.set_xlim(pts[:, 0].min() - 0.4, pts[:, 0].max() + 0.4); ax.set_ylim(pts[:, 1].min() - 0.3, pts[:, 1].max() + 0.3)
ax.axhline(0, color=C["grid"]); ax.axvline(0, color=C["grid"]); ax.set_aspect("equal")
ax.set_title("one update: v_c moves from ū towards u_o")
plt.tight_layout(); plt.show()

# ==============================================================================
# 6 · How many numbers change per training pair?
# ==============================================================================
#
# From the boxes above, one skip-gram pair updates:
# * all n rows of U → n×d numbers;
# * one column of V → d numbers.
#
# The notes (page 14): "All d×n update ≈ 3×10^7" for d=300, n=10^5, "roughly 10^7 updates" per
# training pair – and a real corpus has billions of pairs. That cost is the subject of
# notebooks 06–08.

for d_, n_ in [(10, 75), (300, 10_000), (300, 100_000)]:
    print(f"d = {d_:>3}, n = {n_:>7,}:  U updates = {n_ * d_:>12,}   V updates = {d_:>4}   per training pair")

# ==============================================================================
# 7 · Summary
# ==============================================================================
#
#   Quantity  |  Formula  |  Code
#   error signal  |  ∂E/∂z=y-t  |  `e = y.copy(); e[o] -= 1`
#   output vectors  |  Δu_k=η(δ_ok-y_k)h^ᵀ  |  `U -= eta * np.outer(e, h)`
#   input vector  |  Δv_c=η(u_o-Σ_i y_iu_i)=-η U^ᵀ e  |  `V[:, c] -= eta * U.T @ e`
#   other input vectors  |  Δv_k=0  |  –
#   cost per pair  |  nd+d numbers  |  –
#
# --- Try it yourself ---
# 1. Derive ∂E/∂z_k for CBOW, where h=frac1rΣ[t] v_c_t. Which columns of V change, and by how
#    much?
# 2. Show that if y=t exactly (a perfect prediction) every gradient is zero.
# 3. Make one word's probability y_k≈0. Confirm its row of U barely changes ("no need to
#    change anything").
# 4. Repeat the SGD step in section 5.1 ten times. How quickly does y_o approach 1?
#
# Next → `06 - Naive Training`: train on the whole corpus with these exact updates.
