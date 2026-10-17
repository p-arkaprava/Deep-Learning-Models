# -*- coding: utf-8 -*-

"""
05 · Training a Small Network from Scratch

Python-script version of 05_training_a_small_network.ipynb.
Run:  python 05_training_a_small_network.py
(each figure opens in a window; close it to continue)
"""

# Module 02 – Feed-Forward Networks and Backpropagation · Notebook 5 of 6
# Course notes: pages 3–5 (sigmoid units, w_ji← w_ji+Δw_ji, backprop for output and hidden
# layers) · Mitchell Table 4.2 (the BACKPROPAGATION algorithm)
#
# Notebooks 01–04 built all the parts: shapes, forward pass, output-layer and hidden-layer
# gradients. Now we put them in a loop and actually train networks – first on XOR (the problem
# a single neuron can never solve), then on a tangled 3-class spiral.
#
# --- What you will learn ---
#   #  |  Idea
#   1  |  The training loop: shuffle → forward → loss → backward → update, repeated over epochs
#   2  |  Mini-batch stochastic gradient descent and momentum
#   3  |  Watching a 2-2-1 sigmoid network learn XOR, exactly as in the notes
#   4  |  Training a 2-64-64-3 network on a spiral and watching its decision regions form
#   5  |  How the learning rate, momentum and batch size change training

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
SPIRAL_CMAP = ListedColormap([C["blue"], C["orange"], C["aqua"]])
REGION3_CMAP = ListedColormap(["#d6e6f8", "#fcdccd", "#cdeee2"])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(Z):
    Z = Z - Z.max(axis=0, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=0, keepdims=True)

# ==============================================================================
# 1 · The variables
# ==============================================================================
#
# --- Notation for the whole of module 02 (same as the course notes) ---
#
#   Symbol  |  Shape  |  Meaning
#   L  |  number  |  number of layers with weights (hidden + output). The input is "layer 0".
#   n_l  |  number  |  number of neurons in layer l (n_0 = number of inputs)
#   vec x = o^([0])  |  n_0×1  |  one input example, a column vector
#   w_ji  |  number  |  weight on the wire from neuron i (layer l-1) to neuron j (layer l). First index = destination.
#   W^([l])  |  n_l×n_l-1  |  all weights of layer l; row j = the incoming weights of neuron j (notes: W^(h1)_5×4)
#   b^([l])  |  n_l×1  |  biases of layer l, one per neuron (the weight of the constant input x_0=1)
#   net^([l])=W^([l])o^([l-1])+b^([l])  |  n_l×1  |  net inputs of layer l; entry j is net_j=Σ_i w_ji o_i+b_j
#   o^([l])=g(net^([l]))  |  n_l×1  |  outputs (activations) of layer l; g = activation function, e.g. sigmoid
#   x_ji  |  number  |  the input that arrives at neuron j from i, i.e. x_ji=o_i (Mitchell / notes)
#   t_k  |  number  |  target for output neuron k
#   E  |  number  |  the error (loss) we minimise
#   δ_j=-(∂E)/(∂net_j)  |  number  |  error term of neuron j ("how much the error drops if net_j rises")
#   η  |  number  |  learning rate
#   N  |  number  |  batch size; a batch is X∈mathbb R^(n_0×N) with one example per column
#
# The notes also name layers by role: h1, h2 (hidden) and o (output), e.g.
# o^(h1)=σ(W^(h1)x+b^(h1)).
# New in this notebook:
#
#   Symbol  |  Meaning
#   epoch  |  one full pass over the training set
#   B  |  mini-batch size: number of examples used for one update
#   θ  |  "any parameter" (every W^([l]) and b^([l]))
#   g=∂E/∂θ  |  its gradient on the current mini-batch
#   μ  |  momentum coefficient, e.g. 0.9
#   v  |  velocity: running memory of past updates (same shape as θ)
#
# ==============================================================================
# 2 · The training loop
# ==============================================================================

fig, ax = plt.subplots(figsize=(13, 2.4))
steps = [("1. shuffle data,\ntake a mini-batch", C["muted"]), ("2. forward pass\nstore net, o", C["blue"]),
         ("3. loss E", C["orange"]), ("4. backward pass\nδ for every layer", C["red"]),
         ("5. update\nΔW = η δ oᵀ", C["green"])]
for i, (txt, col) in enumerate(steps):
    ax.add_patch(plt.Rectangle((i * 2.6, 0), 2.0, 1.0, fc="white", ec=col, lw=2.2))
    ax.text(i * 2.6 + 1.0, 0.5, txt, ha="center", va="center", fontsize=10.5)
    if i < len(steps) - 1:
        ax.annotate("", xy=(i * 2.6 + 2.55, 0.5), xytext=(i * 2.6 + 2.05, 0.5), arrowprops=dict(arrowstyle="-|>", lw=1.8))
ax.annotate("", xy=(1.0, -0.05), xytext=(4 * 2.6 + 1.0, -0.05),
            arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C["muted"], connectionstyle="arc3,rad=0.25"))
ax.text(6.2, -0.75, "repeat for every mini-batch, for many epochs", ha="center", color=C["muted"])
ax.set_xlim(-0.2, 12.4); ax.set_ylim(-1.0, 1.2); ax.axis("off")
plt.tight_layout(); plt.show()

# --- Mini-batch SGD ---
# The notes' E sums over the whole dataset D, which gives the exact gradient but only one
# update per epoch (batch GD, module 01). Using a random mini-batch of B examples per update
# gives a noisy but cheap gradient estimate and many updates per epoch. B=1 is the notes'
# stochastic/incremental gradient descent.
#
# --- Momentum ---
# Plain SGD: θ←θ-η g. With momentum we keep a velocity v:
#
#     v←μ v-η g, θ←θ+v
#
# Intuition: a ball rolling downhill. Consistent gradients add up and the ball speeds up along
# the valley; gradients that flip sign (zig-zag across a narrow valley, as in module 01)
# cancel out. With μ=0 it is plain SGD.
#
# ==============================================================================
# 3 · A compact MLP class
# ==============================================================================
#
# Everything from notebooks 02–04 in one class, in the notes' notation (columns = examples,
# W^([l]) is n_l×n_l-1, δ=-∂E/∂net). We build it one method per cell (attached with `MLP.name
# = function`, as in module 01) so each piece can be explained on its own.
#
# --- 3.1 · Skeleton: activations and parameters ---
#
# What `__init__` creates:
# * `self.W[l]`, `self.b[l]` – the weights and biases of each layer, with the shapes from
#   notebook 01. Weights use He scaling (sqrt2/n_l-1) for ReLU and Xavier (sqrt1/n_l-1)
#   otherwise (notebook 04); `init_scale` overrides this when we want a specific size (used
#   for XOR below).
# * `self.vW[l]`, `self.vb[l]` – the momentum velocities, one per parameter array, starting at
#   0 (no motion yet).
# * `ACT` – each hidden activation paired with its slope written in terms of the stored (net,
#   o), as in notebook 04.

class MLP:
    """Fully-connected network trained with backprop + mini-batch SGD (+ momentum).
    hidden: 'sigmoid' | 'tanh' | 'relu'      output: 'sigmoid_mse' (notes) | 'softmax_ce' (classification)"""

    ACT = {"sigmoid": (sigmoid,                       lambda net, o: o * (1 - o)),
           "tanh":    (np.tanh,                       lambda net, o: 1 - o ** 2),
           "relu":    (lambda z: np.maximum(0, z),    lambda net, o: (net > 0).astype(float))}

    def __init__(self, sizes, hidden="relu", output="softmax_ce", seed=0, init_scale=None):
        r = np.random.default_rng(seed)
        gain = 2.0 if hidden == "relu" else 1.0                      # He for ReLU, Xavier otherwise
        self.W = [r.normal(0, init_scale or np.sqrt(gain / sizes[l - 1]), (sizes[l], sizes[l - 1]))
                  for l in range(1, len(sizes))]                     # W^[l]: (n_l × n_{l-1})
        self.b = [np.zeros((n, 1)) for n in sizes[1:]]               # b^[l]: (n_l × 1)
        self.vW = [np.zeros_like(W) for W in self.W]                 # momentum velocities
        self.vb = [np.zeros_like(b) for b in self.b]
        self.hidden, self.output = hidden, output

# --- 3.2 · Forward pass (training-loop step 2) ---
#
# net^([l])=W^([l])o^([l-1])+b^([l]) layer by layer; hidden layers use the chosen activation,
# the last layer uses sigmoid or softmax depending on the output type. The lists `self.net`
# and `self.o` are the cache – `backward` reads them, so `forward` must always run first on
# the same batch (the notes: "F_i has to come before B_i").

def forward(self, X):
    """X: (n_0 × N). Stores net^[l] and o^[l] for backprop."""
    g = self.ACT[self.hidden][0]
    self.net, self.o = [None], [X]
    L = len(self.W)
    for l in range(L):
        NET = self.W[l] @ self.o[-1] + self.b[l]                 # net = W o_prev + b
        self.net.append(NET)
        if l < L - 1:
            self.o.append(g(NET))
        else:
            self.o.append(sigmoid(NET) if self.output == "sigmoid_mse" else softmax(NET))
    return self.o[-1]

MLP.forward = forward

# --- 3.3 · Loss (training-loop step 3) ---
#
# Only used for reporting progress – the gradient is written directly in `backward`. Both
# losses are averaged over the examples (`/ T.shape[1]`), so numbers are comparable across
# batch sizes. The `1e-12` avoids log 0.

def loss(self, O, T):
    if self.output == "sigmoid_mse":
        return 0.5 * np.sum((T - O) ** 2) / T.shape[1]           # ½ squared error, averaged per example
    return -np.sum(T * np.log(O + 1e-12)) / T.shape[1]           # mean cross-entropy

MLP.loss = loss

# --- 3.4 · Backward pass (training-loop step 4) ---
#
# Exactly notebook 04:
# 1. output delta – sigmoid + squared error: (T-O)⊙ O⊙(1-O); softmax + cross-entropy: T-P.
#    Both divided by N because the loss is a mean.
# 2. walk the layers backwards; gradient -δ (o^([l-1]))^ᵀ, bias gradient -Σ_colsδ;
# 3. pass the blame down: δ^([l-1])=g'⊙(W^([l]))^ᵀδ^([l]).
#
# Index note: `self.o[l]` is the input of weight layer `l` (Python lists start at 0, and
# `self.o[0]` is X).

def backward(self, T):
    """Returns gradients ∂E/∂W, ∂E/∂b for the batch (mean over examples)."""
    gprime = self.ACT[self.hidden][1]
    N, L = T.shape[1], len(self.W)
    O = self.o[-1]
    delta = (T - O) * O * (1 - O) / N if self.output == "sigmoid_mse" else (T - O) / N   # δ^[L]
    gW, gb = [None] * L, [None] * L
    for l in reversed(range(L)):
        gW[l] = -delta @ self.o[l].T                             # ∂E/∂W^[l] = -δ o_prevᵀ
        gb[l] = -delta.sum(axis=1, keepdims=True)
        if l > 0:
            delta = gprime(self.net[l], self.o[l]) * (self.W[l].T @ delta)   # δ^[l-1]
    return gW, gb

MLP.backward = backward

# --- 3.5 · Update with momentum (training-loop step 5) ---
#
# For every parameter array: first update the velocity v←μ v-η g, then move the parameter
# θ←θ+v. With `momentum=0` the velocity is just -η g, i.e. plain SGD θ←θ-η g (which equals the
# notes' ΔW=η δ o^ᵀ, since g=-δ o^ᵀ).

def step(self, gW, gb, lr, momentum=0.0):
    for l in range(len(self.W)):
        self.vW[l] = momentum * self.vW[l] - lr * gW[l]          # v ← μ v − η g
        self.vb[l] = momentum * self.vb[l] - lr * gb[l]
        self.W[l] += self.vW[l]                                  # θ ← θ + v
        self.b[l] += self.vb[l]

MLP.step = step

# --- 3.6 · One epoch (training-loop steps 1–5) ---
#
# Step 1 – shuffle and slice: `r.permutation(N)` is a new random order every epoch, so the
# mini-batches differ each time; `idx[s:s + batch]` takes the next B column indices (the last
# batch may be smaller). Then forward → backward → step on that batch.

def train_epoch(self, X, T, batch, lr, momentum, r):
    idx = r.permutation(X.shape[1])                              # 1. shuffle every epoch
    for s in range(0, X.shape[1], batch):
        cols = idx[s:s + batch]                                  #    take a mini-batch
        self.forward(X[:, cols])                                 # 2. forward
        gW, gb = self.backward(T[:, cols])                       # 3-4. loss gradient by backprop
        self.step(gW, gb, lr, momentum)                          # 5. update

MLP.train_epoch = train_epoch

# --- 3.7 · Prediction and evaluation ---
#
# * `predict` returns the class with the largest output (`argmax` over the rows of each
#   column).
# * `evaluate` runs the network on a whole dataset and returns loss and accuracy. For one
#   sigmoid output (XOR) a prediction counts as correct if it is on the right side of 0.5;
#   with several outputs we compare the `argmax` with the true class.

def predict(self, X):
    return self.forward(X).argmax(axis=0)


def evaluate(self, X, T):
    O = self.forward(X)
    acc = np.mean(O.argmax(0) == T.argmax(0)) if T.shape[0] > 1 else np.mean((O > 0.5) == T)
    return self.loss(O, T), acc

MLP.predict, MLP.evaluate = predict, evaluate

# --- 3.8 · The full training loop (`fit`) ---
#
# Repeat `train_epoch` for the requested number of epochs. After each epoch, record train (and
# optionally test) loss and accuracy in `hist`. `snapshot_fn` is an optional hook – a function
# called at chosen epochs whose result is stored in `self.snapshots`; we use it to save the
# decision regions for plotting, without mixing any plotting into the model.

def fit(self, X, T, X_test=None, T_test=None, epochs=100, batch=32, lr=0.1, momentum=0.0,
        seed=0, snapshot_epochs=(), snapshot_fn=None):
    r = np.random.default_rng(seed)
    hist = {k: [] for k in ("loss", "test_loss", "acc", "test_acc")}
    self.snapshots = {}
    for ep in range(1, epochs + 1):
        self.train_epoch(X, T, batch, lr, momentum, r)
        loss_tr, acc_tr = self.evaluate(X, T)                    # bookkeeping on the full sets
        hist["loss"].append(loss_tr); hist["acc"].append(acc_tr)
        if X_test is not None:
            loss_te, acc_te = self.evaluate(X_test, T_test)
            hist["test_loss"].append(loss_te); hist["test_acc"].append(acc_te)
        if ep in snapshot_epochs and snapshot_fn is not None:
            self.snapshots[ep] = snapshot_fn(self)
    return hist

MLP.fit = fit

# ==============================================================================
# 4 · XOR with a 2-2-1 sigmoid network (the notes' setting) ⊕
# ==============================================================================
#
# Sigmoid hidden units, a sigmoid output, squared error, and online updates (B=1) – exactly
# the algorithm in the notes and in Mitchell's Table 4.2.
#
# Step A – the data. The four XOR inputs as columns of a 2×4 matrix, and the targets as a 1×4
# row (one output neuron).

X_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=float)     # 4 examples as columns
T_xor = np.array([[0, 1, 1, 0]], dtype=float)

# Step B – a snapshot function for the plots. It evaluates the network on a 120 × 120 grid
# covering the input square; `fit` will call it at epochs 1, 100, 300 and 3000 and keep the
# results.

g = np.linspace(-0.5, 1.5, 120); gx, gy = np.meshgrid(g, g); G = np.vstack([gx.ravel(), gy.ravel()])
snap = lambda net: net.forward(G).reshape(gx.shape)               # output surface for plotting

# Step C – build and train. `[2, 2, 1]` with sigmoid everywhere and squared error; `batch=1`
# makes it online learning; `init_scale=1.0` gives the weights a standard deviation of 1 (a
# larger start helps the tiny network break symmetry); `seed=3` picks a start that converges
# (see the warning below).

xor_net = MLP([2, 2, 1], hidden="sigmoid", output="sigmoid_mse", seed=3, init_scale=1.0)
h_xor = xor_net.fit(X_xor, T_xor, epochs=3000, batch=1, lr=0.5, momentum=0.9,
                    snapshot_epochs=(1, 100, 300, 3000), snapshot_fn=snap)
print("outputs after training:", xor_net.forward(X_xor).ravel().round(3), "  targets:", T_xor.ravel())
print(f"loss: {h_xor['loss'][0]:.4f} → {h_xor['loss'][-1]:.5f}")

#  Plot: training loss (left) and the output surface at the four snapshot epochs, with the 0.5
#  contour in black.

fig, axs = plt.subplots(1, 5, figsize=(17, 3.6), gridspec_kw=dict(width_ratios=[1.4, 1, 1, 1, 1]))
axs[0].semilogy(h_xor["loss"], color=C["violet"])
axs[0].set_title("training loss"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("E (log)")
for ax, (ep, S) in zip(axs[1:], xor_net.snapshots.items()):
    ax.contourf(gx, gy, S, levels=np.linspace(0, 1, 11), cmap="RdBu_r", vmin=0, vmax=1)
    ax.contour(gx, gy, S, levels=[0.5], colors=C["ink"], linewidths=1.8)
    ax.scatter(X_xor[0], X_xor[1], c=T_xor.ravel().astype(int), cmap=ListedColormap([C["blue"], C["orange"]]),
               s=90, edgecolor="white", linewidth=2, zorder=3)
    ax.set_title(f"epoch {ep}"); ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
fig.suptitle("A 2-2-1 sigmoid network learning XOR: the 0.5 boundary bends into a band", fontweight="bold")
plt.tight_layout(); plt.show()

# The network found, by itself, a solution like the one we hand-built in module 01: two hidden
# "lines" whose combination makes a band.
#
# Local minima. With only 2 hidden units, some random starts get stuck (one hidden unit "dies"
# in a flat region and the output stays at 0.5 for two of the points). Try other `seed`
# values. Wider hidden layers make this far less likely – one practical reason networks are
# often bigger than strictly necessary.
#
# ==============================================================================
# 5 · A harder problem: the 3-armed spiral
# ==============================================================================
#
# 200 points per class wound around each other (480 for training, 120 held out for testing).
# No straight line (or two) can separate them.
#
# Step A – generate the spiral. For arm j: the radius grows linearly from the centre outwards
# and the angle sweeps 4 radians (plus noise), starting at a different angle for each arm.
# Converting (radius, angle) to (x_1,x_2) with sin/cos gives the arms.

def make_spiral(n=200, k=3, noise=0.2, seed=0):
    """k interleaved spiral arms, n points each. Returns X (2 × nk) and labels y (nk,)."""
    r = np.random.default_rng(seed)
    X = np.zeros((2, n * k)); y = np.zeros(n * k, dtype=int)
    for j in range(k):
        ix = slice(n * j, n * (j + 1))
        rad = np.linspace(0.05, 1.0, n)                                     # radius grows along the arm
        th = np.linspace(j * 4, (j + 1) * 4, n) + r.normal(0, noise, n)    # angle, with noise
        X[:, ix] = np.vstack([rad * np.sin(th), rad * np.cos(th)])
        y[ix] = j
    return X, y

# Step B – shuffle, split 80/20 and one-hot encode. `np.eye(3)[:, y]` picks column y of the
# identity matrix for every label, giving a 3×N matrix with a single 1 per column – the target
# format for softmax + cross-entropy.

X, y = make_spiral()
perm = np.random.default_rng(0).permutation(X.shape[1])
X, y = X[:, perm], y[perm]
n_tr = int(0.8 * X.shape[1])
Xtr, ytr, Xte, yte = X[:, :n_tr], y[:n_tr], X[:, n_tr:], y[n_tr:]
Ttr, Tte = np.eye(3)[:, ytr], np.eye(3)[:, yte]                            # one-hot targets as columns
print(f"train: {Xtr.shape[1]} points, test: {Xte.shape[1]} points, one-hot targets {Ttr.shape}")

#  Plot: the full dataset, coloured by class.

fig, ax = plt.subplots(figsize=(4.4, 4.2))
ax.scatter(X[0], X[1], c=y, cmap=SPIRAL_CMAP, s=14, edgecolor="white", linewidth=0.3)
ax.set_title("spiral dataset (3 classes)"); ax.set_aspect("equal"); ax.grid(False)
plt.tight_layout(); plt.show()

# --- Train a 2 → 64 → 64 → 3 network (ReLU hidden, softmax output) ---
#
# Snapshot function: the predicted class on a 250 × 250 grid, saved at epochs 1, 5, 20 and
# 150.

gs = np.linspace(-1.2, 1.2, 250); sx, sy = np.meshgrid(gs, gs); SG = np.vstack([sx.ravel(), sy.ravel()])
spiral_snap = lambda net: net.predict(SG).reshape(sx.shape)

# Train: 150 epochs, mini-batches of 32, η=0.05, momentum 0.9. The test set is only evaluated,
# never trained on.

net = MLP([2, 64, 64, 3], hidden="relu", output="softmax_ce", seed=0)
hist = net.fit(Xtr, Ttr, Xte, Tte, epochs=150, batch=32, lr=0.05, momentum=0.9,
               snapshot_epochs=(1, 5, 20, 150), snapshot_fn=spiral_snap)
print(f"final train acc = {hist['acc'][-1]:.3f} | test acc = {hist['test_acc'][-1]:.3f} | "
      f"train loss = {hist['loss'][-1]:.4f}")

#  Plot: loss and accuracy per epoch, for the training and the test set.

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
axs[0].plot(hist["loss"], color=C["blue"], label="train"); axs[0].plot(hist["test_loss"], color=C["orange"], label="test")
axs[0].set_yscale("log"); axs[0].set_title("cross-entropy loss"); axs[0].set_xlabel("epoch"); axs[0].legend()
axs[1].plot(hist["acc"], color=C["blue"], label="train"); axs[1].plot(hist["test_acc"], color=C["orange"], label="test")
axs[1].set_title("accuracy"); axs[1].set_xlabel("epoch"); axs[1].set_ylim(0.3, 1.02); axs[1].legend(loc="lower right")
plt.tight_layout(); plt.show()

#  Plot: the decision regions saved by the snapshot function.

fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
for ax, (ep, S) in zip(axs, net.snapshots.items()):
    ax.contourf(sx, sy, S, levels=[-0.5, 0.5, 1.5, 2.5], cmap=REGION3_CMAP)
    ax.scatter(Xtr[0], Xtr[1], c=ytr, cmap=SPIRAL_CMAP, s=9, edgecolor="white", linewidth=0.3)
    ax.set_title(f"after epoch {ep}"); ax.set_aspect("equal"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("decision regions forming during training", fontweight="bold")
plt.tight_layout(); plt.show()

# Early on the regions are nearly straight wedges; as training proceeds the network bends them
# to follow each arm. Every bit of that shape came from the same five-step loop.
#
# ==============================================================================
# 6 · The learning rate matters more than anything else
# ==============================================================================
#
# Plain SGD (no momentum), 60 epochs, five learning rates:
#
# Train five identical networks (same seed, so the same starting weights) that differ only in
# η. `np.errstate(all="ignore")` silences overflow warnings from the run that diverges.

lr_hist = {}
for lr in (0.001, 0.01, 0.1, 1.0, 5.0):
    m = MLP([2, 64, 64, 3], seed=0)
    with np.errstate(all="ignore"):
        hh = m.fit(Xtr, Ttr, Xte, Tte, epochs=60, batch=32, lr=lr)
    lr_hist[lr] = hh
    print(f"η = {lr:<6} final train loss = {hh['loss'][-1]:8.4f}   test acc = {hh['test_acc'][-1]:.3f}")

#  Plot: training loss per epoch for each learning rate.

fig, ax = plt.subplots(figsize=(8, 4))
for lr, col in zip(lr_hist, [C["muted"], C["yellow"], C["aqua"], C["blue"], C["red"]]):
    ax.plot(lr_hist[lr]["loss"], color=col, label=f"η = {lr}")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("train loss (log)"); ax.set_ylim(None, 5)
ax.set_title("too small = crawls, too big = unstable"); ax.legend(ncol=5, fontsize=9)
plt.tight_layout(); plt.show()

# ==============================================================================
# 7 · Momentum and batch size
# ==============================================================================
#
# Momentum experiment: same network and η=0.02, three momentum values.

mom_hist = {mu: MLP([2, 64, 64, 3], seed=0).fit(Xtr, Ttr, epochs=60, batch=32, lr=0.02, momentum=mu)
            for mu in (0.0, 0.5, 0.9)}
for mu, hh in mom_hist.items():
    print(f"μ = {mu}:  train loss after 60 epochs = {hh['loss'][-1]:.4f}")

# Batch-size experiment: η=0.05, μ=0.9, three batch sizes – 4, 32, and the whole training set
# (480, i.e. one update per epoch).

batch_hist = {B: MLP([2, 64, 64, 3], seed=0).fit(Xtr, Ttr, epochs=60, batch=B, lr=0.05, momentum=0.9)
              for B in (4, 32, Xtr.shape[1])}
for B, hh in batch_hist.items():
    print(f"B = {B:<4} ({int(np.ceil(Xtr.shape[1] / B)):>3} updates/epoch):  train loss after 60 epochs = {hh['loss'][-1]:.4f}")

#  Plot: loss curves for both experiments.

fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.9))
for mu, col in zip(mom_hist, [C["muted"], C["yellow"], C["blue"]]):
    axs[0].plot(mom_hist[mu]["loss"], color=col, label=f"μ = {mu}")
axs[0].set_yscale("log"); axs[0].set_title("momentum (η = 0.02, B = 32)"); axs[0].set_xlabel("epoch"); axs[0].legend()

for B, col in zip(batch_hist, [C["violet"], C["blue"], C["red"]]):
    hh = batch_hist[B]
    lbl = f"B = {B}" + ("  (full batch: 1 update/epoch)" if B == Xtr.shape[1] else f"  ({int(np.ceil(Xtr.shape[1] / B))} updates/epoch)")
    axs[1].plot(hh["loss"], color=col, label=lbl)
axs[1].set_yscale("log"); axs[1].set_title("mini-batch size (η = 0.05, μ = 0.9)"); axs[1].set_xlabel("epoch"); axs[1].legend(fontsize=9)
plt.tight_layout(); plt.show()

# * Momentum reaches a given loss in far fewer epochs at the same η.
# * Small batches make many cheap, noisy updates per epoch and progress fastest per epoch –
#   but noisily: see the spikes for B=4, where single unlucky mini-batches kick the weights
#   around; full batch makes one exact update per epoch and is slowest here. On a GPU, larger
#   batches are cheaper per example, so in practice a middle size (32–512) is a common
#   compromise.
#
# ==============================================================================
# 8 · Summary
# ==============================================================================
#
#   Ingredient  |  What it does
#   forward pass  |  o^([l])=g(W^([l])o^([l-1])+b^([l])), store everything
#   backward pass  |  δ^([L]) from the loss, δ^([l])=g'⊙(W^([l+1]))^ᵀδ^([l+1])
#   update  |  ΔW^([l])=η δ^([l])(o^([l-1]))^ᵀ, with momentum v←μ v-η g
#   mini-batches  |  many cheap noisy updates per epoch
#   results  |  XOR solved by a 2-2-1 sigmoid net; spiral: ~100% train / test accuracy with 2-64-64-3
#
# --- Try it yourself ---
# 1. Train XOR with `seed` values 0–9. How many get stuck? Repeat with a `[2, 4, 1]` network.
# 2. Switch the spiral network to `hidden="sigmoid"`. How do speed and final accuracy change?
#    (Remember notebook 04's vanishing-gradient plot.)
# 3. Add L2 weight decay to `step` (`g + λθ`) and watch the test loss.
# 4. Use `sklearn.datasets.load_digits` (8×8 digits) with `sizes=[64, 64, 10]`. What test
#    accuracy do you get?
#
# Next → `06 - Memory Footprint and Multi-GPU Training`: what happens when the network no
# longer fits on one GPU?
