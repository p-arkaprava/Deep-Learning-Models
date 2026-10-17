# -*- coding: utf-8 -*-

"""
02 · The Unthresholded Perceptron and the Delta Rule

Python-script version of 02_unthresholded_delta_rule.ipynb.
Run:  python 02_unthresholded_delta_rule.py
(each figure opens in a window; close it to continue)
"""

# Module 01 – Perceptron and Gradient Descent · Notebook 2 of 3
# Course notes: Unthresholded perceptron, cost function, gradient of cost function (pages 2–3)
# · Mitchell, Machine Learning §4.4
#
# Notebook 01 ended with two problems: a perceptron can only draw one line, and its step
# function has zero slope, so we can't use calculus to train it. Here we fix the second
# problem with one small change: throw the step away. The output becomes a real number, we can
# measure how wrong it is, and we can roll downhill on the error – this is gradient descent,
# the engine behind every model in this repository.
#
# --- What you will learn ---
#
#   #  |  Idea  |  Formula
#   1  |  The linear (unthresholded) unit  |  o(vec x)=vec w·vec x
#   2  |  One number for "how bad": the cost function  |  E(vec w)=tfrac12Σ[d∈ D] (t_d-o_d)^2
#   3  |  The gradient: which way is uphill?  |  ∇E=[(∂E)/(∂w_0),...,(∂E)/(∂w_n)]
#   4  |  Gradient descent / the delta rule  |  Δw_i=ηΣ_d (t_d-o_d) x_i,d
#   5  |  Batch vs stochastic (incremental) gradient descent  |  update after all data vs after each example
#   6  |  Choosing the learning rate η  |  too small = slow, too big = explodes

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
REGION_CMAP = ListedColormap(["#d6e6f8", "#fcdccd"])
POINT_CMAP  = ListedColormap([C["blue"], C["orange"]])
np.set_printoptions(precision=4, suppress=True)

# ==============================================================================
# 1 · The variables
# ==============================================================================
#
#   Symbol  |  Meaning  |  Example
#   D  |  the training set; d indexes one example in it  |  3 examples
#   vec x_d=(x_0,d,x_1,d,...,x_n,d)  |  input vector of example d, with x_0,d=1 (bias)  |  (1, 2)
#   x_i,d  |  input i of example d  |  x_1,2=2
#   t_d  |  target (true value) of example d  |  6
#   vec w=(w_0,...,w_n)  |  the weights we are learning  |  (-2, 4)
#   o_d=vec w·vec x_d  |  output of the linear unit for example d (a real number now!)  |  5.7
#   t_d-o_d  |  error on example d (signed)  |  0.3
#   E(vec w)  |  cost: one number summarising all errors  |  –
#   ∇E  |  gradient: vector of slopes of E, one per weight  |  –
#   η  |  learning rate, the step size  |  0.02
#
# ==============================================================================
# 2 · Remove the threshold
# ==============================================================================
#
#     thresholded: o=sgn(vec w·vec x)∈ -1,+1 Longrightarrow unthresholded: o(vec x)=vec w·vec x∈mathbb R
#
# Why does that help? Look at how the output reacts to a tiny change in net.  Plot: left, the
# output of each unit as a function of net; right, its slope do/dnet – the number calculus
# needs.

net = np.linspace(-3, 3, 601)
step = np.where(net > 0, 1.0, -1.0)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].plot(net[net <= 0], step[net <= 0], color=C["red"]); axs[0].plot(net[net > 0], step[net > 0], color=C["red"])
axs[0].plot(net, net, color=C["blue"], label="linear: o = net")
axs[0].plot([], [], color=C["red"], label="step: o = sgn(net)")
axs[0].set_title("output o as a function of net"); axs[0].set_xlabel("net"); axs[0].set_ylabel("o"); axs[0].legend()

axs[1].plot(net, np.zeros_like(net), color=C["red"], label="step: slope 0 (undefined at 0)")
axs[1].plot(net, np.ones_like(net), color=C["blue"], label="linear: slope 1 everywhere")
axs[1].scatter([0], [0], s=60, facecolor="white", edgecolor=C["red"], zorder=3)
axs[1].set_ylim(-0.3, 1.4); axs[1].set_title("slope do/dnet (what calculus needs)")
axs[1].set_xlabel("net"); axs[1].legend(loc="center right")
plt.tight_layout(); plt.show()

# Intuition. With the step, being "a bit wrong" and "very wrong" look identical, and nudging a
# weight changes nothing until the output suddenly flips. With the linear unit, an output of
# 5.7 against a target of 6 says "you are 0.3 too low" – and the slope tells us exactly how
# each weight affects that.
#
# ==============================================================================
# 3 · One number for "how bad": the cost function
# ==============================================================================
#
# From the course notes:
#
#     E(vec w)=frac12Σ[d∈ D] (t_d-o_d)^2
#
# * Why square? Positive and negative errors can't cancel, and big errors are punished much
#   more than small ones.
# * Why tfrac12? Pure convenience: when we differentiate (·)^2 a factor 2 appears and cancels
#   it.
# * Key point: the data D is fixed, so E is a function of the weights only. Learning = finding
#   the vec w where E is smallest.
#
# --- Our running example (from the notes) ---
# The notes list the data x = 1, 2, 3 → y = 2, 6, 10 and remark that the thresholded
# perceptron "won't work" (it can only say ±1). A linear unit can: the data lie exactly on y =
# 4x - 2, so the perfect weights are w_0=-2, w_1=4 with E=0.
#
# Step A – the data. Each example becomes a row (x_0, x_1)=(1, x); the column of ones is the
# bias input, so w_0 is learned like any other weight.

x_notes = np.array([1.0, 2.0, 3.0])
t_notes = np.array([2.0, 6.0, 10.0])
X_notes = np.c_[np.ones(3), x_notes]               # add the bias input x0 = 1 → rows (1, x)
print("X (one example per row):\n", X_notes, "\n targets t =", t_notes)

# Step B – the cost function. `X @ w` computes o_d=vec w·vec x_d for every row at once (a
# matrix-vector product), `t - o` gives the three errors, and `0.5 * np.sum(... ** 2)` is
# tfrac12Σ_d (t_d-o_d)^2.

def cost(w, X, t):
    """E(w) = 1/2 * sum_d (t_d - o_d)^2   with   o_d = w · x_d"""
    o = X @ w                                      # outputs for all examples at once
    return 0.5 * np.sum((t - o) ** 2)

# Step C – try three weight vectors. The cost should be largest for "no opinion" (0,0), small
# for a near miss, and exactly 0 for the true line (-2,4).

for w_try in [np.array([0.0, 0.0]), np.array([0.0, 3.0]), np.array([-2.0, 4.0])]:
    print(f"w = {w_try}:  outputs o = {X_notes @ w_try},  E(w) = {cost(w_try, X_notes, t_notes):.2f}")

# Because o is linear in vec w and E squares it, E(vec w) is a bowl (a paraboloid) with a
# single lowest point. Let us draw it over the (w_0, w_1) plane.
#
# Compute the surface. We evaluate E on a 241 × 241 grid of (w_0,w_1) pairs. Instead of a
# double loop, broadcasting does it in one line: `w0g[..., None] + w1g[..., None] * x_notes`
# adds a third axis of length 3 (one entry per example), giving every output o_d for every
# grid point; summing the squared errors over that last axis gives E.

w0g, w1g = np.meshgrid(np.linspace(-8, 4, 241), np.linspace(-1, 8, 241))
O = w0g[..., None] + w1g[..., None] * x_notes          # outputs o_d for every grid point, shape (241, 241, 3)
Egrid = 0.5 * np.sum((t_notes - O) ** 2, axis=-1)      # E(w0, w1), shape (241, 241)
print("E on the grid: min =", Egrid.min().round(4), " max =", Egrid.max().round(1))

#  Plot: the same surface as a 3-D bowl (left, walls cut at E=150 so the shape is visible) and
#  as contour lines seen from above (right).

fig = plt.figure(figsize=(12.5, 4.8))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
Eclip = np.where(Egrid < 150, Egrid, np.nan)                 # cut the walls so the bowl shape is visible
ax3.plot_surface(w0g, w1g, Eclip, cmap="viridis", linewidth=0, alpha=0.92, rstride=3, cstride=3, vmax=150)
ax3.scatter([-2], [4], [0], color=C["red"], s=50)
ax3.set_xlabel("$w_0$"); ax3.set_ylabel("$w_1$"); ax3.set_zlabel("E(w)"); ax3.set_zlim(0, 150)
ax3.set_title("the error surface is a bowl"); ax3.view_init(38, -115)

ax = fig.add_subplot(1, 2, 2)
cs = ax.contour(w0g, w1g, Egrid, levels=np.geomspace(0.05, 400, 16), cmap="viridis", linewidths=1.2)
ax.plot(-2, 4, "*", color=C["red"], ms=16, label="minimum  w = (-2, 4),  E = 0")
ax.set_xlabel("$w_0$ (bias)"); ax.set_ylabel("$w_1$"); ax.set_title("same bowl seen from above (contours of E)")
ax.legend(loc="upper right"); ax.grid(False)
plt.tight_layout(); plt.show()

# The contours are long, thin ellipses: the bowl is steep in one direction and nearly flat in
# another. Remember this – it will explain the zig-zag path of gradient descent below.
#
# ==============================================================================
# 4 · Which way is downhill? The gradient
# ==============================================================================
#
# Imagine standing on the bowl in thick fog. You can't see the bottom, but you can feel the
# slope under your feet. The gradient collects the slope in every weight direction:
#
#     ∇E(vec w)=[(∂E)/(∂w_0), (∂E)/(∂w_1), ..., (∂E)/(∂w_n)]
#
# It points uphill (steepest increase). So we step the opposite way:
#
#     vec w←vec w+Δvec w, Δvec w=-η ∇E(vec w)
#
# --- Deriving ∂E/∂w_i step by step ---
#
#     beginaligned (∂E)/(∂w_i) &=(∂)/(∂w_i) frac12Σ[d] (t_d-o_d)^2 &&definition of E; &=frac12Σ_d 2 (t_d-o_d) (∂)/(∂w_i)(t_d-o_d) &&chain rule; the tfrac12 cancels the 2; &=Σ_d (t_d-o_d) (∂)/(∂w_i)(t_d-vec w·vec x_d) &&o_d=vec w·vec x_d; &=Σ_d (t_d-o_d) (-x_i,d) &&t_d does not depend on w_i; (∂)/(∂w_i)(vec w·vec x_d)=x_i,d endaligned
#
#     (∂E)/(∂w_i)=-Σ[d∈ D] (t_d-o_d) x_i,d Longrightarrow Δw_i=-η(∂E)/(∂w_i)=ηΣ[d∈ D] (t_d-o_d) x_i,d
#
# This is the delta rule (also called the LMS or Widrow–Hoff rule). Compare it with the
# perceptron rule Δw_i=η(t-o)x_i – same shape! The difference is that o is now a real number,
# so (t-o) measures how much we are wrong, not just whether.
#
# In matrix form for all weights at once (rows of X are examples): ∇E = -X^ᵀ(vec t-vec o).
#
# --- Check the formula against a "brute force" slope ---
# A derivative is the slope of a tiny step: (∂E)/(∂w_i)≈(E(w_i+ε)-E(w_i-ε))/(2ε). If our
# algebra is right, both must agree.
#
# The analytic gradient. `X @ w` gives the outputs, `t - X @ w` the errors, and `X.T @ errors`
# multiplies each error by the input it came with and sums over examples – exactly Σ_d (t_d-
# o_d)x_i,d for every i. The leading minus sign comes from the derivation.

def gradient(w, X, t):
    """∇E = - sum_d (t_d - o_d) x_d   (one entry per weight)   = -X^T (t - o)"""
    return -X.T @ (t - X @ w)

# The brute-force gradient. For each weight separately: nudge it up by a tiny ε, nudge it down
# by ε, and measure how much E changed. No calculus involved – so if it agrees with the
# formula, the formula is right.

def numerical_gradient(w, X, t, eps=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = eps                                # nudge only weight i
        g[i] = (cost(w + e, X, t) - cost(w - e, X, t)) / (2 * eps)   # central difference
    return g

# Compare them at an arbitrary point vec w=(0.5,1.5):

w_test = np.array([0.5, 1.5])
print("analytic  ∇E :", gradient(w_test, X_notes, t_notes))
print("numerical ∇E :", numerical_gradient(w_test, X_notes, t_notes))
print("match?       :", np.allclose(gradient(w_test, X_notes, t_notes), numerical_gradient(w_test, X_notes, t_notes)))

# Both entries are negative → increasing w_0 and w_1 lowers the error, so gradient descent
# will increase them. That makes sense: vec w=(0.5,1.5) predicts outputs that are too small.
#
# ==============================================================================
# 5 · Implementation: batch and stochastic gradient descent
# ==============================================================================
#
# There are two ways to use the delta rule:
#
#     |  Batch gradient descent  |  Stochastic / incremental gradient descent
#   Update after  |  all examples (sum over D)  |  each example
#   Update  |  Δw_i=ηΣ_d (t_d-o_d)x_i,d  |  Δw_i=η(t_d-o_d)x_i,d
#   Follows  |  the true gradient of E  |  a noisy estimate of it
#   Cost per update  |  high (one pass over the data)  |  tiny
#   Path  |  smooth  |  jittery, but often reaches a good region faster
#
# We build the class in five small cells, one job per cell. (As in notebook 01, a method can
# be written as a normal function and attached afterwards with `LinearUnit.name = function`.)
#
# --- 5.1 · Skeleton: settings, bias input and the cost ---
#
# Why these three live together: they are the pieces both training modes share.
# * `__init__` stores the settings: η, the number of epochs, which mode to use, and a random
#   generator. The generator gets a fixed `seed`, so the "random" order of stochastic GD is
#   the same every run (results are reproducible).
# * `add_bias` adds the column of ones (x_0=1).
# * `cost` evaluates E(vec w)=tfrac12Σ_d (t_d-o_d)^2 for the current weights; we call it once
#   per epoch to track progress.

class LinearUnit:
    """Unthresholded perceptron  o = w·x,  trained with the delta rule (gradient descent on E)."""

    def __init__(self, learning_rate=0.01, epochs=50, mode="batch", seed=0):
        self.eta = learning_rate      # η : step size
        self.epochs = epochs          # passes over the data
        self.mode = mode              # "batch" or "stochastic"
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def add_bias(X):
        return np.c_[np.ones((X.shape[0], 1)), X]          # x0 = 1 for every example

    def cost(self, Xb, t):
        return 0.5 * np.sum((t - Xb @ self.w) ** 2)       # E(w) = 1/2 Σ (t_d - o_d)^2

# --- 5.2 · One batch step ---
#
# What it does: compute the outputs for all examples, form the exact gradient ∇E=-X^ᵀ(vec
# t-vec o), and take one step against it: vec w←vec w-η∇E.
#
# Why `-Xb.T @ (t - o)` is the sum: row i of `Xb.T` holds x_i,d for every example d, so the
# product gives Σ_d x_i,d(t_d-o_d) for every weight i at once.

def batch_step(self, Xb, t):
    """Delta rule, batch version: ONE update using the gradient summed over all examples."""
    o = Xb @ self.w                            # outputs for ALL examples
    grad = -Xb.T @ (t - o)                     # ∂E/∂w_i = -Σ_d (t_d - o_d) x_id
    self.w = self.w - self.eta * grad          # one step downhill
    self.w_history.append(self.w.copy())

LinearUnit.batch_step = batch_step

# --- 5.3 · One stochastic epoch ---
#
# What it does: visit the examples one at a time and update after each one, using only that
# example's error: Δvec w=η(t_d-o_d)vec x_d.
#
# Why shuffle? `self.rng.permutation(len(t))` gives a fresh random order every epoch. A fixed
# order can make the weights drift in a pattern tied to the order of the data; shuffling
# removes that bias.

def stochastic_epoch(self, Xb, t):
    """Delta rule, incremental version: one small update per example, in random order."""
    for d in self.rng.permutation(len(t)):
        o_d = Xb[d] @ self.w                                   # output for example d only
        self.w = self.w + self.eta * (t[d] - o_d) * Xb[d]      # Δw = η (t_d - o_d) x_d
        self.w_history.append(self.w.copy())

LinearUnit.stochastic_epoch = stochastic_epoch

# --- 5.4 · The training loop (`fit`) ---
#
# What it does: add the bias column, start the weights at 0 (or at `w_init` if given), then
# repeat for each epoch: one batch step or one stochastic epoch, followed by recording the
# cost.
#
# `cost_history` and `w_history` are bookkeeping for the plots; the algorithm itself is only
# the two step functions above. At the end `w_history` becomes a NumPy array so we can slice
# it (`w_history[:400]`) when plotting paths.

def fit(self, X, t, w_init=None):
    Xb = self.add_bias(X)
    self.w = np.zeros(Xb.shape[1]) if w_init is None else np.array(w_init, float)
    self.cost_history = [self.cost(Xb, t)]             # E before any training
    self.w_history = [self.w.copy()]
    for epoch in range(self.epochs):
        if self.mode == "batch":
            self.batch_step(Xb, t)                     # 1 update per epoch
        else:
            self.stochastic_epoch(Xb, t)               # N updates per epoch
        self.cost_history.append(self.cost(Xb, t))     # E after this epoch
    self.w_history = np.array(self.w_history)
    return self

LinearUnit.fit = fit

# --- 5.5 · Prediction ---
#
# The linear unit's output is simply o=vec w·vec x – no threshold – so prediction is one
# matrix-vector product.

def predict(self, X):
    return self.add_bias(X) @ self.w

LinearUnit.predict = predict

# ==============================================================================
# 6 · Watch it learn on the notes' data, step by step
# ==============================================================================
#
# We repeat `batch_step` by hand for 3 steps and print every intermediate quantity – outputs,
# errors, gradient, new weights and cost – so each one can be checked with pen and paper.

Xb = LinearUnit.add_bias(x_notes[:, None])     # rows (1, x)
w = np.zeros(2)
print("start: w =", w, " E =", cost(w, Xb, t_notes))
for step in range(1, 4):
    o = Xb @ w
    err = t_notes - o
    grad = -Xb.T @ err
    w = w - 0.02 * grad
    print(f"step {step}: o = {o}, t - o = {err}, ∇E = {grad}, new w = {w}, E = {cost(w, Xb, t_notes):.3f}")

# Check step 1 by hand: o=(0,0,0), errors =(2,6,10).
# ∂E/∂w_0=-(2+6+10)=-18 and ∂E/∂w_1=-(2·1+6·2+10·3)=-44.
# With η=0.02: Δvec w=(0.36, 0.88).
#
# Now run it for many epochs, in both modes. `x_notes[:, None]` turns the 1-D array of inputs
# into a column (3 × 1), because the class expects one example per row.

batch = LinearUnit(learning_rate=0.02, epochs=1500, mode="batch").fit(x_notes[:, None], t_notes)
sgd   = LinearUnit(learning_rate=0.02, epochs=500, mode="stochastic").fit(x_notes[:, None], t_notes)
print(f"batch GD      → w = {batch.w}   E = {batch.cost_history[-1]:.2e}")
print(f"stochastic GD → w = {sgd.w}   E = {sgd.cost_history[-1]:.2e}   (constant η keeps it jittering near the minimum)")
print("exact answer  → w = [-2.  4.]")

#  Plot: left, the first 400 weight vectors of each run drawn on top of the contour map of E;
#  right, the cost after every epoch (log scale).

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axs[0]
ax.contour(w0g, w1g, Egrid, levels=np.geomspace(0.05, 400, 16), cmap="viridis", linewidths=1)
ax.plot(*batch.w_history[:400].T, ".-", color=C["blue"], ms=3, lw=1.2, label="batch GD (first 400 steps)")
ax.plot(*sgd.w_history[:400].T, ".-", color=C["orange"], ms=2, lw=0.8, alpha=0.8, label="stochastic GD (first 400 steps)")
ax.plot(0, 0, "o", color=C["ink"], ms=7, label="start (0, 0)")
ax.plot(-2, 4, "*", color=C["red"], ms=16, label="minimum (-2, 4)")
ax.set_xlim(-4, 1.5); ax.set_ylim(-0.5, 5.5); ax.grid(False)
ax.set_xlabel("$w_0$"); ax.set_ylabel("$w_1$"); ax.set_title("paths across the error bowl"); ax.legend(loc="lower left", fontsize=9)

ax = axs[1]
ax.semilogy(batch.cost_history, color=C["blue"], label="batch GD (1 update / epoch)")
ax.semilogy(np.arange(len(sgd.cost_history)) , sgd.cost_history, color=C["orange"], label="stochastic GD (3 updates / epoch)")
ax.set_xlabel("epoch"); ax.set_ylabel("E(w)  (log scale)"); ax.set_title("cost falls towards 0"); ax.legend()
plt.tight_layout(); plt.show()

# What you see:
# 1. Both paths first rush down the steep wall of the bowl, then crawl along the flat valley
#    floor towards (-2,4). The steep direction is learned in a few steps; the flat direction
#    takes hundreds.
# 2. The straight-line fit through the data improves at every stage.  Plot: the line
#    o=w_0+w_1x after 1, 5, 30 and 1500 epochs, taken from `batch.w_history`.

fig, ax = plt.subplots(figsize=(6.5, 4))
xs = np.linspace(0, 4, 50)
for k, col in zip([1, 5, 30, 1500], [C["yellow"], C["magenta"], C["violet"], C["blue"]]):
    wk = batch.w_history[k]
    ax.plot(xs, wk[0] + wk[1] * xs, color=col, label=f"after {k} epochs:  o = {wk[0]:.2f} + {wk[1]:.2f} x")
ax.scatter(x_notes, t_notes, s=90, color=C["red"], zorder=3, edgecolor="white", linewidth=2, label="data from the notes")
ax.set_xlabel("x"); ax.set_ylabel("o"); ax.set_title("the fitted line during batch gradient descent"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# ==============================================================================
# 7 · The learning rate η: the most important knob
# ==============================================================================
#
# η controls the step length. For this bowl, any η above about 0.12 makes the steps overshoot
# across the valley and grow – the cost explodes.
#
# Train once per learning rate (same data, same start, 60 epochs each) and keep the models:

etas = {0.002: "too small: crawls", 0.02: "good", 0.11: "large: zig-zags (still converges)", 0.125: "too large: diverges"}
lr_models = {}
for eta, label in etas.items():
    lr_models[eta] = LinearUnit(learning_rate=eta, epochs=60, mode="batch").fit(x_notes[:, None], t_notes)
    print(f"η = {eta:<6} ({label:<34}) E after 60 epochs = {lr_models[eta].cost_history[-1]:.3g}")

#  Plot: each model's path across the bowl. For the diverging run only the first steps are
#  drawn – after that the weights fly off the chart.

fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
for ax, (eta, label) in zip(axs, etas.items()):
    m = lr_models[eta]
    ax.contour(w0g, w1g, Egrid, levels=np.geomspace(0.05, 400, 14), cmap="viridis", linewidths=0.8)
    path = m.w_history[:18] if "diverges" in label else m.w_history   # diverging path: first steps only
    ax.plot(*path.T, ".-", color=C["red"] if "diverges" in label else C["blue"], ms=4, lw=1)
    ax.plot(-2, 4, "*", color=C["red"], ms=13)
    ax.set_xlim(-6, 3); ax.set_ylim(-1.5, 8); ax.grid(False)
    ax.set_title(f"η = {eta}: {label}\nE after 60 epochs = {m.cost_history[-1]:.3g}", fontsize=10)
    ax.set_xlabel("$w_0$")
axs[0].set_ylabel("$w_1$")
fig.suptitle("Same bowl, same start, four learning rates", fontweight="bold")
plt.tight_layout(); plt.show()

# Rule of thumb: start with a small η, increase it by ×3 until the cost stops falling
# smoothly, then back off. (Why exactly 0.12 here? The curvature of the bowl in its steepest
# direction is the largest eigenvalue λ_max of X^ᵀ X≈16.7; plain gradient descent is stable
# only if η<2/λ_max≈0.12.)
#
# ==============================================================================
# 8 · Batch vs stochastic on a bigger, noisy dataset
# ==============================================================================
#
# Now 100 noisy points around y=3x+5. With many examples, stochastic GD gets 100 updates per
# epoch instead of 1.
#
# Step A – the data: 100 inputs evenly spaced in [0,10], targets 3x+5 plus Gaussian noise with
# standard deviation 2.

rng = np.random.default_rng(42)
X_data = np.linspace(0, 10, 100).reshape(-1, 1)            # 100 × 1: one example per row
t_data = 3 * X_data.ravel() + 5 + rng.normal(0, 2, 100)    # noisy targets around the line 3x + 5
print("first 3 examples:", X_data[:3].ravel(), "→", t_data[:3].round(2))

# Step B – train both modes and compare with the exact answer. The learning rates differ
# because a batch step sums 100 gradients (so it must be smaller), while a stochastic step
# uses one. `np.linalg.lstsq` solves the least-squares problem directly; it is the lowest E
# any line can reach.

batch_n = LinearUnit(learning_rate=4e-4, epochs=100, mode="batch").fit(X_data, t_data)
sgd_n   = LinearUnit(learning_rate=2e-3, epochs=100, mode="stochastic").fit(X_data, t_data)
w_exact = np.linalg.lstsq(np.c_[np.ones(100), X_data], t_data, rcond=None)[0]   # best possible weights
print(f"batch GD      w = {batch_n.w}   E = {batch_n.cost_history[-1]:8.1f}")
print(f"stochastic GD w = {sgd_n.w}   E = {sgd_n.cost_history[-1]:8.1f}")
print(f"least squares w = {w_exact}   E = {0.5 * np.sum((t_data - np.c_[np.ones(100), X_data] @ w_exact) ** 2):8.1f}  (the true minimum)")

#  Plot: cost per epoch for both modes (left) and the two fitted lines over the data (right).

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogy(batch_n.cost_history, color=C["blue"], label="batch GD")
axs[0].semilogy(sgd_n.cost_history, color=C["orange"], label="stochastic GD")
axs[0].axhline(0.5 * np.sum((t_data - np.c_[np.ones(100), X_data] @ w_exact) ** 2), color=C["muted"], ls="--", lw=1.2, label="lowest possible E")
axs[0].set_title("cost vs epochs"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("E(w)  (log scale)"); axs[0].legend()

axs[1].scatter(X_data, t_data, s=14, color=C["muted"], alpha=0.6, label="data")
xs = np.linspace(0, 10, 2)
axs[1].plot(xs, batch_n.w[0] + batch_n.w[1] * xs, color=C["blue"], label=f"batch: {batch_n.w[0]:.2f} + {batch_n.w[1]:.2f}x")
axs[1].plot(xs, sgd_n.w[0] + sgd_n.w[1] * xs, color=C["orange"], label=f"stochastic: {sgd_n.w[0]:.2f} + {sgd_n.w[1]:.2f}x")
axs[1].set_title("fitted lines after 100 epochs"); axs[1].set_xlabel("x"); axs[1].set_ylabel("t"); axs[1].legend(fontsize=9)
plt.tight_layout(); plt.show()

# Reading the plots. After 100 epochs, batch GD has the slope (≈ 3) right but its intercept is
# still crawling up from 0 (the flat valley again). Stochastic GD made 100× more updates and
# is already near the best line, but its cost wiggles because each step only listens to one
# example.
#
# ==============================================================================
# 9 · The delta rule for classification (and XOR again)
# ==============================================================================
#
# We can also use the linear unit as a classifier: train it on ±1 targets with the delta rule,
# then apply the threshold only when predicting. Two big differences from the perceptron:
# * It always converges (to the least-squares weights), even when the data are not separable –
#   no endless cycling.
# * But on XOR the best straight line is useless: the least-squares answer is vec w=vec 0.
#
# Train on AND and XOR. Training uses the raw output (delta rule); only the prediction is
# thresholded with `np.where(o > 0, 1, -1)`.

X_gate = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
gate_targets = {"AND": np.array([-1, -1, -1, 1.0]), "XOR": np.array([-1, 1, 1, -1.0])}

gate_models = {}
for name, tg in gate_targets.items():
    m = LinearUnit(learning_rate=0.05, epochs=500, mode="batch").fit(X_gate, tg)
    pred = np.where(m.predict(X_gate) > 0, 1, -1)          # threshold only at prediction time
    gate_models[name] = m
    print(f"{name}: w = {m.w + 0.0},  raw outputs o = {m.predict(X_gate) + 0.0},  "
          f"thresholded = {pred},  accuracy = {(pred == tg).mean():.0%}")

#  Plot: colour = the raw output o over the input plane; black line = where o=0 (drawn only if
#  the weights are not all zero).

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (name, tg) in zip(axs, gate_targets.items()):
    m = gate_models[name]
    g = np.linspace(-0.6, 1.6, 300); xx, yy = np.meshgrid(g, g)
    oo = m.w[0] + m.w[1] * xx + m.w[2] * yy
    ax.contourf(xx, yy, oo, levels=20, cmap="RdBu_r", alpha=0.55, vmin=-1.5, vmax=1.5)
    if np.abs(m.w[1:]).max() > 1e-6:
        ax.contour(xx, yy, oo, levels=[0], colors=C["ink"], linewidths=2)
    ax.scatter(X_gate[:, 0], X_gate[:, 1], c=(tg > 0).astype(int), cmap=POINT_CMAP, s=120, edgecolor="white", linewidth=2, zorder=3)
    ax.set_title(f"{name}: colour = raw output o, black line = o = 0", fontsize=10)
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
plt.tight_layout(); plt.show()

# For AND the line at o=0 separates the classes. For XOR every output is 0 – the "best" line
# sits nowhere. A single linear unit still cannot do XOR. We need non-linear units stacked in
# layers – that is notebook 03.
#
# ==============================================================================
# 10 · Summary
# ==============================================================================
#
#   Concept  |  Formula
#   Linear unit  |  o=vec w·vec x
#   Cost  |  E(vec w)=tfrac12Σ_d (t_d-o_d)^2
#   Gradient  |  (∂E)/(∂w_i)=-Σ_d (t_d-o_d)x_i,d
#   Batch delta rule  |  Δw_i=ηΣ_d (t_d-o_d)x_i,d
#   Stochastic delta rule  |  Δw_i=η(t_d-o_d)x_i,d, after every example
#   Stability  |  η must be small enough (η<2/λ_max for batch GD)
#
# Perceptron rule vs delta rule: same formula shape, but the perceptron uses the thresholded
# output (and stops only if the data are separable), while the delta rule uses the raw output
# (and always converges to the least-squares fit).
#
# Still missing: a network of linear units is no better than one linear unit (notes: "network
# of linear neurons will have same effect as a single neuron"). We need non-linearity →
# sigmoid.
#
# --- Try it yourself ---
# 1. Find the largest η that still converges on the notes' data. Compare with 2/λ_max from
#    `np.linalg.eigvalsh(Xb.T @ Xb)`.
# 2. Standardise x (subtract the mean, divide by the std) before training on the noisy data.
#    Why does batch GD suddenly converge in a few epochs? (Hint: the bowl becomes round.)
# 3. Implement mini-batch GD (update after every 10 examples) and compare its cost curve with
#    the two above.
# 4. Replace tfrac12Σ by the mean tfrac12NΣ. How must η change to get the same steps?
#
# Next → `03 - Sigmoid Unit and Non-linearity`
