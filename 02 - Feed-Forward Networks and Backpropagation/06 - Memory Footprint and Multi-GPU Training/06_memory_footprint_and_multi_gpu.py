# -*- coding: utf-8 -*-

"""
06 · Memory Footprint and Multi-GPU Training

Python-script version of 06_memory_footprint_and_multi_gpu.ipynb.
Run:  python 06_memory_footprint_and_multi_gpu.py
(each figure opens in a window; close it to continue)
"""

# Module 02 – Feed-Forward Networks and Backpropagation · Notebook 6 of 6
# Course notes: page 7 "Memory footprint of FFNN" (4-5-3-2, 51 F.P., inference peak 51+4+5,
# training 51+4+5+3+2), page 5 "Model size: 1 GPU / multiple … each GPU collects local
# updates, collect all updates together", "F_i has to come before B_i", page 6 "Divide by
# layers … pipelined", "weight matrix too big … split weight matrices"
#
# Everything so far fit comfortably in a laptop's memory. Real models do not: a 7-billion-
# parameter model cannot even be trained on one 80 GB GPU. This notebook answers two practical
# questions:
#
# 1. Will it fit? – estimate memory before running anything.
# 2. If not, how do we split the work across GPUs? – data, ZeRO, pipeline and tensor
#    parallelism.
#
# --- What you will learn ---
#   #  |  Idea
#   1  |  Why training needs more memory than inference (the notes' 4-5-3-2 example, step by step)
#   2  |  The 16 bytes per parameter rule for Adam training, plus activations
#   3  |  A memory estimator, checked against real NumPy arrays
#   4  |  Data parallelism and why averaging local gradients gives exactly the right answer
#   5  |  Ring all-reduce, ZeRO / FSDP sharding
#   6  |  Pipeline parallelism (divide by layers) and its "bubble"
#   7  |  Tensor parallelism (split one weight matrix)

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
rng = np.random.default_rng(3)

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
#   Symbol  |  Meaning  |  Example
#   P  |  number of parameters (weights + biases)  |  51
#   bytes / number  |  float32 = 4 bytes, float16/bfloat16 = 2 bytes  |  4
#   N  |  batch size (examples processed together)  |  256
#   n_l  |  neurons in layer l (activations per example)  |  5
#   K  |  number of GPUs (workers)  |  8
#   g_k  |  gradient computed by GPU k on its slice of the batch  |  –
#   N_k  |  number of examples given to GPU k  |  N/K
#   m, v  |  Adam's two running averages (one of each per parameter)  |  –
#   F_i, B_i  |  forward / backward pass of micro-batch i (notes, page 5)  |  –
#
# Network notation (W^([l]) is n_l×n_l-1, one example per column) is the same as notebooks
# 01–05.
#
# ==============================================================================
# 2 · Inference vs training memory: the notes' example
# ==============================================================================
#
# The 4-5-3-2 network has (20+5)+(15+3)+(6+2)=51 parameters (notebook 01). Count the numbers
# that must be in memory at the same time (one example):
#
# * Inference (forward only). Layer l only needs o^([l-1]) to compute o^([l]); afterwards
#   o^([l-1]) can be thrown away (the notes: "no need to keep"). The worst moment is layer 1,
#   where the 4 inputs and 5 hidden outputs coexist → peak 51+4+5=60.
# * Training. Backprop needs every o^([l]) again (for δ^([l])(o^([l-1]))^ᵀ and for g'), so
#   nothing can be thrown away → peak 51+4+5+3+2=65 (notes: "don't need output, but at peak we
#   need all").
#
# Let us simulate the forward pass and track the live count.
#
# Step A – the parameter count (formula from notebook 01):

sizes = [4, 5, 3, 2]
P = sum(sizes[l] * sizes[l - 1] + sizes[l] for l in range(1, len(sizes)))
print("P =", P)

# Step B – simulate the forward pass, event by event. `live` is a dictionary of the activation
# vectors currently held in memory (name → number of values). After every event we record
# "weights + everything in `live`".
# * computing o^([l]) adds n_l numbers;
# * in inference (`keep_all=False`) the input of the layer, o^([l-1]), is deleted right after
#   use;
# * in training (`keep_all=True`) nothing is deleted, because backprop will need it.

def live_memory(sizes, keep_all):
    """Numbers alive in memory after each event of a forward pass (weights always loaded)."""
    timeline, live = [("load weights", P)], {}
    live["x"] = sizes[0]; timeline.append(("input x", P + sum(live.values())))
    names = ["x"] + [f"o{l}" for l in range(1, len(sizes))]
    for l in range(1, len(sizes)):
        live[names[l]] = sizes[l]                                    # compute o^[l]
        timeline.append((f"compute {names[l]}", P + sum(live.values())))
        if not keep_all:
            del live[names[l - 1]]                                   # inference: drop o^[l-1]
            timeline.append((f"free {names[l-1]}", P + sum(live.values())))
    return timeline

# Step C – run both modes and read off the peaks.

inf = live_memory(sizes, keep_all=False)
trn = live_memory(sizes, keep_all=True)
print(f"parameters P = {P}")
print(f"inference peak = {max(v for _, v in inf)}   (= 51 + 4 + 5)")
print(f"training  peak = {max(v for _, v in trn)}   (= 51 + 4 + 5 + 3 + 2)")

#  Plot: numbers held in memory after each event (dashed line = the weights, always loaded).

fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
for ax, tl, title, col in [(axs[0], inf, "inference: free each layer's input after use", C["blue"]),
                           (axs[1], trn, "training: keep every activation for backprop", C["orange"])]:
    vals = [v for _, v in tl]
    xs_, ys_ = range(len(vals) + 1), vals + [vals[-1]]                 # one extra point so the last step shows
    ax.step(xs_, ys_, where="post", color=col, lw=2.5)
    ax.fill_between(xs_, P, ys_, step="post", color=col, alpha=0.15, label="activations")
    ax.axhline(P, color=C["muted"], ls="--", lw=1.2, label=f"weights: {P}")
    pk = int(np.argmax(vals)); ax.annotate(f"peak {vals[pk]}", (pk + 0.5, vals[pk]), textcoords="offset points", xytext=(-14, 6), fontweight="bold")
    ax.set_xticks(np.arange(len(tl)) + 0.5, [n for n, _ in tl], rotation=35, ha="right", fontsize=8.5)
    ax.set_title(title); ax.set_ylim(40, 72); ax.legend(loc="lower right")
axs[0].set_ylabel("numbers held in memory")
plt.tight_layout(); plt.show()

# ==============================================================================
# 3 · The full training bill: 16 bytes per parameter
# ==============================================================================
#
# During training, every parameter needs several numbers stored next to it:
#
#   item  |  why it exists  |  float32 bytes / param
#   weight  |  the parameter itself  |  4
#   gradient ∂E/∂θ  |  produced by backprop, consumed by the update  |  4
#   optimiser state  |  plain SGD: none; momentum: v; Adam: m and v  |  0 / 4 / 8
#   total (Adam)  |    |  16
#
#     memory≈ 16 P bytes + N·(n_0+2Σ[l≥1] n_l)·4_activations: x, and net^([l]),o^([l]) for every layer
#
# The parameter part is fixed; the activation part grows with the batch size N.
#
# Step A – the two counts.
# * `n_params` – weights + biases (notebook 01).
# * `activation_numbers` – what the forward cache holds: the input X (n_0 values per example)
#   plus net^([l]) and o^([l]) for every layer (2n_l per example), times the batch size.

def n_params(sizes):
    return sum(sizes[l] * sizes[l - 1] + sizes[l] for l in range(1, len(sizes)))


def activation_numbers(sizes, batch):
    """Numbers stored for backprop: the input X plus net^[l] and o^[l] of every layer."""
    return batch * (sizes[0] + 2 * sum(sizes[1:]))

# Step B – turn counts into bytes. `OPT_STATES` says how many extra numbers each optimiser
# keeps per parameter (SGD 0, momentum 1, Adam 2). Weights, gradients and activations use
# `bytes_per` (4 for float32); optimiser states are always counted in float32.

OPT_STATES = {"sgd": 0, "momentum": 1, "adam": 2}            # extra numbers stored per parameter


def training_memory(sizes, batch, optimizer="adam", bytes_per=4):
    P = n_params(sizes)
    return dict(weights=P * bytes_per,
                grads=P * bytes_per,
                optimizer=OPT_STATES[optimizer] * P * 4,      # optimiser states are kept in fp32
                activations=activation_numbers(sizes, batch) * bytes_per)

# Step C – estimate a 20-million-parameter MLP at batch 256 with Adam:

sizes = [784, 4096, 4096, 10]
mem = training_memory(sizes, batch=256)
print(f"network {sizes}: {n_params(sizes):,} parameters, batch 256, Adam, float32")
for k, v in mem.items():
    print(f"  {k:<12}{v / 1e6:10.2f} MB")
print(f"  {'TOTAL':<12}{sum(mem.values()) / 1e6:10.2f} MB")

# --- Check the estimate against real arrays ---
# Allocate float32 weights, gradients, Adam states and a real forward cache, then count their
# bytes.
#
# Step A – allocate the real float32 arrays (`.astype(np.float32)` makes each number 4 bytes):

B = 256
W = [rng.normal(0, 0.02, (sizes[l], sizes[l - 1])).astype(np.float32) for l in range(1, len(sizes))]
b = [np.zeros((n, 1), np.float32) for n in sizes[1:]]
X = rng.normal(size=(sizes[0], B)).astype(np.float32)

# Step B – run a forward pass and keep everything backprop would need (`store` = the cache: X,
# then net and o of each layer):

O, store = X, [X]                                           # forward pass storing net and o
for l in range(len(W)):
    NET = W[l] @ O + b[l]
    O = softmax(NET) if l == len(W) - 1 else np.maximum(0, NET)
    store += [NET, O]
print(len(store), "arrays in the cache")

# Step C – count the bytes with `.nbytes` and compare with the estimate. Gradients have
# exactly the shapes of the parameters, and Adam keeps two more arrays of those shapes.

real = dict(weights=sum(a.nbytes for a in W + b),
            grads=sum(a.nbytes for a in W + b),               # gradients have the same shapes
            optimizer=2 * sum(a.nbytes for a in W + b),       # Adam: m and v
            activations=sum(a.nbytes for a in store))
for k in mem:
    print(f"{k:<12} estimate = {mem[k] / 1e6:9.2f} MB | measured = {real[k] / 1e6:9.2f} MB | equal: {mem[k] == real[k]}")

# How memory splits as the batch grows: the fixed part (weights + gradients + Adam) against
# the activations, for batch sizes 1 to 8192.

batches = np.array([1, 8, 32, 128, 512, 2048, 8192])
act = np.array([training_memory(sizes, bs)["activations"] for bs in batches]) / 1e6
fixed = sum(v for k, v in training_memory(sizes, 1).items() if k != "activations") / 1e6
cross = batches[np.argmax(act > fixed)]                       # first batch size where activations win
print(f"fixed part = {fixed:.0f} MB;  activations: {act[0]:.3f} MB at batch 1 → {act[-1]:.0f} MB at batch 8192")
print(f"activations exceed the fixed part from batch ≈ {cross}")

#  Plot (log-log): activations grow in a straight line with batch size; the fixed part is
#  flat.

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.loglog(batches, act, "o-", color=C["orange"], label="activations (grow with batch size)")
ax.axhline(fixed, color=C["blue"], ls="--", label=f"weights + grads + Adam = {fixed:.0f} MB (fixed)")
ax.axvline(cross, color=C["muted"], lw=0.8); ax.text(cross * 1.1, act.min() * 2, f"activations dominate\nfrom batch ≈ {cross}", color=C["muted"], fontsize=9)
ax.set_xlabel("batch size N"); ax.set_ylabel("MB"); ax.set_title(f"memory vs batch size for {sizes}"); ax.legend(loc="upper left")
plt.tight_layout(); plt.show()

# ==============================================================================
# 4 · Scaling up: will an LLM fit?
# ==============================================================================
#
# Ignoring activations, Adam training costs 16 bytes per parameter (mixed precision also lands
# at about 16: 2+2 bytes for fp16 weights and gradients, plus 4+4+4 for fp32 master weights, m
# and v).

for name, P_ in [("125M (GPT-2 small)", 125e6), ("1B", 1e9), ("7B", 7e9), ("70B", 70e9)]:
    gb = 16 * P_ / 1e9
    print(f"{name:<19} → {gb:8,.1f} GB of training state   ({gb / 80:5.2f} × an 80 GB GPU)")

# A 7B model needs 112 GB just for its training state – more than any single GPU. We must
# split it. The notes list the two axes along which to split: data size and model size.
#
# ==============================================================================
# 5 · Data parallelism: copy the model, split the data
# ==============================================================================
#
# The notes: "Each GPU collects local updates. Collect all updates together."
#
# Every GPU holds a full copy of the model, takes a different slice of the mini-batch (N_k
# examples), runs forward + backward independently, and then the gradients are averaged (an
# all-reduce).
#
# Why is the result exactly right? Our loss is a mean over examples, so its gradient is a mean
# of per-example gradients, and a mean can be computed piece by piece:
#
#     g_full=frac1NΣ[n=1..N] g_n=Σ[k=1..K] N_k/N(frac1N_kΣ_ninGPU kg_n)_g_k
#
# Step A – a small ReLU / softmax network (He initialisation, notebook 04):

def init_params(sizes, rng):
    return [(rng.normal(0, np.sqrt(2 / sizes[l - 1]), (sizes[l], sizes[l - 1])), np.zeros((sizes[l], 1)))
            for l in range(1, len(sizes))]

# Step B – one GPU's work: forward + backward on its examples. `grads_for` is the full
# backprop of notebook 04 in compact form: forward storing (net, o), output delta (T-P)/N
# (mean loss!), then the backward loop. The ReLU slope is `(net > 0)`. It returns ∂E/∂W, ∂E/∂b
# for every layer.
#
# `flat` joins all gradients into one long vector, which makes it easy to compare and to
# "send" them in the all-reduce below.

def grads_for(params, X, T):
    """Mean-loss gradients ∂E/∂W, ∂E/∂b of a ReLU / softmax MLP on (X, T); columns = examples."""
    O, cache, L = X, [(None, X)], len(params)
    for l, (W, b) in enumerate(params, 1):                     # forward, storing net and o
        NET = W @ O + b
        O = softmax(NET) if l == L else np.maximum(0, NET)
        cache.append((NET, O))
    delta, out = (T - O) / X.shape[1], [None] * L                # δ^[L] = (T - P) / N
    for l in range(L, 0, -1):                                    # backward
        out[l - 1] = (-delta @ cache[l - 1][1].T, -delta.sum(1, keepdims=True))
        if l > 1:
            delta = (params[l - 1][0].T @ delta) * (cache[l - 1][0] > 0)
    return out


flat = lambda gr: np.concatenate([np.concatenate([dW.ravel(), db.ravel()]) for dW, db in gr])

# Step C – the experiment. 64 examples; the reference gradient is computed on all of them at
# once. Then `np.array_split` deals the example indices out to K=4 "GPUs", each computes its
# own gradient independently, and the all-reduce combines them with weights N_k/N.

sizes = [20, 32, 16, 4]
params = init_params(sizes, rng)
N, K = 64, 4
X = rng.normal(size=(20, N)); T = np.eye(4)[:, rng.integers(0, 4, N)]

# Run it and measure the largest difference between the two gradient vectors:

g_full = flat(grads_for(params, X, T))                          # one device, whole batch
shards = np.array_split(np.arange(N), K)                        # GPU k gets N/K examples
g_local = [flat(grads_for(params, X[:, s], T[:, s])) for s in shards]   # independent backward passes
g_avg = sum(len(s) / N * g for s, g in zip(shards, g_local))    # all-reduce: weighted average

print(f"{K} workers × {N // K} examples each")
print("max |single-GPU − data-parallel| =", np.abs(g_full - g_avg).max())

# Identical up to floating-point rounding.  Data parallelism changes where the work happens,
# not what is computed.
#
# --- Ring all-reduce: how the averaging is really done ---
# Sending every gradient to one "master" GPU makes that GPU a bottleneck (it receives (K-1)×
# the gradient). In a ring all-reduce, the gradient is cut into K chunks that travel around a
# circle of GPUs: first each chunk is summed as it passes (reduce-scatter), then the finished
# sums are passed around (all-gather). Each GPU sends only (2(K-1))/K times the gradient size
# – almost independent of K.
#
# The simulation, line by line:
# * every worker cuts its vector into K chunks: `ch[worker][chunk]`;
# * reduce-scatter (K-1 steps): in each step every worker sends one chunk to its right-hand
#   neighbour, which adds it to its own copy. After K-1 steps each worker owns one chunk that
#   holds the complete sum;
# * all-gather (K-1 steps): the finished chunks travel around the ring once more and overwrite
#   the neighbours' copies, so every worker ends with every summed chunk.
#
# All messages of a step are copied first (`msgs`) and delivered afterwards, as if all GPUs
# sent at the same time.

def ring_allreduce(vectors):
    """Simulate ring all-reduce over K workers. Returns what every worker holds at the end."""
    K = len(vectors)
    ch = [np.array_split(v.copy(), K) for v in vectors]          # ch[worker][chunk]
    for step in range(K - 1):                                    # phase 1: reduce-scatter
        sends = [(w, (w - step) % K) for w in range(K)]
        msgs = [ch[w][c].copy() for w, c in sends]
        for (w, c), msg in zip(sends, msgs):
            ch[(w + 1) % K][c] = ch[(w + 1) % K][c] + msg        # neighbour adds the incoming chunk
    for step in range(K - 1):                                    # phase 2: all-gather
        sends = [(w, (w + 1 - step) % K) for w in range(K)]
        msgs = [ch[w][c].copy() for w, c in sends]
        for (w, c), msg in zip(sends, msgs):
            ch[(w + 1) % K][c] = msg                             # neighbour overwrites with the full sum
    return [np.concatenate(c) for c in ch]

# Test it with 5 workers, each holding a random "gradient" of 1003 numbers (deliberately not
# divisible by 5):

vecs = [rng.normal(size=1003) for _ in range(5)]
out = ring_allreduce(vecs)
print("every worker ends with the full sum:", all(np.allclose(o, sum(vecs)) for o in out))

# Traffic per GPU (in units of one gradient), for 2 to 64 GPUs: a central server must receive
# K-1 gradients; in the ring each GPU sends 2(K-1)/K.

Ks = np.arange(2, 65)
traffic_server = Ks - 1
traffic_ring = 2 * (Ks - 1) / Ks
print(f"K = 64:  server receives {traffic_server[-1]} gradients,  ring sends {traffic_ring[-1]:.2f} per GPU")

#  Plot (log scale):

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(Ks, traffic_server, color=C["red"], label="central server: receives (K−1) × gradient")
ax.plot(Ks, traffic_ring, color=C["aqua"], label="ring: each GPU sends 2(K−1)/K × gradient")
ax.set_xlabel("number of GPUs K"); ax.set_ylabel("traffic (in gradient sizes)"); ax.set_yscale("log")
ax.set_title("ring all-reduce traffic stays flat as K grows"); ax.legend()
plt.tight_layout(); plt.show()

# --- ZeRO / FSDP: stop storing identical copies ---
# In plain data parallelism every GPU stores the same weights, gradients and Adam states –
# wasteful. ZeRO shards them: each GPU keeps only 1/K of a component and fetches the rest when
# needed.
#
#   strategy  |  weights  |  gradients  |  Adam states
#   plain data parallel (DDP)  |  full  |  full  |  full
#   ZeRO-1  |  full  |  full  |  1/K
#   ZeRO-2  |  full  |  1/K  |  1/K
#   ZeRO-3 / FSDP  |  1/K  |  1/K  |  1/K
#
# The table in code. Start with the full bytes per GPU (4 + 4 + 8 per parameter) and divide by
# K whatever the chosen ZeRO stage shards. Each stage shards everything the previous stage
# did, plus one more component.

def per_gpu_gb(P, K, strategy):
    w, g, o = 4 * P, 4 * P, 8 * P                                # bytes: weights, grads, Adam m+v
    if strategy in ("ZeRO-1", "ZeRO-2", "ZeRO-3"): o /= K
    if strategy in ("ZeRO-2", "ZeRO-3"):           g /= K
    if strategy == "ZeRO-3":                       w /= K
    return w / 1e9, g / 1e9, o / 1e9

# Apply it to a 7B model on 8 GPUs:

P7, K = 7e9, 8
strategies = ["DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
parts = np.array([per_gpu_gb(P7, K, s) for s in strategies])
for s, tot in zip(strategies, parts.sum(axis=1)):
    print(f"{s:<7} {tot:7.1f} GB per GPU   {'fits' if tot <= 80 else 'does NOT fit'} on an 80 GB GPU")

#  Plot: memory per GPU for each strategy, split into its three parts, with the 80 GB limit.

fig, ax = plt.subplots(figsize=(7.5, 3.9))
bottom = np.zeros(len(strategies))
for j, (lab, col) in enumerate(zip(["weights", "gradients", "Adam states"], [C["blue"], C["orange"], C["aqua"]])):
    ax.bar(strategies, parts[:, j], bottom=bottom, label=lab, color=col, width=0.55, edgecolor="white", linewidth=2)
    bottom += parts[:, j]
ax.axhline(80, color=C["ink"], ls="--", lw=1.2); ax.text(3.3, 84, "80 GB GPU", ha="right")
ax.set_ylabel("GB per GPU"); ax.set_title(f"7B model, Adam, K = {K} GPUs"); ax.legend(); ax.grid(axis="x", visible=False)
plt.tight_layout(); plt.show()

# ==============================================================================
# 6 · Pipeline parallelism: divide by layers
# ==============================================================================
#
# When the model itself does not fit, split it by layers (notes, page 6): GPU 1 holds the
# first layers, GPU 2 the next, and so on. Activations flow forward from GPU to GPU, blame (δ)
# flows back.
#
# Why does backprop still work? Notebook 04: each layer's update needs only "info about next
# layer and previous layer". GPU s needs only the activations coming in and the δ coming back
# from GPU s+1 – local information.
#
# The split in code. `layers` lists the (inputs, outputs) of each weight layer;
# `np.array_split` gives consecutive layers to each GPU. For each GPU we count its parameters
# and the size of the activation it must send to the next GPU for one micro-batch of 64
# examples.

sizes_big = [1024, 8192, 8192, 8192, 8192, 1000]
n_dev, micro = 2, 64
layers = list(zip(sizes_big[:-1], sizes_big[1:]))
split = np.array_split(np.arange(len(layers)), n_dev)
for d, idx in enumerate(split):
    Pd = sum(layers[i][0] * layers[i][1] + layers[i][1] for i in idx)
    print(f"GPU {d}: layers {[int(i) + 1 for i in idx]} → {Pd / 1e6:6.1f} M params = {16 * Pd / 1e9:5.2f} GB (fp32 + Adam)")
    if d < n_dev - 1:
        w_out = layers[idx[-1]][1]
        print(f"        sends o^[{idx[-1] + 1}] ({w_out}×{micro}) = {w_out * micro * 4 / 1e6:.1f} MB forward, "
              f"receives δ of the same size backward, per micro-batch")

# --- The pipeline "bubble" and micro-batches ---
# The notes: "F_i has to come before B_i" – a micro-batch's backward pass can only start after
# its forward pass has gone through all stages. With one big batch, each GPU sits idle while
# the others work. Splitting the batch into micro-batches F_1,F_2,... lets the stages overlap
# (the GPipe schedule):
#
# The schedule in code. Time is measured in steps; every forward or backward job takes one
# step.
# * Forward of micro-batch m on stage s starts at time m+s: a micro-batch can enter stage s
#   only after stage s-1 finished it, and a stage handles one micro-batch at a time.
# * Backward passes start once all forwards are done (time M+S-1), in reverse order of micro-
#   batches and of stages – B_i always after F_i.
# * Utilisation = busy job-slots ÷ (stages × total time).

def schedule(S, M):
    """GPipe-style schedule: forwards of M micro-batches through S stages, then backwards in reverse."""
    jobs = []                                                     # (stage, start, kind, micro-batch)
    for m in range(M):
        for s in range(S):
            jobs.append((s, m + s, "F", m + 1))
    t0 = M + S - 1
    for m in reversed(range(M)):
        for s in reversed(range(S)):
            jobs.append((s, t0 + (M - 1 - m) + (S - 1 - s), "B", m + 1))
    return jobs, t0 + M + S - 1

# Compute both schedules for 4 stages (GPUs): one big batch (M=1) and 4 micro-batches (M=4).

S = 4
schedules = {}
for M in (1, 4):
    jobs, T_total = schedule(S, M)
    schedules[M] = (jobs, T_total, 2 * S * M / (S * T_total))
    print(f"M = {M}: total time {T_total} steps, utilisation {schedules[M][2]:.0%}   (formula M/(M+S-1) = {M / (M + S - 1):.0%})")

#  Plot: a Gantt chart per schedule – one row per GPU, blue = forward, orange = backward, gaps
#  = idle (the "bubble").

fig, axs = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True)
for ax, M, title in [(axs[0], 1, "one big batch: each GPU idle most of the time"),
                     (axs[1], 4, "4 micro-batches: stages overlap, much smaller bubble")]:
    jobs, T_total, busy = schedules[M]
    for s, t, kind, m in jobs:
        ax.broken_barh([(t, 0.95)], (S - 1 - s - 0.4, 0.8), color=C["blue"] if kind == "F" else C["orange"])
        ax.text(t + 0.47, S - 1 - s, f"{kind}{m}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.set_yticks(range(S), [f"GPU {S - i}" for i in range(S)]); ax.set_title(f"{title}   (utilisation {busy:.0%})", fontsize=10.5)
    ax.grid(axis="y", visible=False)
axs[1].set_xlabel("time step"); axs[1].set_xlim(0, 16)
axs[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["blue"]), plt.Rectangle((0, 0), 1, 1, color=C["orange"])],
              labels=["forward F_i", "backward B_i"], loc="upper right", ncol=2)
plt.tight_layout(); plt.show()

# Utilisation rises from 25% to about 57% here, and approaches 100% as the number of micro-
# batches grows.
#
# ==============================================================================
# 7 · Tensor parallelism: split one weight matrix
# ==============================================================================
#
# The notes: "Weight matrix too big. Layer so wide [we can't] fit even 1 layer weight matrix →
# split weight matrices." Because net=Wo+b treats each row of W (each neuron) independently,
# we can give half the rows to each GPU and concatenate the results. Splitting by columns
# instead gives partial sums that must be added.
#
# Step A – one layer computed normally: 8 neurons, 6 inputs, a batch of 3 examples.

W_big = rng.normal(size=(8, 6)); b_big = rng.normal(size=(8, 1)); x_in = rng.normal(size=(6, 3))
full = W_big @ x_in + b_big

# Step B – the same layer split across two "GPUs", both ways.
# * Row split: GPU 0 holds rows 0–3 (neurons 0–3), GPU 1 rows 4–7. Each computes its own
#   neurons for the whole input; stacking the halves (`np.vstack`) gives the full result.
# * Column split: each GPU holds half of the columns (the weights for inputs 0–2 or 3–5) and
#   sees only those inputs. Each produces a partial sum for all neurons; adding the two (an
#   all-reduce) and the bias gives the full result.

# row split: GPU 0 computes neurons 0-3, GPU 1 neurons 4-7 → concatenate
row_parts = [W_big[:4] @ x_in + b_big[:4], W_big[4:] @ x_in + b_big[4:]]
# column split: each GPU sees half of the inputs → partial sums are added (an all-reduce)
col_parts = [W_big[:, :3] @ x_in[:3], W_big[:, 3:] @ x_in[3:]]

print("row split    (concatenate) equals full:", np.allclose(np.vstack(row_parts), full))
print("column split (sum)         equals full:", np.allclose(col_parts[0] + col_parts[1] + b_big, full))

# ==============================================================================
# 8 · Summary
# ==============================================================================
#
#   technique  |  splits  |  communicates  |  solves
#   data parallel  |  the batch  |  gradients (all-reduce)  |  throughput
#   ZeRO / FSDP  |  optimiser states, gradients, weights  |  shards on demand  |  memory per GPU
#   pipeline parallel (divide by layers)  |  layers  |  activations forward, δ backward  |  model too big for one GPU
#   tensor parallel (split weight matrices)  |  each matrix  |  partial results  |  a single layer too big
#
#   fact  |  value
#   notes' 4-5-3-2  |  51 params; inference peak 60 numbers; training peak 65
#   Adam training  |  16 bytes / parameter + activations (∝ batch size)
#   7B model  |  112 GB with plain DDP → 14 GB per GPU with ZeRO-3 on 8 GPUs
#   ring all-reduce  |  2(K-1)/K gradient sizes sent per GPU
#
# --- Try it yourself ---
# 1. Gradient checkpointing: store only every sqrt L-th layer's activations and recompute the
#    rest during backward. Add it to `activation_numbers`. How much memory is saved for a
#    16-layer network?
# 2. Extend `per_gpu_gb` for bf16 mixed precision (2-byte weights and grads, plus 4-byte
#    master weights).
# 3. Gradient accumulation: prove with `grads_for` that 4 micro-batches of 16 (gradients
#    averaged) give exactly the gradient of one batch of 64.
# 4. In `schedule`, how does utilisation change with M=8 and M=16? Derive the formula
#    M/(M+S-1).
#
# Module 02 complete!  Next module: Text to Numbers and Word Embeddings (word2vec, pages 6–13
# of the notes).
