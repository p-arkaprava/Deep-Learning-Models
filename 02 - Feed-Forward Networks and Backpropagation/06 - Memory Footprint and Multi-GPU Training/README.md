# 06 · Memory Footprint and Multi-GPU Training

Estimate whether a model fits in GPU memory **before** running it, and see how training is split across several GPUs.

> 📘 For more explanation of each parallelism strategy and the debugging checklist, read **Sections 7–8** of **[FFNN_Backpropagation_Guide Simplified.pdf](../FFNN_Backpropagation_Guide%20Simplified.pdf)**.

---

## 1. Why training needs more memory than inference

Take the small `4-5-3-2` network from the handwritten notes: $(20+5)+(15+3)+(6+2)=\mathbf{51}$ parameters.

- **Inference (forward only):** at any moment you only need the current layer's input and output, then throw them away. The peak is about 51 params + 4 inputs + 5 hidden values.
- **Training:** backprop needs **every** layer's activations later, so all of them must be kept: 51 + 4 + 5 + 3 + 2 numbers, plus gradients and optimizer state.

## 2. The 16-bytes-per-parameter rule

For fp32 training with Adam, **each parameter** costs:

$$\underbrace{4}_{\text{weight}}+\underbrace{4}_{\text{gradient}}+\underbrace{8}_{\text{Adam's }m,\,v}=16\text{ bytes}\quad+\quad\text{activations}\;(\propto\text{batch size})$$

**Checked against real arrays** for a `784-4096-4096-10` MLP at batch 256 (20,037,642 params):

| Item | MB |
|---|---|
| weights | 80.15 |
| gradients | 80.15 |
| Adam states | 160.30 |
| activations | 17.60 |
| **total** | **338** |

At LLM scale (ignoring activations): 1B → 16 GB, 7B → 112 GB, 70B → 1,120 GB. A single 80 GB GPU can't hold a 7B model's training state.

![memory](../figures/memory.png)

## 3. Splitting the work across GPUs

**Data parallel: copy the model, split the data.** Each GPU has a full copy, processes $N_k$ examples, and the gradients are averaged with an **all-reduce**. Because the loss is a mean, the result is *exactly* the single-GPU gradient:

$$g_{\text{full}}=\frac1N\sum_{n=1}^{N}g_n=\sum_{k=1}^{K}\frac{N_k}{N}\,g_k$$

```python
shards  = np.array_split(np.arange(N), K)                         # each "GPU" gets N/K samples
g_local = [flat(grads_for(params, X[s], Y[s])) for s in shards]   # independent backward passes
g_avg   = sum(len(s) / N * g for s, g in zip(shards, g_local))    # all-reduce (weighted mean)
assert np.allclose(flat(grads_for(params, X, Y)), g_avg)          # max diff ~5.6e-17
```

A **ring all-reduce** passes chunks around a circle of GPUs. Each GPU sends only $\frac{2(K-1)}{K}$ times the gradient size, which is almost independent of $K$.

**ZeRO / FSDP: don't store identical copies.**

| Strategy | Weights | Grads | Adam states | 7B model, 8 GPUs |
|---|---|---|---|---|
| plain DDP | full | full | full | 112 GB |
| ZeRO-1 | full | full | 1/K | 63 GB |
| ZeRO-2 | full | 1/K | 1/K | 38.5 GB |
| ZeRO-3 / FSDP | 1/K | 1/K | 1/K | 14 GB |

**Pipeline parallel: split by layers** (the notes call this "divide by layers"). GPU 1 holds the first layers, GPU 2 the next, and activations flow between them. Backprop still works because each layer only needs local information. The forward pass $F_i$ must finish before the backward pass $B_i$ for each micro-batch.

**Tensor parallel: split one huge matrix** (the notes: "weight matrix too big, split weight matrices"). Used when even a single layer doesn't fit on one GPU.

| Technique | Splits | Communicates | Solves |
|---|---|---|---|
| Data parallel | the batch | gradients (all-reduce) | throughput |
| ZeRO / FSDP | optimizer, grads, weights | shards on demand | memory per GPU |
| Pipeline parallel | layers | activations | model too big for one GPU |
| Tensor parallel | each matmul | partial results | a single layer too big |

## Contents

| File | Description |
|---|---|
| `06_memory_footprint_and_multi_gpu.ipynb` | memory estimator checked against real arrays, batch-size scaling, LLM-scale table, ZeRO stages, simulated data parallelism, ring all-reduce, pipeline split |

## Run

```bash
jupyter notebook 06_memory_footprint_and_multi_gpu.ipynb
```

## Exercises
1. Add gradient checkpointing to the estimator (store only every $\sqrt L$-th layer's activations).
2. Add mixed-precision (bf16) accounting.
3. Prove that gradient accumulation over 4 micro-batches equals one big batch.

**Previous:** [05](../05%20-%20Training%20a%20Small%20Network/) · **Back to module overview:** [README](../README.md)
