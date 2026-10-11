# 01 · Perceptron and Gradient Descent

**From a single artificial neuron to gradient-based learning.**

This module builds one neuron, shows where it breaks, and fixes it twice. Each fix is a small change to the maths and a small change to the code. By the end you will know *why* modern networks use smooth activation functions such as the sigmoid, and you will be ready for backpropagation in the next module.

> 📘 **Read the full guide for more detail:**
> **[Deep Learning Internals - Perceptron to Sigmoid Simplified.pdf](Deep%20Learning%20Internals%20-%20Perceptron%20to%20Sigmoid%20Simplified.pdf)** (13 pages). It covers intuition, step-by-step maths, worked examples with exact numbers, line-by-line code walk-throughs, figures and self-check questions.
> For what comes next, see **[FFNN_Backpropagation_Guide Simplified.pdf](../02%20-%20Feed-Forward%20Networks%20and%20Backpropagation/FFNN_Backpropagation_Guide%20Simplified.pdf)** in module 02.

![roadmap](figures/p1_roadmap.png)

---

## Why copy the brain?

| Fact | Number | What it tells us |
|---|---|---|
| Neurons | $\approx 10^{11}$ | a huge number of **very simple** units |
| Connections per neuron | $\approx 10^{4}$ | each unit listens to many others |
| Switching time of a neuron | $\approx 10^{-3}$ s | each unit is **slow**... |
| Time to recognise a face | $\approx 10^{-1}$ s | ...so only ~100 steps can happen one after another |

With only about 100 slow steps, the brain can't be running a long sequential program. It must do **lots of simple work in parallel**, with knowledge spread across connection strengths (*distributed representation*) that were **learned** rather than coded. Artificial neural networks copy this idea: many simple units plus learnable weights.

---

## The three stages

### 1 · Thresholded perceptron → [folder](01%20-%20Thresholded%20Perceptron/)

![neuron](figures/p1_neuron.png)

$$net = \sum_{i=0}^{n} w_i x_i ,\qquad y=\begin{cases}+1 & net>0\\ -1 & \text{otherwise}\end{cases},\qquad \Delta w_i=\eta\,(t-o)\,x_i$$

The perceptron draws **one straight line** (a hyperplane) and says YES on one side. It learns AND and OR but **can never learn XOR**. Its step function also has slope 0 everywhere, so it can't be trained inside a multi-layer network.

### 2 · Unthresholded perceptron and the Delta rule → [folder](02%20-%20Unthresholded%20Perceptron%20and%20Delta%20Rule/)

Remove the step: $o=\vec w\cdot\vec x$. Now we can measure *how wrong* we are and roll downhill on a smooth error surface:

$$E(\vec w)=\tfrac12\sum_{d\in D}(t_d-o_d)^2,\qquad \vec w\leftarrow \vec w-\eta\,\nabla E,\qquad \Delta w_i=\eta\sum_d (t_d-o_d)\,x_{i,d}$$

![bowl](figures/bowl.png)

### 3 · Sigmoid unit and non-linearity → [folder](03%20-%20Sigmoid%20Unit%20and%20Non-linearity/)

Stacked linear layers **collapse into one** ($W_2W_1\vec x=W_3\vec x$). The sigmoid is a smooth, differentiable version of the step that prevents this collapse:

$$\sigma(x)=\frac{1}{1+e^{-x}},\qquad \sigma'(x)=y\,(1-y)$$

![step vs sigmoid](figures/step_vs_sigmoid.png)

---

## Side-by-side summary

| | Thresholded perceptron | Linear unit (Delta rule) | Sigmoid unit |
|---|---|---|---|
| Output | $\text{step}(\vec w\cdot\vec x)\in\{-1,+1\}$ | $\vec w\cdot\vec x$ | $\sigma(\vec w\cdot\vec x)\in(0,1)$ |
| Error measured as | right / wrong | $\tfrac12\sum(t-o)^2$ | $\tfrac12\sum(t-o)^2$ |
| Update $\Delta w_i$ | $\eta(t-o)x_i$ | $\eta\sum_d(t_d-o_d)x_{i,d}$ | $\eta\sum_d(t_d-o_d)\,o_d(1-o_d)\,x_{i,d}$ |
| Differentiable? | ❌ | ✅ | ✅ |
| Useful when stacked? | can't be trained | ❌ collapses | ✅ non-linear |
| XOR with one unit? | ❌ | ❌ | ❌ (needs a hidden layer) |

> ⚠️ **A common misconception:** a *single* sigmoid neuron still draws a straight boundary and **can't** solve XOR. You need at least one hidden layer, which is exactly what module 02 trains with backpropagation.

![xor needs hidden layer](figures/xor_hidden.png)

---

## Folder structure

```
01 - Perceptron and Gradient Descent/
├── README.md                                                    ← you are here
├── Deep Learning Internals - Perceptron to Sigmoid Simplified.pdf   ← full guide
├── figures/                                                     ← all images used in the READMEs
├── 01 - Thresholded Perceptron/                   README + .py + .ipynb
├── 02 - Unthresholded Perceptron and Delta Rule/  README + .py + .ipynb
└── 03 - Sigmoid Unit and Non-linearity/           README + .py + .ipynb
```

## Quick start

```bash
pip install numpy matplotlib jupyter
python "01 - Thresholded Perceptron/01_thresholded_perceptron.py"
```

Every script uses only NumPy and Matplotlib and runs in seconds.

## Topics
- [x] 01 - Thresholded Perceptron
- [x] 02 - Unthresholded Perceptron and Delta Rule
- [x] 03 - Sigmoid Unit and Non-linearity

**Next module:** [02 - Feed-Forward Networks and Backpropagation](../02%20-%20Feed-Forward%20Networks%20and%20Backpropagation/)
