# ⚙️ Deep Learning Internals: From Perceptron to GPT

Implementing the core ideas of deep learning **from scratch, in code**, following the path from a single perceptron to feed-forward networks, word embeddings, RNNs, LSTM/GRU, attention, the Transformer, BERT, and GPT.

Every concept in my course notes gets a matching implementation here: first by hand in **NumPy** so the math is visible, then compared against **PyTorch** to confirm it is correct.

> 📌 This repo is a **living project**. I add code, experiments, and notes as I work through each topic.

---

## 🔗 Repository

**GitHub:** https://github.com/p-arkaprava/Deep-Learning-and-Machine-Learning-Models.git

---

## 🎯 Objectives

* Implement every model in the course from first principles, not just call a library.
* Derive and code backpropagation by hand, then verify it with gradient checking and PyTorch autograd.
* Understand why each architecture exists: what broke in the previous one and how the next one fixes it.
* Count parameters and memory by hand for every model, and confirm the numbers in code.
* Build up to a working Transformer, a BERT-style encoder, and a GPT-style decoder.
* Finish with the course project: an LLM + verifier loop for NP-complete problems.
* Keep a clear record of progress through regular commits.

---

## 🧭 The big picture

Each model fixes a weakness of the one before it:

```text
Perceptron → FFNN + Backprop → Word2Vec → RNN → LSTM / GRU → Seq2Seq + Attention → Transformer → BERT / GPT
```

| From → To | What broke | What fixed it |
|---|---|---|
| Perceptron → FFNN | Cannot learn non-linearly separable data; no way to train hidden units | Differentiable activations + backpropagation |
| FFNN → RNN | Fixed input/output size; no memory across positions | Reused cell with a hidden state carried over time |
| RNN → LSTM / GRU | Vanishing and exploding gradients | Gated cell state with a near-identity gradient path |
| LSTM → Seq2Seq + Attention | Whole input squeezed into one vector | Decoder looks at all encoder states, weighted by relevance |
| Attention → Transformer | Still sequential and slow; encoder side not improved | Self-attention in parallel + positional embeddings |
| Transformer → BERT / GPT | One architecture does not fit every task | Keep only the encoder (understanding) or only the decoder (generation) |

---

## 🗺️ Topics

| # | Topic | Focus |
|---|-------|-------|
| 01 | [Perceptron and Gradient Descent](01%20-%20Perceptron%20and%20Gradient%20Descent/) | From a single neuron to gradient-based learning |
| 02 | [Feed-Forward Networks and Backpropagation](02%20-%20Feed-Forward%20Networks%20and%20Backpropagation/) | Multi-layer networks and the backprop derivation |
| 03 | [Text to Numbers and Word Embeddings](03%20-%20Text%20to%20Numbers%20and%20Word%20Embeddings/) | One-hot, TF-IDF, and Word2Vec in depth |
| 04 | Computation Graphs and Autodiff | Seeing networks as DAGs the compiler can optimise |
| 05 | Recurrent Neural Networks | Sequence models with shared weights and a hidden state |
| 06 | LSTM and GRU | Gated cells that fight vanishing gradients |
| 07 | Seq2Seq and Attention | Encoder-decoder models and the attention fix for the bottleneck |
| 08 | Transformer | Self-attention, multi-head attention, and the full encoder-decoder |
| 09 | BERT | Encoder-only Transformer for understanding tasks |
| 10 | GPT | Decoder-only Transformer for generation |
| 11 | Project - LLM with Verifier Loop | Course project: solve an NP-complete problem by iterating between an LLM and verifier code |

---

## ✅ Progress tracker

**01 - Perceptron and Gradient Descent**
- [x] 01 - Thresholded Perceptron
- [x] 02 - Unthresholded Perceptron and Delta Rule
- [x] 03 - Sigmoid Unit and Non-linearity

**02 - Feed-Forward Networks and Backpropagation**
- [x] 01 - Architecture and Parameter Counting
- [x] 02 - Matrix-Form Forward Pass
- [x] 03 - Backprop for the Output Layer
- [x] 04 - Backprop for Hidden Layers
- [x] 05 - Training a Small Network
- [x] 06 - Memory Footprint and Multi-GPU Training

**03 - Text to Numbers and Word Embeddings**
- [x] 01 - One-Hot and TF-IDF
- [x] 02 - Distributional Semantics and Co-occurrence
- [x] 03 - PageRank Analogy
- [x] 04 - Word2Vec Model
- [x] 05 - Softmax and Cross-Entropy Gradients
- [x] 06 - Naive Training
- [x] 07 - Negative Sampling
- [x] 08 - Hierarchical Softmax
- [x] 09 - Comparing Training Variants

**04 - Computation Graphs and Autodiff**
- [ ] 01 - Declarative vs Procedural Form
- [ ] 02 - Computation Graph for FFNN
- [ ] 03 - Mini Autodiff Engine
- [ ] 04 - Unrolling RNNs into Acyclic Graphs

**05 - Recurrent Neural Networks**
- [ ] 01 - RNN Cell and Unrolling
- [ ] 02 - Input-Output Architectures
- [ ] 03 - Backpropagation Through Time
- [ ] 04 - Vanishing and Exploding Gradients
- [ ] 05 - Stacked and Bidirectional RNNs
- [ ] 06 - Character-Level Text Generation

**06 - LSTM and GRU**
- [ ] 01 - LSTM Cell
- [ ] 02 - GRU Cell
- [ ] 03 - Comparing RNN, GRU and LSTM

**07 - Seq2Seq and Attention**
- [ ] 01 - Encoder-Decoder Seq2Seq
- [ ] 02 - Bahdanau Attention
- [ ] 03 - Luong Attention
- [ ] 04 - Decoding Strategies
- [ ] 05 - Attention Visualisation

**08 - Transformer**
- [ ] 01 - Self-Attention
- [ ] 02 - Multi-Head Attention
- [ ] 03 - Positional Embeddings
- [ ] 04 - Residual Connections and Layer Norm
- [ ] 05 - Feed-Forward Sub-layer and Encoder Block
- [ ] 06 - Decoder Block
- [ ] 07 - Teacher Forcing and Causal Mask
- [ ] 08 - Full Encoder-Decoder Transformer
- [ ] 09 - KV Caching and Autoregressive Inference
- [ ] 10 - Parameter Counting

**09 - BERT**
- [ ] 01 - Encoder-Only Model and CLS Token
- [ ] 02 - Masked Language Modelling
- [ ] 03 - Next Sentence Prediction
- [ ] 04 - Fine-Tuning a Classifier Head

**10 - GPT**
- [ ] 01 - Decoder-Only Architecture
- [ ] 02 - Next-Token Pretraining
- [ ] 03 - Text Generation and Sampling
- [ ] 04 - Choosing an Architecture

**11 - Project - LLM with Verifier Loop**
- [ ] 01 - Instance Generator
- [ ] 02 - Verifier
- [ ] 03 - LLM Solver Interface
- [ ] 04 - Feedback Loop
- [ ] 05 - Evaluation and Report

### Status overview

| Topic | Status |
|-------|--------|
| 01 - Perceptron and Gradient Descent | 🟢 Completed |
| 02 - Feed-Forward Networks and Backpropagation | 🟢 Completed |
| 03 - Text to Numbers and Word Embeddings | 🟢 Completed |
| 04 - Computation Graphs and Autodiff | ⚪ Upcoming |
| 05 - Recurrent Neural Networks | ⚪ Upcoming |
| 06 - LSTM and GRU | ⚪ Upcoming |
| 07 - Seq2Seq and Attention | ⚪ Upcoming |
| 08 - Transformer | ⚪ Upcoming |
| 09 - BERT | ⚪ Upcoming |
| 10 - GPT | ⚪ Upcoming |
| 11 - Project - LLM with Verifier Loop | ⚪ Upcoming |

---

## 🗂️ Repository structure

```text
dl-internals/
│
├── README.md
├── .gitignore
│
├── 01 - Perceptron and Gradient Descent/
│   ├── README.md
│   ├── 01 - Thresholded Perceptron/
│   │   └── README.md  (+ code, notebooks, plots)
│   ├── 02 - Unthresholded Perceptron and Delta Rule/
│   │   └── README.md  (+ code, notebooks, plots)
│   └── ...
│
├── 02 - Feed-Forward Networks and Backpropagation/
│   ├── README.md
│   ├── 01 - Architecture and Parameter Counting/
│   │   └── README.md  (+ code, notebooks, plots)
│   ├── 02 - Matrix-Form Forward Pass/
│   │   └── README.md  (+ code, notebooks, plots)
│   └── ...
│
├── 03 - Text to Numbers and Word Embeddings/
│   ├── README.md
│   ├── 01 - One-Hot and TF-IDF/
│   │   └── README.md  (+ code, notebooks, plots)
│   ├── 02 - Distributional Semantics and Co-occurrence/
│   │   └── README.md  (+ code, notebooks, plots)
│   └── ...
│
├── ...
└── 11 - Project - LLM with Verifier Loop/
```

Each topic folder has a `README.md` with an overview of the module, its notation and its key results. Each completed sub-topic folder contains:

* `README.md`: the core idea, the variables, the maths step by step, worked examples, figures and the core code;
* a Jupyter notebook (`.ipynb`) with outputs and plots already rendered;
* an equivalent plain-Python script (`.py`) with the explanations as comments, runnable with `python <file>.py`.

Figures used by the READMEs live in each module's `figures/` folder.

---

## 🛠️ Tech stack

* Python
* NumPy (from-scratch implementations)
* PyTorch (verification and larger models)
* Matplotlib (plots and attention heatmaps)
* Jupyter Notebook / Google Colab

---

## 🧪 How I implement each topic

1. Write the math in the sub-topic README (equations and shapes).
2. Implement it from scratch in NumPy.
3. Verify it: gradient checking, shape checks, parameter-count checks, or comparison with the PyTorch equivalent.
4. Run a small experiment and save a plot or result table.
5. Tick the box in the tracker and add a line to the progress log.

---

## 📆 Daily progress

I aim for at least one meaningful commit per working session: a new implementation, a bug fix, an experiment, or notes.

```text
Complete 02/03: backprop for the output layer with gradient check
Add 05/04: vanishing gradient experiment and plots
Update README progress tracker
```

---

## 📚 References

* Tom Mitchell, *Machine Learning*: perceptrons, gradient descent, backpropagation
* Xin Rong, *word2vec Parameter Learning Explained*, and Mikolov et al., word2vec papers
* Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*
* Luong et al., *Effective Approaches to Attention-based Neural Machine Translation*
* Vaswani et al., *Attention Is All You Need*
* Devlin et al., *BERT*; Radford et al., *GPT*

---

## 🗒️ Progress log

| Date | Update |
|------|--------|
| 2026-10-08 | Created repo structure from the course notes |
| 2026-10-09 | Added Perceptron Code and concepts and Explaination |
| 2026-10-10 | Added Chapter 02 : FFNN, Backpropagation |
| 2026-10-17 | Added Chapter 03 : One-hot, TF-IDF, distributional semantics, PageRank, word2vec (full softmax, negative sampling, hierarchical softmax); `.py` script versions of every notebook in chapters 01–03 |

---

## 📄 License

This is a personal learning project. Course material belongs to its original instructors; this repository contains only my own code and notes. Licensing details will be added later.
