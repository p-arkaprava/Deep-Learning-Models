# 04 · The Word2Vec Model

> **Module:** [03 - Text to Numbers and Word Embeddings](../) · **Notebook:** [`04_word2vec_model.ipynb`](04_word2vec_model.ipynb) · **Script:** [`04_word2vec_model.py`](04_word2vec_model.py)
> **Course notes:** page 10 (*"pick up one word → want to predict next word"*, the $x\to V\to h\to U\to z\to$ softmax picture, *"No bias, no act. func except softmax. No. of parameters = 2×d×n"*), page 11 (*"Where are my vectors? Each column of V"*) · **Further reading:** Xin Rong, *word2vec Parameter Learning Explained*

---

## 📌 The core idea in one sentence

Word2vec builds a tiny network whose only job is to **predict a neighbouring word**; we don't care about the predictions, we keep the first weight matrix $V$, whose columns become dense word vectors in which similar words are close together.

![word2vec architecture](../figures/c3_word2vec_architecture.png)

---

## 1. The variables you need first

| Symbol | Shape | Meaning |
|---|---|---|
| $n$ | number | vocabulary size (75 here; ~$10^5$ in practice) |
| $d$ | number | embedding size: numbers per word (10 here; notes: typically 300) |
| $x$ | $n\times1$ | one-hot vector of the **input** word |
| $c$ | index | the input (center) word |
| $o$ | index | the true output (context) word |
| $V$ | $d\times n$ | **input** matrix; column $v_k$ is the **word vector** of $w_k$ |
| $h=Vx$ | $d\times1$ | hidden vector |
| $U$ | $n\times d$ | **output** matrix; row $u_k$ is the "output vector" of $w_k$ |
| $z=Uh$ | $n\times1$ | one score per word, $z_k=u_k\cdot h$ |
| $y=\text{softmax}(z)$ | $n\times1$ | predicted probability of each word being the neighbour |
| $E=-\log y_o$ | number | error for one training pair |
| $m$ | number | window size |

---

## 2. The training task: predict a neighbour

The notes' sentence: *"I like going **to** my old house"*. Pick one word and predict the words around it (window $m$). There are two mirror-image versions:

| | input | output | |
|---|---|---|---|
| **skip-gram** | the center word | **each** neighbour, one at a time | used in notebooks 05–09 |
| **CBOW** | **all** neighbours together | the center word | $h$ = average of the context columns |

```text
  input = i       →  predict like
  input = i       →  predict going
  input = like    →  predict i
  input = like    →  predict going
  input = like    →  predict to
  input = going   →  predict i
  input = going   →  predict like
  input = going   →  predict to
  input = going   →  predict my
  input = to      →  predict like
  input = to      →  predict going
  input = to      →  predict my
  ... 22 pairs from one 7-word sentence
```

```text
31,734 skip-gram training pairs
```

```text
  input = ['like', 'going']  →  predict i
  input = ['i', 'going', 'to']  →  predict like
  input = ['i', 'like', 'to', 'my']  →  predict going
  input = ['like', 'going', 'my', 'old']  →  predict to
```

Each skip-gram pair $(c,o)$ is one training example. A real corpus has billions of them.

---

## 3. The architecture

$$x\ \xrightarrow{\ V\ }\ h=Vx\ \xrightarrow{\ U\ }\ z=Uh\ \xrightarrow{\text{softmax}}\ y,\qquad y_k=\frac{e^{z_k}}{\sum_{m=1}^{n}e^{z_m}}$$

| Layer | Operation | Shape |
|---|---|---|
| input | one-hot $x$ | $n$ |
| hidden | $h=Vx$, **no bias, no activation** | $d$ |
| output | $z=Uh$, then softmax | $n$ |

### $h=Vx$ is just a column lookup

Because $x$ has a single 1 at position $c$:

$$h=Vx=\sum_{k=1}^{n}x_k\ v_k=v_c$$

```text
V @ x equals column c of V: True
multiplications: V @ x = 750   lookup = 0
```

So the "first layer" is a **table lookup**: no multiplications at all. In PyTorch this is `nn.Embedding`.

### The error

For one pair $(c,o)$ we want $y_o$ (the probability of the true neighbour) to be large:

$$E=-\log y_o$$

---

## 4. Counting parameters

$V$ has $d\times n$ numbers and $U$ has $n\times d$, with no biases (notes: *"No. of parameters = 2×d×n"*):

$$\text{params}=dn+nd=2dn$$

```text
toy model: d = 10, n = 75  →  2dn = 1,500   counted: 1,500
d = 100, n =    10,000  →      2,000,000 parameters  ( 0.01 GB in float32)
d = 300, n =   100,000  →     60,000,000 parameters  ( 0.24 GB in float32)
d = 300, n = 3,000,000  →  1,800,000,000 parameters  ( 7.20 GB in float32)
```

---

## 5. What does an untrained model predict?

With small random weights every score $z_k\approx0$, so every word gets $y_k\approx1/n$, and the error is $E\approx-\log(1/n)=\log n$:

```text
untrained predictions for 'king': [('interest', 0.014), ('king', 0.014), ('milk', 0.014), ('tea', 0.014), ('shore', 0.014)]
mean error on 2000 pairs = 4.321    log n = 4.317
```

This is the starting point that training (notebooks 05–06) must improve.

---

## 6. "Where are my vectors?"

The notes' answer: **each column of $V$**. Before training they are random, so *king* is no closer to *queen* than to *mango*:

```text
word vectors are the columns of V: (10, 6)
cosine(king, queen) before training = -0.066   cosine(king, mango) = -0.156
```

After training (notebook 06) the same columns put *king* next to *queen* and *prince*.

**Intuition.** Words that appear in similar contexts must produce similar predictions $y$. The only input-specific part of the network is the column $v_c$, so similar words are pushed towards similar columns.

---

## 7. Code, step by step

| Step | Code | Maths |
|---|---|---|
| 1. Pairs | `pairs.append((c, ids[ctx_pos]))` within the window | $(c,o)$ |
| 2. Parameters | `V = rng.normal(0, 0.1, (d, n))`, `U = rng.normal(0, 0.1, (n, d))` | small random $V$, $U$ |
| 3. Hidden | `self.V[:, c]` | $h=Vx=v_c$ |
| 4. Scores | `self.U @ h` | $z=Uh$ |
| 5. Probabilities | `e = np.exp(z - z.max()); e / e.sum()` | softmax (shifted for stability) |
| 6. Error | `-np.log(y[o])` | $E=-\log y_o$ |
| 7. Count | `V.size + U.size` | $2dn$ |

The class is built **one method per cell** in the notebook (`Word2Vec.forward = forward`), so each piece can be explained on its own.

<details>
<summary><b>Click to expand: the core code</b></summary>

```python
import numpy as np

def skipgram_pairs(sentences, index, window=2):
    """All (center, context) index pairs within `window` positions."""
    pairs = []
    for s in sentences:
        ids = [index[w] for w in s]
        for pos, c in enumerate(ids):
            for ctx_pos in range(max(0, pos - window), min(len(ids), pos + window + 1)):
                if ctx_pos != pos:
                    pairs.append((c, ids[ctx_pos]))
    return np.array(pairs)

def cbow_examples(sentences, index, window=2):
    """(list of context indices, center index) for every position."""
    examples = []
    for s in sentences:
        ids = [index[w] for w in s]
        for pos, c in enumerate(ids):
            ctx = [ids[q] for q in range(max(0, pos - window), min(len(ids), pos + window + 1)) if q != pos]
            examples.append((ctx, c))
    return examples

for ctx, c in cbow_examples(notes_sentence, index, window=2)[:4]:
    print(f"  input = {[vocab[k] for k in ctx]}  →  predict {vocab[c]}")

class Word2Vec:
    """x (one-hot, n) --V (d×n)--> h (d) --U (n×d)--> z (n) --softmax--> y (n). No biases."""

    def __init__(self, n, d, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.d = n, d
        self.V = rng.normal(0, 0.1, (d, n))      # input matrix: column k = word vector v_k
        self.U = rng.normal(0, 0.1, (n, d))      # output matrix: row k = output vector u_k

    def n_params(self):
        return self.V.size + self.U.size        # should be 2·d·n

def hidden(self, c):
    """Skip-gram: h = V x = column c of V."""
    return self.V[:, c]


def hidden_cbow(self, context):
    """CBOW: h = average of the context words' columns."""
    return self.V[:, context].mean(axis=1)

Word2Vec.hidden, Word2Vec.hidden_cbow = hidden, hidden_cbow

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def forward(self, c):
    """Skip-gram forward pass for input word c. Returns h, z, y."""
    h = self.hidden(c)                 # (d,)
    z = self.U @ h                     # (n,)  z_k = u_k · h
    y = softmax(z)                     # (n,)  probabilities
    return h, z, y

Word2Vec.forward = forward

def predict(self, word, k=5):
    _, _, y = self.forward(index[word])
    top = np.argsort(-y)[:k]
    return [(vocab[j], round(float(y[j]), 3)) for j in top]


def error(self, c, o):
    _, _, y = self.forward(c)
    return -np.log(y[o])

Word2Vec.predict, Word2Vec.error = predict, error
```
</details>

The complete runnable version (corpus, plots) is [`04_word2vec_model.py`](04_word2vec_model.py).

---

## 8. What is in the notebook

| Section | Content |
|---|---|
| 2 | skip-gram pairs, CBOW examples, target-frequency plot |
| 3 | architecture diagram |
| 4 | the `Word2Vec` class, method by method; one-hot product vs lookup |
| 5 | parameter counts from the toy model up to $n=3\times10^6$ |
| 6 | untrained predictions, $E\approx\log n$ |
| 7 | the columns of $V$ as (still random) word vectors |

```bash
jupyter notebook 04_word2vec_model.ipynb     # or:  python 04_word2vec_model.py
```

## ✍️ Exercises
1. Write the CBOW forward pass with `hidden_cbow`. How many columns of $V$ does one example touch?
2. How much memory do $V$ and $U$ need in float32 for $d=300$, $n=3\times10^6$?
3. Initialise $V$ and $U$ with standard deviation 1 instead of 0.1. What is the untrained error now, and why?

**Previous:** [03 - PageRank Analogy](../03%20-%20PageRank%20Analogy/) · **Next:** [05 - Softmax and Cross-Entropy Gradients](../05%20-%20Softmax%20and%20Cross-Entropy%20Gradients/)
