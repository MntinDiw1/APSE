# Person C — Multi-class LogisticRegression (softmax)

You own the two private methods `fit_multiclass_()` and
`predict_multiclass_()` in
`include/sklearn_cpp/linear_model/logistic_regression.hpp`,
plus the test program `examples/main_mnist.cpp`.

---

## Your job in one paragraph

Extend the LogisticRegression class to handle problems with more than
two classes, using the **softmax** function. Train it on the
`mnist_micro.csv` dataset (3000 hand-drawn digit images) and get
reasonable accuracy. The class shell, the `softmax()` helper, the public
`fit()` dispatcher, the `score()` method, and the main file are already
written for you. You implement two private methods.

---

## How this fits with Person B

`LogisticRegression` handles both binary (K=2) and multi-class (K>2)
problems through a single class. The public `fit()` looks at how many
distinct labels appear in `y`:

- 2 distinct labels (ECG) → Person B's `fit_binary_()`
- 10 distinct labels (digits) → **your** `fit_multiclass_()`

Person B's work doesn't depend on yours and vice versa. Use a separate
git branch (e.g. `feature/multiclass`) to stay out of each other's way.

---

## What changes between binary and multi-class

In the binary case there's **one** weight vector and **one** bias.
In the multi-class case there's **K** weight vectors and **K** biases —
one per class.

| | Binary | Multi-class |
|---|---|---|
| Weights | `w` of length `n` | `W` of shape `K x n` |
| Bias | scalar `b` | vector `b` of length `K` |
| Activation | `sigmoid(z)` | `softmax(z)` |
| Loss | binary cross-entropy | categorical cross-entropy |
| Prediction | threshold at 0.5 | argmax over K probs |

---

## The maths in plain English

For each sample, you compute one raw score per class:

```
z_k = W_k . x + b_k                  for k = 0, 1, ..., K-1
```

Then squash all K scores into a probability distribution using
**softmax** — the helper is already written for you, numerically stable:

```
p_k = exp(z_k) / sum over all classes of exp(z_j)
```

For the loss, we need a **one-hot encoding** of the true label.
If the true class for sample `i` is `c`, then the one-hot vector is:

```
y_one_hot = [0, 0, ..., 1, ..., 0]
                       ^
                  at position c
```

You don't have to build the full one-hot vector explicitly. Inside the
loop you can compute on the fly:

```
y_ik = 1 if k == label else 0
```

The gradient simplifies beautifully because of how softmax + cross-entropy
combine. The error term per class is just `p_k - y_k`:

```
dL/dW_kj = (1/m) * sum_i (p_ik - y_ik) * x_ij  +  2 * lambda * W_kj
dL/db_k  = (1/m) * sum_i (p_ik - y_ik)
```

Update step is identical in form to linear and binary logistic regression:

```
W_kj  <-  W_kj  -  learning_rate * dL/dW_kj
b_k   <-  b_k   -  learning_rate * dL/db_k
```

You're doing the same gradient descent loop you'd do for binary,
just with an outer loop over the K classes.

---

## Step-by-step

1. Open `include/sklearn_cpp/linear_model/logistic_regression.hpp`.
2. Find the section titled `PERSON C -- MULTI-CLASS LOGISTIC REGRESSION`.
3. Inside `fit_multiclass_()`, follow the commented-out skeleton — it
   spells out every nested loop you need.
4. Inside `predict_multiclass_()`, compute the softmax probabilities
   and pick the class with the highest probability (argmax).
5. Use the static helper `softmax()` at the top of the class — it
   already does the "subtract max before exp" trick for numerical
   stability. Don't reinvent it.
6. Build with `./build.sh`, run `./bin/mnist.out`.

---

## What success looks like

```
$ ./bin/mnist.out
Loaded mnist_micro.csv
  Samples  : 3000
  Features : 784
  Classes  : 10 (digits 0-9)

Training LogisticRegression (multi-class)...
Accuracy (training set): ~85-90%

Sample predictions:
  sample 0     true=0  pred=0
  sample 100   true=0  pred=0
  sample 500   true=1  pred=1
  sample 1500  true=5  pred=5
  sample 2500  true=8  pred=8
```

If you hit ~85% on the first run you're golden. Push `n_iterations` up
to chase 90+.

---

## Watch out for

- **The four nested loops feel scary but are mechanical.**
  - outer: training iterations
  - sample loop: each of m samples
  - class loop: each of K classes
  - feature loop: each of n features

  Don't try to think about all four at once. Write the innermost ones
  first using fixed indices, then wrap loops around them.

- **MNIST is BIG.** 3000 samples × 784 features × 10 classes × 500
  iterations means about 12 billion floating-point operations. With `-O2`
  this runs in maybe 30 seconds. Without it, several minutes. The
  `build.sh` script already passes `-O2`.

- **Use the provided `softmax()` helper.** It subtracts the max before
  exponentiating; otherwise `exp(z)` overflows when `z` is large.

- **Zero-init is fine.** Don't use random weights — softmax + zero init
  starts everything at uniform 1/K probability, which is a perfectly
  good starting point.

- **Cast the label to int once at the top of the inner loop.** Reading
  `static_cast<int>(y[i])` inside the class loop repeatedly is wasteful.

---

## Hand-off

Once your test passes:
- Commit on your branch
- Merge with Person B's branch
- Verify both `./bin/ecg.out` and `./bin/mnist.out` still work after the
  merge — the public API hasn't changed, so they should
