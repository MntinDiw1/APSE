# Person B — Binary LogisticRegression

You own the two private methods `fit_binary_()` and `predict_binary_()`
in `include/sklearn_cpp/linear_model/logistic_regression.hpp`,
plus the test program `examples/main_ecg.cpp`.

---

## Your job in one paragraph

Implement binary logistic regression so that the ECG anomaly detector
gets around 99% accuracy on the training set. The class shell, the
`sigmoid()` helper, the public `fit()` dispatcher, the `score()` method,
and the main file are already written. You only need to fill in two
private methods.

---

## How this fits with Person C

`LogisticRegression` handles *both* binary (K=2) and multi-class (K>2)
problems. The public `fit()` looks at how many unique labels are in `y`
and dispatches to the right private method:

- If `y` contains only 0s and 1s → calls `fit_binary_()` → **your code**
- If `y` contains 0..9 (or any K>2) → calls `fit_multiclass_()` → Person C

Person C's work doesn't depend on yours and vice versa. Both of you edit
the same file, but completely different methods. **Use a git branch**
called something like `feature/binary` to avoid stepping on Person C.

---

## The maths in plain English

Linear regression outputs any real number — useless for "is this normal
or abnormal?" Logistic regression solves this by squashing the linear
output through the **sigmoid** function, which always lands in (0, 1).
We then read that as "probability of class 1".

```
z      = w . x + b                  (the same linear part as before)
y_hat  = sigmoid(z) = 1 / (1+e^-z)  (squashed to a probability)
```

Loss is **binary cross-entropy** with an L2 regularisation term:

```
L = -(1/m) * sum_i [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]
    + lambda * sum_j (w_j)^2
```

Gradients work out *almost identical* to linear regression:

```
dL/dw_j = (1/m) * sum_i (y_hat - y) * x_j  +  2 * lambda * w_j
dL/db   = (1/m) * sum_i (y_hat - y)
```

The L2 term `2 * lambda * w_j` only applies to weights, not the bias.

Update step is the standard gradient descent:

```
w_j  <-  w_j  -  learning_rate * dL/dw_j
b    <-  b    -  learning_rate * dL/db
```

---

## Step-by-step

1. Open `include/sklearn_cpp/linear_model/logistic_regression.hpp`.
2. Find the section titled `PERSON B -- BINARY LOGISTIC REGRESSION`.
3. Inside `fit_binary_()`, follow the commented-out skeleton — it
   walks you through the gradient descent loop line by line.
4. Inside `predict_binary_()`, compute `sigmoid(w . x + b)` and threshold
   at 0.5.
5. Use the static helper `sigmoid()` defined at the top of the class —
   it's already numerically stable, don't reinvent it.
6. Build with `./build.sh`, run `./bin/ecg.out`.

---

## What success looks like

```
$ ./bin/ecg.out
Loaded ecg.csv
  Samples  : 4997
  Features : 140
  Classes  : 2 (0 = abnormal, 1 = normal)

Training LogisticRegression (binary)...
Accuracy (training set): ~98-99%
```

---

## Watch out for

- **The 1/m factor goes on the data gradient, NOT on the regularisation.**
  The regularisation gradient is `2 * lambda * w_j`, with no `1/m`
  in front.

- **Use the provided `sigmoid()` helper.** It guards against overflow
  for very negative `z`. A naive `1 / (1 + exp(-z))` will produce NaN
  when `z` is large and negative.

- **Initialise weights to zero**, not random values. Logistic regression
  is convex, so zero-init is fine and reproducible.

- **The ECG labels are 0 and 1** — read them as `double` from the CSV
  but treat them as binary in the maths.

---

## Hand-off

Once your test passes:
- Commit on your branch
- Open a pull request / merge with Person C's branch
- Verify Person C's `fit_multiclass_()` still works after the merge
