# Person A — LinearRegression

You own `include/sklearn_cpp/linear_model/linear_regression.hpp`
and `examples/main_concrete.cpp`.

---

## Your job in one paragraph

Implement `fit()` and `predict()` on the `LinearRegression` class so that
the concrete strength predictor produces an R² score around 0.6 on the
training set. The class skeleton, the constructor, the score function,
and the main file are already written for you. You only need to fill in
two methods.

---

## The maths in plain English

The model is just a weighted sum of input features plus a bias:

```
y_hat = w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
```

To train, we measure how wrong the predictions are with the
**Mean Squared Error (MSE)** loss, then nudge `w` and `b` in the direction
that reduces it. We repeat this thousands of times.

The gradient of the MSE with respect to each weight `w_j` and the bias `b`:

```
dL/dw_j = (2/m) * sum over all samples of  (y_hat - y) * x_j
dL/db   = (2/m) * sum over all samples of  (y_hat - y)
```

Then the update step:

```
w_j  <-  w_j  -  learning_rate * dL/dw_j
b    <-  b    -  learning_rate * dL/db
```

That's it.

---

## Step-by-step

1. Open `include/sklearn_cpp/linear_model/linear_regression.hpp`.
2. Inside `fit()`, follow the commented-out skeleton — it spells out
   exactly what to write line by line.
3. Inside `predict()`, return `w . x + b` for every row of `X`.
4. Run `./build.sh` to compile, then `./bin/concrete.out` to test.

---

## What success looks like

```
$ ./bin/concrete.out
Loaded concrete.csv
  Samples  : 1030
  Features : 8

Training LinearRegression...
R^2 score (training set): 0.6  (give or take)
```

---

## Watch out for

- **Numerical stability.** The features in `concrete.csv` have very
  different scales (cement ≈ 500, age ≈ 28). Using a learning rate that's
  too big will make the gradient explode to NaN. The default `1e-7` should
  work; if not, try `1e-8` or normalise the features before training.

- **Loop order.** Compute *all* the gradient contributions first
  (accumulate `dw[j]` and `db` across every sample), then update the
  weights *after* the inner loop. Don't update weights mid-loop.

- **Don't shadow the member `bias_`** with a local variable called `bias`.
  Use the trailing underscore convention consistently.

---

## Hand-off

Once your test passes:
- Commit your changes
- Tell the team you're done
- Help Person B and C debug their integration if needed (they depend on
  the same scoring patterns you've validated)
