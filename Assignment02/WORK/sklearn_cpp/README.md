# ENG3091 APSE Assignment 2

A header-only C++ library that implements `LinearRegression` and `LogisticRegression` from Python's scikit-learn. The library is then applied to three datasets: predicting concrete strength, detecting ECG anomalies, and classifying handwritten digits.

## Project structure

```
include/sklearn_cpp/
    linear_model/      LinearRegression and LogisticRegression
    utils/             CSV reader, scoring metrics, plotting helpers
examples/              main programs - one per dataset
data/                  CSV files (concrete.csv, ecg.csv, mnist_micro.csv)
plots/                 PNG outputs from the visualisation routines
```

## Implementation details

- **`linear_regression.hpp`** — implements linear regression with gradient descent. The model is `y = w . x + b` where `w` is a weight vector and `b` is a scalar bias. Training minimises mean squared error by computing the gradient over all samples and stepping the weights in the opposite direction. Provides `fit()`, `predict()` and `score()` matching the scikit-learn interface.

- **`logistic_regression.hpp`** — implements logistic regression for both binary (2-class) and multi-class (K>2) problems through a single class. The public `fit()` counts the unique labels in `y` and dispatches to either a sigmoid-based binary path or a softmax-based multi-class path. Both paths use gradient descent with L2 regularisation to prevent the weights growing without bound.

- **`csv_reader.hpp`** — opens a CSV file and parses each line into doubles. Used by all three main programs. Convention: every column except the last is a feature, the last column is the label. Handles files with or without a header row.

- **`metrics.hpp`** — provides `r2_score()` for regression (used by concrete) and `accuracy_score()` for classification (used by ECG and MNIST).

- **`visualize.hpp`** — wraps `matplotlibcpp.h` to produce evaluation plots: actual vs predicted, residuals, and confusion matrices. Plots are saved to `./plots/` and displayed on screen.

## Applications

| Program | Dataset | Model | Expected score |
|---|---|---|---|
| `main_concrete.cpp` | `concrete.csv` | LinearRegression | R² ~ 0.6 |
| `main_ecg.cpp` | `ecg.csv` | LogisticRegression (binary) | ~99% accuracy |
| `main_mnist.cpp` | `mnist_micro.csv` | LogisticRegression (multi-class softmax) | ~90% accuracy |

## Setup

Drop `concrete.csv`, `ecg.csv`, and `mnist_micro.csv` into `./data/`.

If you need `mnist_micro.csv`, generate it with:
```
python3 generate_mnist_csv.py
```

The plotting code depends on the Python development headers and matplotlib. Install them in WSL with:
```
sudo apt install python3-dev python3-matplotlib
```

## Build and run

```
./build.sh
./bin/concrete.out
./bin/ecg.out
./bin/mnist.out
```

If `./build.sh` gives "Permission denied", run `chmod +x build.sh` first (only needed once).

## Use of generative AI

Generative AI was used to scaffold the project structure, draft the plotting wrapper around `matplotlibcpp.h`, and assist with debugging. The core algorithms (gradient descent for linear regression, sigmoid and softmax classifiers for logistic regression) were written and tuned by the team.