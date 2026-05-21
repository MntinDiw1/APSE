# ENG3091 APSE Assignment 2: sklearn implementation in C++

A header-only C++ library that re-implements `LinearRegression` and `LogisticRegression` from Python's scikit-learn. The library is then applied to three datasets: predicting concrete strength, detecting ECG anomalies, and classifying handwritten digits.

## Project structure

```
include/sklearn_cpp/
    linear_model/      LinearRegression and LogisticRegression
    utils/             CSV reader, scoring metrics, plotting helpers
examples/              main programs - one per dataset
data/                  CSV files (concrete.csv, ecg.csv, mnist_*.csv)
plots/                 PNG outputs from the visualisation routines
bin/                   compiled executables (created by build.sh)
```

## Implementation details

- **`linear_regression.hpp`** — implements linear regression with gradient descent. The model is `y = w . x + b` where `w` is a weight vector and `b` is a scalar bias. Training minimises mean squared error by computing the gradient over all samples and stepping the weights in the opposite direction. Provides `fit()`, `predict()` and `score()` matching the scikit-learn interface.

- **`logistic_regression.hpp`** — implements logistic regression for both binary (2-class) and multi-class (K>2) problems through a single class. The public `fit()` counts the unique labels in `y` and dispatches to either a sigmoid-based binary path or a softmax-based multi-class path. Both paths use gradient descent with L2 regularisation to prevent the weights growing without bound.

- **`csv_reader.hpp`** — opens a CSV file and parses each line into doubles by splitting on commas. Used by all three main programs. Convention: every column except the last is a feature, the last column is the label. Handles files with or without a header row.

- **`metrics.hpp`** — provides `r2_score()` for regression (used by concrete) and `accuracy_score()` for classification (used by ECG and MNIST).

- **`visualize.hpp`** — wraps `matplotlibcpp.h` to produce evaluation plots: actual vs predicted, residuals, and confusion matrices. Plots are saved to `./plots/` and displayed on screen.

## Applications

| Program | Dataset | Model | Expected score |
|---|---|---|---|
| `main_concrete.cpp` | `concrete.csv` | LinearRegression | R² ~ 0.6 |
| `main_ecg.cpp` | `ecg.csv` | LogisticRegression (binary) | ~99% accuracy |
| `main_mnist.cpp` | `mnist_train.csv` + `mnist_test.csv` | LogisticRegression (multi-class softmax) | ~90% accuracy |

## How to run the project

The project is designed to be run from WSL (Ubuntu) on Windows, or any Linux environment with `g++` and Python 3 installed.

### Step 1: Get the data

The `data/` folder needs to contain the following CSV files:
- `concrete.csv` — concrete strength dataset
- `ecg.csv` — ECG anomaly detection dataset
- `mnist_train.csv` — MNIST training data
- `mnist_test.csv` — MNIST test data

These can be obtained two ways:

**Option A — generate them locally** by running the included Python script:
```
python3 generate_mnist_csv.py
```
This downloads the MNIST dataset and produces the required CSV files. The concrete and ECG datasets need to be supplied separately.

**Option B — download from OneDrive** (recommended as a fallback if anything goes wrong with the script):
https://gla-my.sharepoint.com/:f:/g/personal/2917444a_student_gla_ac_uk/IgDTXauKkff-SaWynTKQfJcyAQa1GDCtvRHjxaVGmFIFKeQ?e=hFRVvP

Drop the downloaded files directly into the `./data/` folder.

### MNIST dataset variants

There are several MNIST files of different sizes. They all contain the same kind of data but differ in how many samples are included.

`main_mnist.cpp` uses `mnist_train.csv` and `mnist_test.csv` by default.

### Step 2: Build the executables

From the project root, make the build script executable (only needed once) and run it:
```
chmod +x build.sh
./build.sh
```

This compiles all three programs and puts them in the `./bin/` folder.

### Step 3: Run each program

```
./bin/concrete.out
./bin/ecg.out
./bin/mnist.out
```

Each program loads a dataset, trains the model, prints the score, and saves plots to `./plots/`.

## Dependencies

The plotting code links against the Python interpreter (`matplotlibcpp` is a C++ wrapper around Python's matplotlib). In WSL these can be installed with:
```
sudo apt install python3-dev python3-matplotlib
```

If `python3-config` is not found when running `build.sh`, this is the package that needs installing.

## Use of generative AI

Generative AI helped scaffold the project structure and assist with debugging. All uses of AI have been explicitly declared. 