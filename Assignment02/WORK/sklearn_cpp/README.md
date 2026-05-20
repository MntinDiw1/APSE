# sklearn_cpp

Header-only C++ implementation of `LinearRegression` and `LogisticRegression` for ENG3091 Assignment 2.

Three applications:
- `main_concrete.cpp` — LinearRegression on `concrete.csv` (R² ~ 0.6)
- `main_ecg.cpp` — LogisticRegression on `ecg.csv` (~99% accuracy)
- `main_mnist.cpp` — LogisticRegression with softmax on `mnist_micro.csv` (~85% accuracy)

## Setup

Drop `concrete.csv`, `ecg.csv`, and `mnist_micro.csv` into `./data/`.

If you need `mnist_micro.csv`, generate it with:
```
python3 generate_mnist_csv.py
```

## Build and run

All three at once:
```
./build.sh
./bin/concrete.out
./bin/ecg.out
./bin/mnist.out
```

Or individually from the project root:
```
g++ -I ./include examples/main_concrete.cpp -o concrete.out
g++ -I ./include examples/main_ecg.cpp -o ecg.out
g++ -I ./include examples/main_mnist.cpp -o mnist.out
```

## Folder layout

```
include/sklearn_cpp/
    utils/             CSV reader and metrics
    linear_model/      LinearRegression and LogisticRegression
examples/              main programs
data/                  CSV files
```