# ENG3091 APSE Assignment 2: sklearn implementation in C++

Header-only C++ implementation of `LinearRegression` and `LogisticRegression`.

## Implementation details: 
- `linear_regression.hpp`: 
    
- `logistic_regression.hpp`:

The system uses three utilities:
- `csv_reader.hpp` converts CSV file line by line into `std::vector`, caple of handling CSVs of any dimention.
- `metrics.hpp`
- `visualize.hpp`


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

## Folder layout

```
include/sklearn_cpp/
    utils/             CSV reader and metrics
    linear_model/      LinearRegression and LogisticRegression
examples/              main programs
data/                  CSV files
```