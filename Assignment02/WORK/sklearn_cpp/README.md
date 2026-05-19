# sklearn_cpp

A header-only C++ library that re-implements two scikit-learn classes from scratch:
`LinearRegression` and `LogisticRegression` (both binary and multi-class).

**Course:** ENG3091 · C++ OOP · Assignment 02

---

## Quick start

```bash
# 1. Put the three CSV files inside ./data/
#    concrete.csv, ecg.csv, mnist_micro.csv

# 2. Build everything
./build.sh

# 3. Run
./bin/concrete.out
./bin/ecg.out
./bin/mnist.out
```

If `./build.sh` doesn't run, do `chmod +x build.sh` once.

---

## What's in the box

```
sklearn_cpp/
├── include/sklearn_cpp/
│   ├── utils/
│   │   ├── csv_reader.hpp           DONE — CSV loader used by all 3 apps
│   │   └── metrics.hpp              DONE — r2_score and accuracy_score
│   └── linear_model/
│       ├── linear_regression.hpp    SKELETON — Person A fills in TODOs
│       └── logistic_regression.hpp  SKELETON — Person B and C share this
├── examples/
│   ├── main_concrete.cpp            Person A runs this to test their code
│   ├── main_ecg.cpp                 Person B runs this to test their code
│   └── main_mnist.cpp               Person C runs this to test their code
├── docs/
│   ├── PERSON_A_GUIDE.md            Person A's exact instructions
│   ├── PERSON_B_GUIDE.md            Person B's exact instructions
│   └── PERSON_C_GUIDE.md            Person C's exact instructions
├── data/                            put CSV files here
├── build.sh                         one-shot build script
└── README.md                        this file
```

---

## Who does what

| Person | File they edit | Methods they implement | Test program |
|---|---|---|---|
| **A** | `linear_regression.hpp` | `fit`, `predict` | `main_concrete.cpp` |
| **B** | `logistic_regression.hpp` | `fit_binary_`, `predict_binary_` | `main_ecg.cpp` |
| **C** | `logistic_regression.hpp` | `fit_multiclass_`, `predict_multiclass_` | `main_mnist.cpp` |

Each person should read their `docs/PERSON_X_GUIDE.md` for the full walkthrough.

**Important:** Person B and C both edit the *same file* but completely
different methods. Use git branches (`feature/binary` and `feature/multiclass`)
to avoid merge conflicts.

---

## How the code is structured (OOP design)

Three design decisions and the reasoning behind each:

1. **No artificial inheritance hierarchy.**
   The two regression classes are independent. They share an interface
   (`fit`, `predict`, `score`) but no base class. This mirrors scikit-learn's
   own design and avoids forcing different return types into a common
   virtual signature.

2. **Composition over inheritance for utilities.**
   The CSV reader and metrics live in `sklearn_cpp::utils`. Both model
   classes use them by composition (calling free functions), not by
   inheriting from a utility base class.

3. **Single dispatching public API on `LogisticRegression`.**
   `fit()` and `predict()` are public and figure out whether to call the
   binary path or the multi-class path based on how many unique labels are
   in `y`. The user never has to care which path is being used. The two
   paths live in clearly-marked private methods so Person B and C can
   work independently.

4. **Strict encapsulation.**
   All learned parameters (`weights_`, `bias_`, `W_`, `b_`) are private.
   Const-correct accessors (`weights()`, `bias()`) are provided for
   inspection and testing only.

5. **Header-only with `#pragma once`.**
   Every header begins with `#pragma once` (the modern guard). Inline
   static methods are used for math helpers so they can be defined in
   the header without violating the one-definition rule.

---

## Compilation explained

```
g++ -std=c++17 -O2 -I ./include examples/main_concrete.cpp -o concrete.out
     ^^^^^^^^^^^^^^  ^^^^^^^^^^^^                                 ^^^^^^^^^^^^
     C++17 + optim   tells g++ where headers live                 binary name
```

The `-I ./include` flag is essential — it lets the compiler resolve
`#include "sklearn_cpp/utils/csv_reader.hpp"` in our source files.

No linking is needed because the entire library is in headers.

---

## Dataset reference

| File | has_header | Samples | Features | Classes | Goal |
|---|---|---|---|---|---|
| `concrete.csv`    | true  | 1030 | 8   | (regression) | predict strength |
| `ecg.csv`         | false | 4997 | 140 | 2  | classify normal/abnormal |
| `mnist_micro.csv` | false | 3000 | 784 | 10 | recognise handwritten digits |
