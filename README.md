# Burnout Prediction — Parallel vs Sequential ML in C++

Predicts developer burnout levels (Low / Medium / High) from 11 workplace
features using multi-feature linear regression trained with gradient descent.
Compares a sequential single-threaded implementation against a parallelised
version using OpenMP across a real-world dataset of 6,750 developers.

## Results

### Speedup across dataset sizes

| Rows    | Sequential | Parallel | Speedup |
|---------|------------|----------|---------|
| 27,450  | 10,990ms   | 3,165ms  | 3.47x   |
| 54,900  | 21,961ms   | 6,175ms  | 3.56x   |
| 109,800 | 44,388ms   | 12,223ms | 3.63x   |
| 274,500 | 111,449ms  | 37,337ms | 2.98x   |

Peak speedup of **3.63x** at 109,800 rows. Both trainers converge to
identical weights, validated automatically by the GoogleTest suite.

### Top predictors by learned weight

| Feature            | Weight  | Interpretation                  |
|--------------------|---------|---------------------------------|
| stress_level       | +0.784  | Strongest burnout driver        |
| bugs_per_day       | +0.479  | High bug load increases burnout |
| daily_work_hours   | +0.460  | Long hours increase burnout     |
| meetings_per_day   | +0.329  | Meeting load adds to burnout    |
| screen_time        | +0.314  | More screen time = more burnout |
| sleep_hours        | -0.257  | Protective factor               |
| exercise_hours     | -0.147  | Protective factor               |

## How to build and run

**Requirements:** C++17, CMake 3.21+, MinGW/GCC with OpenMP, Git, Ninja

```bash
git clone https://github.com/mufhatu/burnout-prediction-ml-openmp
cd ml_burnout_parallel
mkdir build && cd build
cmake -G "Ninja" ..
ninja
./main
```

## Run tests

```bash
./runTests
```
Expected output:
[==========] Running 3 tests from 3 test suites.
[  PASSED  ] 3 tests.

## Project structure

ml_burnout_parallel/
├── data/
│   └── developer_burnout_dataset.csv   # 6,750 rows, 12 columns
├── include/
│   ├── sequential_trainer.h
│   ├── parallel_trainer.h
│   ├── data_loader.h
│   └── encoder.h
├── src/
│   ├── main.cpp
│   ├── sequential_trainer.cpp          # Gradient descent, single-threaded
│   ├── parallel_trainer.cpp            # Gradient descent, OpenMP parallelised
│   └── data_loader.cpp
├── tests/
│   └── test_main.cpp                   # GoogleTest suite (3 tests)
└── CMakeLists.txt
## Tech stack

- C++17
- OpenMP — data-parallel gradient accumulation across training samples
- CMake + Ninja — build system
- GoogleTest — unit testing

## Key engineering decisions

**Why gradient descent over least squares?**
Gradient descent scales to any number of features and creates a meaningful
parallelism opportunity in the inner loop. Least squares solves in one pass
with no parallel inner loop to exploit.

**Why OpenMP over std::thread?**
The parallelism is purely data-parallel — the same computation applied
independently to each training sample. OpenMP's parallel for reduction
expresses this in two lines. std::thread would require manual work
partitioning and synchronisation for no benefit.

**Why normalise features?**
Raw features span very different ranges (age 20–60, caffeine 0–500mg).
Without normalisation, large-scale features dominate the gradient and
training diverges to NaN within a few epochs.

**Why does speedup drop at 274,500 rows?**
The normalisation step and weight update are sequential — they do not
scale with thread count. At very large sizes these sequential sections
become a larger fraction of total runtime, capping speedup per
Amdahl's Law.
