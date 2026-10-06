# Standalone tests for the compiled solver (`src/irwls_core.h`)

`harness.cpp` exercises the numerical kernel without an R runtime, so it can be built with
AddressSanitizer, UndefinedBehaviorSanitizer and LeakSanitizer (plain R aborts at startup under an
ASan `LD_PRELOAD`, so the kernel is kept free of the R API for exactly this reason).

What it checks:

* a normal batch (full mode and doublet mode) solves every bead, and the result is bitwise identical for 1, 2, 4, 8 and 32 threads
  (a data race between beads would show up as a difference);
* hostile inputs are reported through `status` and never crash: NaN / negative / infinite / zero /
  huge counts and nUMI, one cell type, two genes, rank-deficient and badly scaled references;
* invalid arguments (dimension mismatches, table too small, `n_iter < 1`, NaN reference) throw
  `std::invalid_argument`;
* the bound-constrained QP on 3,000 random positive-definite problems converges and respects its bounds;
* repeated calls do not grow resident memory, and LeakSanitizer reports nothing at exit
  (`PLANT_LEAK=1` plants a deliberate 4 kB leak to prove the detector is live).

Data: `dump_bin.R` (in the analysis scripts) writes the reference profiles, counts and likelihood tables
from a real run as raw doubles; the harness reads them from the directory given as its argument.

```sh
g++ -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer \
    -I../../src -I<RcppArmadillo>/include -DARMA_DONT_USE_OPENMP harness.cpp -o harness_asan -llapack -lblas -lgfortran
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ./harness_asan <dump_dir>
```
