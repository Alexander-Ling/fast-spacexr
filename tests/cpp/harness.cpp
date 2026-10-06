// Standalone test harness for src/irwls_core.h (no R). Build with sanitizers, e.g.
//   g++ -std=c++17 -O1 -g -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer -I../../src harness.cpp -llapack -lblas
// Usage: ./harness <dump_dir>   (files written by dump_bin.R: S, nUMI, beads, Q, SQ, X; Kval.txt)
#include "irwls_core.h"
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <unistd.h>

static arma::mat load(const std::string& f) {
  std::ifstream in(f, std::ios::binary);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", f.c_str()); std::exit(2); }
  int32_t d[2]; in.read(reinterpret_cast<char*>(d), sizeof d);
  arma::mat m(d[0], d[1]); in.read(reinterpret_cast<char*>(m.memptr()), sizeof(double) * m.n_elem);
  return m;
}
static long rss_kb() {
  long pages = 0; std::ifstream f("/proc/self/statm"); long sz; f >> sz >> pages; return pages * (sysconf(_SC_PAGESIZE) / 1024);
}
static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++failures; } } while (0)

struct Ctx { arma::mat S, beads, Q, SQ; arma::vec nUMI, X; double K; };

static spx::BatchResult run(const Ctx& c, const arma::mat& S, const arma::vec& u, const arma::mat& B, int thr, double tol = 1e-3, int it = 50) {
  return spx::irwls_batch(S, u, B, c.Q, c.SQ, c.X, c.K, tol, it, thr);
}
static void report(const char* label, const spx::BatchResult& r) {
  int cnt[10] = {0}; for (int s : r.status) ++cnt[std::min(std::max(s, 0), 9)];
  std::printf("%-44s n=%zu status:", label, r.status.size());
  for (int k = 0; k < 10; ++k) if (cnt[k]) std::printf(" %d:%d", k, cnt[k]);
  std::printf("\n");
}
template <class F> static void expect_throw(const char* label, F f) {
  try { f(); std::printf("%-44s NO EXCEPTION\n", label); ++failures; }
  catch (const std::invalid_argument& e) { std::printf("%-44s threw invalid_argument (ok): %s\n", label, e.what()); }
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  Ctx c; c.S = load(dir + "/S.bin"); c.beads = load(dir + "/beads.bin"); c.Q = load(dir + "/Q.bin"); c.SQ = load(dir + "/SQ.bin");
  c.nUMI = load(dir + "/nUMI.bin").col(0); c.X = load(dir + "/X.bin").col(0);
  { std::ifstream k(dir + "/Kval.txt"); k >> c.K; }
  const arma::uword n = 400;
  arma::mat B = c.beads.rows(0, n - 1); arma::vec U = c.nUMI.subvec(0, n - 1);
  std::printf("S %llux%llu  beads %llux%llu  Q %llux%llu  K_val %g\n", (unsigned long long)c.S.n_rows, (unsigned long long)c.S.n_cols,
              (unsigned long long)c.beads.n_rows, (unsigned long long)c.beads.n_cols, (unsigned long long)c.Q.n_rows, (unsigned long long)c.Q.n_cols, c.K);

  // 1. normal batch; determinism across thread counts (a data race would show up as a difference)
  auto r1 = run(c, c.S, U, B, 1); report("normal, 1 thread", r1);
  CHECK(std::count(r1.status.begin(), r1.status.end(), 0) == (long)n, "all beads should solve");
  for (int t : {2, 4, 8, 32}) {
    auto rt = run(c, c.S, U, B, t);
    const bool same = arma::approx_equal(rt.weights, r1.weights, "absdiff", 0.0) && rt.status == r1.status && rt.iterations == r1.iterations;
    std::printf("  %2d threads bitwise identical to 1 thread: %s\n", t, same ? "yes" : "NO");
    CHECK(same, "thread count changed the result");
  }
  // 2. empty batch, more threads than beads
  report("zero beads", run(c, c.S, arma::vec(), arma::mat(0, c.S.n_rows), 4));
  report("3 beads, 32 threads", run(c, c.S, U.subvec(0, 2), B.rows(0, 2), 32));
  report("n_threads = 0", run(c, c.S, U, B, 0)); report("n_threads = -3", run(c, c.S, U, B, -3));
  // 3. hostile bead data: must be reported through status, never crash
  arma::mat Bn = B; Bn(2, 4) = arma::datum::nan; Bn.row(3).fill(-1.0); Bn(4, 6) = arma::datum::inf; Bn.row(5).zeros(); Bn.row(6).fill(1e6); Bn(7, 0) = -0.5;
  auto rn = run(c, c.S, U, Bn, 4); report("NaN / negative / Inf / zero / huge counts", rn);
  CHECK(rn.status[2] == 4 && rn.status[3] == 4 && rn.status[7] == 4, "NaN / negative counts must give status 4");
  CHECK(rn.status[4] == 0, "Inf count is clipped to K_val like the R code, not an error");
  arma::vec Un = U; Un[1] = arma::datum::nan; Un[2] = 0; Un[3] = 1e12; Un[4] = -5; Un[5] = arma::datum::inf;
  report("NaN / zero / huge / negative / Inf nUMI", run(c, c.S, Un, B, 4));
  // 4. hostile reference matrices
  report("single cell type (K=1)", run(c, c.S.col(0), U, B, 2));
  report("two genes, two types", run(c, c.S.submat(0, 0, 1, 1), U, B.cols(0, 1), 2));
  arma::mat Sneg = c.S; Sneg.submat(0, 0, 4, 0).fill(-1.0); report("negative profile entries", run(c, Sneg, U, B, 2));
  arma::mat Sz = c.S; Sz.col(1).zeros(); report("all-zero profile column", run(c, Sz, U, B, 2));
  arma::mat Sd = c.S; Sd.col(2) = Sd.col(1); report("duplicated columns (rank deficient)", run(c, Sd, U, B, 2));
  arma::mat Sbig = c.S * 1e9; report("profiles scaled by 1e9", run(c, Sbig, U, B, 2));
  arma::mat Ssm = c.S * 1e-12; report("profiles scaled by 1e-12", run(c, Ssm, U, B, 2));
  // 5. invalid arguments must throw, not crash
  expect_throw("beads/gene mismatch", [&] { run(c, c.S, U, B.cols(0, 9), 2); });
  expect_throw("nUMI length mismatch", [&] { run(c, c.S, U.subvec(0, 9), B, 2); });
  expect_throw("SQ dimension mismatch", [&] { spx::irwls_batch(c.S, U, B, c.Q, c.SQ.cols(0, 99), c.X, c.K, 1e-3, 50, 2); });
  expect_throw("X_vals length mismatch", [&] { spx::irwls_batch(c.S, U, B, c.Q, c.SQ, c.X.subvec(0, 99), c.K, 1e-3, 50, 2); });
  expect_throw("K_val larger than table", [&] { spx::irwls_batch(c.S, U, B, c.Q, c.SQ, c.X, 1e6, 1e-3, 50, 2); });
  expect_throw("n_iter = 0", [&] { run(c, c.S, U, B, 2, 1e-3, 0); });
  expect_throw("negative tolerance", [&] { run(c, c.S, U, B, 2, -1.0, 50); });
  arma::mat Snan = c.S; Snan(0, 0) = arma::datum::nan; expect_throw("NaN in reference", [&] { run(c, Snan, U, B, 2); });
  // 6. QP solver on random and degenerate problems
  std::mt19937 gen(1); std::normal_distribution<double> nd(0, 1); std::uniform_real_distribution<double> ud(0, 1);
  int qp_ok = 0, qp_total = 0;
  for (int i = 0; i < 3000; ++i) {
    const int K = 1 + gen() % 19; arma::mat A(K, K); for (auto& x : A) x = nd(gen);
    arma::mat D = A.t() * A / K + 1e-7 * arma::eye(K, K); arma::vec d(K), lb(K), s;
    for (int k = 0; k < K; ++k) { d[k] = nd(gen); lb[k] = ud(gen) > 0.3 ? -std::fabs(nd(gen)) : 0.0; }
    ++qp_total; if (spx::bound_qp(D, d, lb, s)) { ++qp_ok; CHECK((s - lb).min() >= -1e-9, "QP solution violates a bound"); }
  }
  std::printf("%-44s %d/%d converged\n", "bound_qp random PD problems", qp_ok, qp_total);
  { arma::vec s; const bool ok = spx::bound_qp(arma::zeros(3, 3), arma::vec{1, 2, 3}, arma::vec{0, 0, -1}, s); std::printf("%-44s ok=%d (no crash)\n", "bound_qp singular D", (int)ok); }
  // 7. repeated-call memory growth (leaks show up as RSS growth). LeakSanitizer reports at exit when enabled.
  const long r0 = rss_kb();
  for (int rep = 0; rep < 1500; ++rep) { auto r = run(c, c.S, U.subvec(0, 49), B.rows(0, 49), 4); if (r.status.empty()) return 3; if (rep == 100) std::printf("  rss after 100 calls: %ld kB\n", rss_kb()); }
  const long r1k = rss_kb();
  std::printf("  rss before %ld kB, after 1500 calls %ld kB (growth %ld kB)\n", r0, r1k, r1k - r0);
  if (std::getenv("PLANT_LEAK")) {   // self-test of the leak detector: one deliberate 4 kB leak
    int* leak = new int[1000]; leak[0] = 1; asm volatile("" : : "r"(leak) : "memory"); leak = nullptr;
    std::printf("(planted a deliberate leak)\n");
  }
  std::printf("\n%s (%d check failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
