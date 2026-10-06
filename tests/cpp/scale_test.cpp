// Thread-scaling test for the full-mode kernel (no R). Usage: ./scale_test <dump_dir> [n_beads]
// Prints entities/s at several thread counts; used to find contention (allocator, BLAS locks, false sharing).
#include "irwls_core.h"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>

static arma::mat load(const std::string& f) {
  std::ifstream in(f, std::ios::binary);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", f.c_str()); std::exit(2); }
  int32_t d[2]; in.read(reinterpret_cast<char*>(d), sizeof d);
  arma::mat m(d[0], d[1]); in.read(reinterpret_cast<char*>(m.memptr()), sizeof(double) * m.n_elem);
  return m;
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  const arma::uword n_total = argc > 2 ? static_cast<arma::uword>(std::atoi(argv[2])) : 12000;
  arma::mat S = load(dir + "/S.bin"), B0 = load(dir + "/beads.bin"), Q = load(dir + "/Q.bin"), SQ = load(dir + "/SQ.bin");
  arma::vec U0 = load(dir + "/nUMI.bin").col(0), X = load(dir + "/X.bin").col(0); double K; { std::ifstream k(dir + "/Kval.txt"); k >> K; }
  arma::mat B(n_total, B0.n_cols); arma::vec U(n_total);
  for (arma::uword i = 0; i < n_total; ++i) { B.row(i) = B0.row(i % B0.n_rows); U[i] = U0[i % B0.n_rows]; }
  std::printf("%-8s %-10s %-12s\n", "threads", "entities/s", "efficiency");
  double base = 0;
  for (int th : {1, 2, 4, 8, 16, 28}) {
    const arma::uword n = th == 1 ? n_total / 4 : n_total;
    const auto t0 = std::chrono::steady_clock::now();
    auto r = spx::irwls_batch(S, U.subvec(0, n - 1), B.rows(0, n - 1), Q, SQ, X, K, 1e-3, 50, th);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double eps = n / s; if (th == 1) base = eps;
    std::printf("%-8d %-10.0f %-12.0f%%  (ok beads: %ld)\n", th, eps, 100.0 * eps / (base * th), (long)std::count(r.status.begin(), r.status.end(), 0));
  }
  return 0;
}
