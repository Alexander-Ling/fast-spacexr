// Compiled core for the full-mode RCTD solver (unconstrained weights, constrain = FALSE).
//
// This is a port of the R path
//   solveIRWLS.weights -> solveWLS -> get_der_fast / calc_Q_all / psd / quadprog::solve.QP
// for one bead, run over many beads in parallel with OpenMP. The only algorithmic substitution is
// the quadratic program: solve.QP(D, d, I, -solution) is a strictly convex QP with only lower
// bounds on the step, solved here with a primal active-set method (warm-started at the bound).
// Anything the C++ path cannot handle (non-finite values, eigen or QP failure) is reported
// through `status` so the caller can re-run that bead with the original R code.

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include <vector>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Tables {
  const double* Q;    // (K_val + 3) x nX, column-major
  const double* SQ;   // same shape
  const double* X;    // nX
  int nrow;
  int nX;
  double Xmax;
  int Kval;
};

// First and second derivative of the Poisson-lognormal log-likelihood at count y and rate lam, by
// cubic-spline interpolation in the precomputed Q table. Mirrors calc_Q_all() (d1_vec, d2_vec).
inline void q_deriv(int y, double lam, const Tables& T, double& d1, double& d2) {
  const double eps = 1e-4, delta = 1e-6;
  lam = std::min(std::max(eps, lam), T.Xmax - eps);
  const double l = std::floor(std::sqrt(lam / delta));
  const double m = std::min(l - 9.0, 40.0) +
                   std::max(std::ceil(std::sqrt(std::max(l - 48.7499, 0.0) * 4.0)) - 2.0, 0.0);
  const int mi = static_cast<int>(m);             // 1-based column index into Q / X
  const double ti1 = T.X[mi - 1], ti = T.X[mi], hi = ti - ti1;
  const std::size_t r = static_cast<std::size_t>(y);
  const std::size_t c0 = static_cast<std::size_t>(mi - 1) * T.nrow, c1 = static_cast<std::size_t>(mi) * T.nrow;
  const double fti1 = T.Q[r + c0], fti = T.Q[r + c1];
  const double zi1 = T.SQ[r + c0], zi = T.SQ[r + c1];
  const double diff1 = lam - ti1, diff2 = ti - lam;
  const double diff3 = fti / hi - zi * hi / 6.0, diff4 = fti1 / hi - zi1 * hi / 6.0;
  const double zdi = zi / hi, zdi1 = zi1 / hi;
  d1 = zdi * diff1 * diff1 / 2.0 - zdi1 * diff2 * diff2 / 2.0 + diff3 - diff4;
  d2 = zdi * diff1 + zdi1 * diff2;
}

// min 1/2 s'Ds - d's   subject to  s >= lb   (lb <= 0, so s = 0 is feasible).
// Primal active-set method. Returns false if it fails to converge.
bool bound_qp(const arma::mat& D, const arma::vec& d, const arma::vec& lb, arma::vec& s) {
  const arma::uword K = D.n_rows;
  s.zeros(K);
  std::vector<char> act(K, 0);                     // 1 = pinned at its lower bound
  for (arma::uword i = 0; i < K; ++i) if (lb[i] >= 0.0) act[i] = 1;
  const int max_iter = 10 * static_cast<int>(K) + 50;
  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<arma::uword> F, A;
    for (arma::uword i = 0; i < K; ++i) (act[i] ? A : F).push_back(i);
    for (arma::uword i : A) s[i] = lb[i];
    if (!F.empty()) {
      arma::uvec fi(F), ai(A);
      arma::vec rhs = d.elem(fi);
      if (!A.empty()) rhs -= D.submat(fi, ai) * s.elem(ai);
      arma::vec sf;
      if (!arma::solve(sf, D.submat(fi, fi), rhs, arma::solve_opts::likely_sympd)) return false;
      arma::vec cur = s.elem(fi), p = sf - cur;
      double t = 1.0; int block = -1;
      for (arma::uword k = 0; k < F.size(); ++k) {
        if (p[k] < 0.0) {
          const double tk = (lb[F[k]] - cur[k]) / p[k];
          if (tk < t) { t = tk; block = static_cast<int>(k); }
        }
      }
      if (t < 0.0) t = 0.0;
      s.elem(fi) = cur + t * p;
      if (block >= 0) {                              // hit a bound: pin it and re-solve
        act[F[block]] = 1; s[F[block]] = lb[F[block]];
        continue;
      }
    }
    // full step taken: check the multipliers of the pinned variables
    const arma::vec g = D * s - d;
    int worst = -1; double gmin = -1e-12;
    for (arma::uword i : A) if (g[i] < gmin) { gmin = g[i]; worst = static_cast<int>(i); }
    if (worst < 0) return true;
    act[worst] = 0;
  }
  return false;
}

// One bead. Returns 0 on success, nonzero on failure.
int irwls_bead(const arma::mat& Sb, const double nUMI, const double* B, const Tables& T,
               const double min_change, const int n_iter, arma::vec& w, int& iters, bool& converged) {
  const arma::uword n = Sb.n_rows, K = Sb.n_cols;
  const arma::mat S = Sb * nUMI;
  arma::vec sol(K); sol.fill(1.0 / static_cast<double>(K));
  const double threshold = std::max(1e-4, nUMI * 1e-7);
  arma::vec pred(n), d1(n), d2(n), s0(K), grad(K), vals(K), step(K), newsol(K);
  arma::mat H(K, K), vecs(K, K), D(K, K);
  double change = 1.0; int it = 0;
  while (change > min_change && it < n_iter) {
    s0 = arma::clamp(sol, 0.0, arma::datum::inf);
    pred = arma::abs(S * s0);
    for (arma::uword g = 0; g < n; ++g) {
      const double lam = pred[g] < threshold ? threshold : pred[g];
      double a, b;
      q_deriv(static_cast<int>(B[g]), lam, T, a, b);
      d1[g] = a; d2[g] = b;
    }
    if (!d1.is_finite() || !d2.is_finite()) return 1;
    grad = S.t() * d1;                                    // = -derivatives$grad
    H = S.t() * (S.each_col() % (-d2));                   // Hessian: -sum_g d2_g S_gi S_gj
    H = 0.5 * (H + H.t());
    if (!arma::eig_sym(vals, vecs, H)) return 2;
    vals = arma::clamp(vals, 1e-3, arma::datum::inf);
    const double norm_factor = vals.max();
    D = vecs * (arma::diagmat(vals) * vecs.t());
    D /= norm_factor;
    grad /= norm_factor;
    D.diag() += 1e-7;
    if (!bound_qp(D, grad, -s0, step)) return 3;
    newsol = s0 + 0.3 * step;
    change = arma::accu(arma::abs(newsol - sol));
    sol = newsol;
    ++it;
  }
  w = sol; iters = it; converged = (change <= min_change);
  return 0;
}

}  // namespace

// Exposed for testing against quadprog::solve.QP.
// [[Rcpp::export]]
Rcpp::List bound_qp_cpp(const arma::mat& D, const arma::vec& d, const arma::vec& lb) {
  arma::vec s;
  const bool ok = bound_qp(D, d, lb, s);
  return Rcpp::List::create(Rcpp::Named("solution") = s, Rcpp::Named("ok") = ok);
}

// [[Rcpp::export]]
Rcpp::List irwls_batch_cpp(const arma::mat& S_base,        // n_genes x K, reference profiles (renormalised)
                           const arma::vec& nUMI,          // n_beads
                           const arma::mat& beads,         // n_beads x n_genes, counts
                           const arma::mat& Q_mat, const arma::mat& SQ_mat, const arma::vec& X_vals,
                           const double K_val, const double min_change, const int n_iter,
                           const int n_threads) {
  const arma::uword nb = beads.n_rows, K = S_base.n_cols, ng = S_base.n_rows;
  if (beads.n_cols != ng) Rcpp::stop("irwls_batch_cpp: beads and S_base disagree on the number of genes");
  Tables T{Q_mat.memptr(), SQ_mat.memptr(), X_vals.memptr(), static_cast<int>(Q_mat.n_rows),
           static_cast<int>(X_vals.n_elem), X_vals.max(), static_cast<int>(K_val)};
  arma::mat W(nb, K, arma::fill::zeros);
  std::vector<int> status(nb, 0), iters(nb, 0); std::vector<char> conv(nb, 0);
  const arma::mat beadsT = beads.t();                       // genes x beads: each bead contiguous
  const int nt = std::max(1, n_threads);
  #pragma omp parallel num_threads(nt)
  {
    arma::vec w(K), clipped(ng);
    #pragma omp for schedule(dynamic, 8)
    for (long long i = 0; i < static_cast<long long>(nb); ++i) {
      const double* col = beadsT.colptr(i);
      for (arma::uword g = 0; g < ng; ++g) clipped[g] = std::min(col[g], K_val);   // B[B > K_val] <- K_val
      int it = 0; bool cv = false; int st;
      try { st = irwls_bead(S_base, nUMI[i], clipped.memptr(), T, min_change, n_iter, w, it, cv); }
      catch (...) { st = 9; }
      status[i] = st; iters[i] = it; conv[i] = cv ? 1 : 0;
      if (st == 0) for (arma::uword k = 0; k < K; ++k) W(i, k) = w[k];
    }
  }
  Rcpp::LogicalVector conv_r(nb);
  for (arma::uword i = 0; i < nb; ++i) conv_r[i] = conv[i] != 0;
  return Rcpp::List::create(Rcpp::Named("weights") = W, Rcpp::Named("converged") = conv_r,
                            Rcpp::Named("status") = Rcpp::wrap(status), Rcpp::Named("iterations") = Rcpp::wrap(iters));
}
