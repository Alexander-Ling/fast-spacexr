// Rcpp glue for the compiled full-mode RCTD solver; the numerical kernel lives in irwls_core.h.

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include "irwls_core.h"

// Exposed for testing against quadprog::solve.QP.
// [[Rcpp::export]]
Rcpp::List bound_qp_cpp(const arma::mat& D, const arma::vec& d, const arma::vec& lb) {
  if (D.n_rows != D.n_cols || d.n_elem != D.n_rows || lb.n_elem != D.n_rows)
    Rcpp::stop("bound_qp_cpp: inconsistent dimensions");
  arma::vec s;
  const bool ok = spx::bound_qp(D, d, lb, s);
  return Rcpp::List::create(Rcpp::Named("solution") = s, Rcpp::Named("ok") = ok);
}

// [[Rcpp::export]]
Rcpp::List irwls_batch_cpp(const arma::mat& S_base,        // n_genes x K, reference profiles (renormalised)
                           const arma::vec& nUMI,          // n_beads
                           const arma::mat& beads,         // n_beads x n_genes, counts
                           const arma::mat& Q_mat, const arma::mat& SQ_mat, const arma::vec& X_vals,
                           const double K_val, const double min_change, const int n_iter,
                           const int n_threads) {
  try {
    spx::BatchResult r = spx::irwls_batch(S_base, nUMI, beads, Q_mat, SQ_mat, X_vals, K_val, min_change, n_iter, n_threads);
    Rcpp::LogicalVector conv(r.converged.size());
    for (std::size_t i = 0; i < r.converged.size(); ++i) conv[i] = r.converged[i] != 0;
    return Rcpp::List::create(Rcpp::Named("weights") = r.weights, Rcpp::Named("converged") = conv,
                              Rcpp::Named("status") = Rcpp::wrap(r.status),
                              Rcpp::Named("iterations") = Rcpp::wrap(r.iterations));
  } catch (const std::invalid_argument& e) {
    Rcpp::stop(std::string("irwls_batch_cpp: ") + e.what());
  }
  return R_NilValue;   // not reached
}
