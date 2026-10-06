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

// Doublet mode for many beads. Returns the pieces of process_bead_doublet()'s result for every bead; beads whose
// `status` is not 0 must be re-run with the R implementation. `cls` gives each cell type's class id.
// [[Rcpp::export]]
Rcpp::List doublet_batch_cpp(const arma::mat& S_base, const arma::vec& nUMI, const arma::mat& beads,
                             const arma::mat& Q_mat, const arma::mat& SQ_mat, const arma::vec& X_vals,
                             const double K_val, const double min_change, const Rcpp::IntegerVector& cls,
                             const double conf_thresh, const double doublet_thresh, const int n_threads) {
  try {
    std::vector<int> cl(cls.begin(), cls.end());
    spx::DoubletBatchResult r = spx::doublet_batch(S_base, nUMI, beads, Q_mat, SQ_mat, X_vals, K_val, min_change, cl,
                                                   conf_thresh, doublet_thresh, n_threads);
    const std::size_t nb = r.beads.size();
    const arma::uword K = S_base.n_cols;
    arma::mat all_w(nb, K, arma::fill::zeros), dw(nb, 2, arma::fill::zeros);
    Rcpp::IntegerVector first(nb), second(nb), spot(nb);
    Rcpp::NumericVector min_score(nb), singlet_score(nb);
    Rcpp::LogicalVector conv_all(nb), conv_d(nb), fcl(nb), scl(nb);
    Rcpp::List cand(nb), score_mat(nb), singlets(nb);
    for (std::size_t i = 0; i < nb; ++i) {
      if (r.status[i] != 0) continue;
      const spx::DoubletBead& b = r.beads[i];
      all_w.row(i) = b.all_weights.t(); dw.row(i) = b.doublet_weights.t();
      first[i] = b.first + 1; second[i] = b.second + 1; spot[i] = b.spot_class;
      min_score[i] = b.min_score; singlet_score[i] = b.singlet_score;
      conv_all[i] = b.conv_all; conv_d[i] = b.conv_doublet; fcl[i] = b.first_class; scl[i] = b.second_class;
      Rcpp::IntegerVector c(b.cand.size());
      for (std::size_t k = 0; k < b.cand.size(); ++k) c[k] = b.cand[k] + 1;
      cand[i] = c; score_mat[i] = Rcpp::wrap(b.score_mat); singlets[i] = Rcpp::wrap(b.singlet_scores);
    }
    return Rcpp::List::create(Rcpp::Named("all_weights") = all_w, Rcpp::Named("doublet_weights") = dw,
                              Rcpp::Named("first") = first, Rcpp::Named("second") = second, Rcpp::Named("spot") = spot,
                              Rcpp::Named("min_score") = min_score, Rcpp::Named("singlet_score") = singlet_score,
                              Rcpp::Named("conv_all") = conv_all, Rcpp::Named("conv_doublet") = conv_d,
                              Rcpp::Named("first_class") = fcl, Rcpp::Named("second_class") = scl,
                              Rcpp::Named("cand") = cand, Rcpp::Named("score_mat") = score_mat,
                              Rcpp::Named("singlet_scores") = singlets, Rcpp::Named("status") = Rcpp::wrap(r.status));
  } catch (const std::invalid_argument& e) {
    Rcpp::stop(std::string("doublet_batch_cpp: ") + e.what());
  }
  return R_NilValue;   // not reached
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
