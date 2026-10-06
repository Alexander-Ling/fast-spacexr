test_single_beads <- function(puck, gene_list, cell_type_info, trust_model = FALSE, constrain = T, OLS = F) {
  cell_type_names = cell_type_info[[2]]; n_cell_types = cell_type_info[[3]]
  beads = t(as.matrix(puck@counts[gene_list,]))
  weights = decompose_batch(puck@nUMI, cell_type_info[[1]], beads, gene_list, constrain = constrain, OLS = OLS)
  pred_labels = unlist(lapply(weights,function(x) which.max(x$weights)))
  cell_type_lev = factor(1:n_cell_types)
  cell_type_map = data.frame(cindex = 1:n_cell_types, row.names = cell_type_names)
  if(trust_model)
    true_labels = pred_labels
  else
    true_labels = lapply(puck@cell_labels, function(x) cell_type_map[as.character(x),"cindex"])
  conf_mat = caret::confusionMatrix(factor(pred_labels,cell_type_lev),factor(true_labels,cell_type_lev))
  rownames(conf_mat$table) = cell_type_names; colnames(conf_mat$table) = cell_type_names
  return(list(conf_mat,weights,pred_labels))
}

#' Runs RCTD in full mode on \code{puck}
#'
#' Renormalizes \code{cell_type_means} to have average the same as the puck
#' if \code{proportions} is given. Then, computes cell type proportions for each pixel
#' in \code{puck}.
#'
#' @param proportions (optional) If given, a named list (for each cell type) of proportion of the cell type on the bulk dataset
#' (not constrained to sum to 1)
#' @param gene_list a list of genes to be used for RCTD
#' @param puck an object of type \linkS4class{SpatialRNA}, the target dataset
#' @param cell_type_info cell type information and profiles of each cell, calculated from the scRNA-seq
#' reference (see \code{\link{get_cell_type_info}})
#' @param constrain logical whether to constrain the weights to sum to one on each pixel
#' @return Returns \code{test_results}, a list of three items:
#' (1) \code{conf_mat} a confusion matrix (not relevant) (2) \code{weights}
#' a dataframe of predicted weights (3) a named list of predicted cell types
#' @export
process_data <- function(puck, gene_list, cell_type_info, proportions = NULL, trust_model = FALSE, constrain = T, OLS = F) {
  cell_type_info_renorm = cell_type_info
  if(!is.null(proportions)) {
    cell_type_info_renorm[[1]] = get_norm_ref(puck, cell_type_info[[1]], gene_list, proportions)
  }
  test_results <- test_single_beads(puck, gene_list, cell_type_info_renorm, trust_model = trust_model, constrain = constrain, OLS = OLS)
  return(test_results)
}


#' Runs RCTD in doublet mode on \code{puck}
#'
#' Then, computes cell type proportions for each pixel in \code{puck}.
#' Classifies each pixel as 'singlet' or 'doublet' and searches for the cell types
#' on the pixel
#'
#' @param class_df A dataframe returned by \code{\link{get_class_df}} to map cell types to
#' classes
#' @param gene_list a list of genes to be used for RCTD
#' @param puck an object of type \linkS4class{SpatialRNA}, the target dataset
#' @param cell_type_info cell type information and profiles of each cell, calculated from the scRNA-seq
#' reference (see \code{\link{get_cell_type_info}})
#' @param constrain logical whether to constrain the weights to sum to one on each pixel
#' @param max_cores number of cores to use (will use parallel processing if more than one).
#' @param CONFIDENCE_THRESHOLD (Default 10) the minimum change in likelihood (compared to other cell types) necessary to determine a cell type identity with confidence
#' @param DOUBLET_THRESHOLD (Default 25) the penalty weight of predicting a doublet instead of a singlet for a pixel
#' @return Returns \code{results}, a list of RCTD results for each pixel, which can be organized by
#' feeding into \code{\link{gather_results}}
#' @export
process_beads_batch <- function(cell_type_info, gene_list, puck, class_df = NULL, constrain = T,
                                MAX_CORES = 8, MIN.CHANGE = 0.001, CONFIDENCE_THRESHOLD = 10, DOUBLET_THRESHOLD = 25) {
  beads = t(as.matrix(puck@counts[gene_list,]))
  #out_file = "logs/process_beads_log.txt"
  #if (file.exists(out_file))
  #  file.remove(out_file)
  one_bead <- function(i)
    process_bead_doublet(cell_type_info, gene_list, puck@nUMI[i], beads[i,],
                         class_df = class_df, constrain = constrain, MIN.CHANGE = MIN.CHANGE,
                         CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = DOUBLET_THRESHOLD)
  if(!constrain && !is.null(class_df) && dim(beads)[1] > 0 && isTRUE(getOption("fastspacexr.use_cpp", TRUE)) &&
     length(cell_type_info[[2]]) >= 2) {
    # compiled doublet search; beads it cannot handle are re-run below with process_bead_doublet
    results <- process_beads_doublet_cpp(cell_type_info, gene_list, puck@nUMI, beads, class_df, MIN.CHANGE,
                                         CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD, MAX_CORES, one_bead)
  } else if(MAX_CORES > 1 && .Platform$OS.type == "unix" && dim(beads)[1] > 1) {
    results <- fork_lapply(dim(beads)[1], MAX_CORES, one_bead)
  } else if(MAX_CORES > 1) {
    numCores = parallel::detectCores();
    if(parallel::detectCores() > MAX_CORES)
      numCores <- MAX_CORES
    cl <- parallel::makeCluster(numCores,setup_strategy = "sequential",outfile="")
    doParallel::registerDoParallel(cl)
    environ = c('decompose_full','decompose_sparse','solveIRWLS.weights','solveOLS','solveWLS','Q_mat','X_vals','K_val', 'SQ_mat')
    results <- foreach::foreach(i = 1:(dim(beads)[1]), .export = environ) %dopar% { #.packages = c("quadprog"),
      #if(i %% 100 == 0)
      #  cat(paste0("Finished sample: ",i,"\n"), file=out_file, append=TRUE)
      assign("Q_mat",Q_mat, envir = globalenv()); assign("X_vals",X_vals, envir = globalenv())
      assign("K_val",K_val, envir = globalenv()); assign("SQ_mat",SQ_mat, envir = globalenv());
      result = process_bead_doublet(cell_type_info, gene_list, puck@nUMI[i], beads[i,],
                                    class_df = class_df, constrain = constrain, MIN.CHANGE = MIN.CHANGE,
                                    CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = DOUBLET_THRESHOLD)
      result
    }
    parallel::stopCluster(cl)
  } else {
    #not parallel
    results <- list()
    for(i in 1:(dim(beads)[1])) {
      results[[i]] <- process_bead_doublet(cell_type_info, gene_list, puck@nUMI[i], beads[i,],
                                           class_df = class_df, constrain = constrain, MIN.CHANGE = MIN.CHANGE,
                                           CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = DOUBLET_THRESHOLD)
    }
  }
  return(results)
}

process_beads_multi <- function(cell_type_info, gene_list, puck, class_df = NULL, constrain = T,
                                MAX_CORES = 8, MIN.CHANGE = 0.001, MAX.TYPES = 4, CONFIDENCE_THRESHOLD = 10, DOUBLET_THRESHOLD = 25) {
  beads = t(as.matrix(puck@counts[gene_list,]))
  if(MAX_CORES > 1 && .Platform$OS.type == "unix" && dim(beads)[1] > 1) {
    results <- fork_lapply(dim(beads)[1], MAX_CORES, function(i)
      process_bead_multi(cell_type_info, gene_list, puck@nUMI[i], beads[i,], class_df = class_df,
                         constrain = constrain, MIN.CHANGE = MIN.CHANGE, MAX.TYPES = MAX.TYPES,
                         CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = DOUBLET_THRESHOLD))
  } else if(MAX_CORES > 1) {
    numCores = parallel::detectCores();
    if(parallel::detectCores() > MAX_CORES)
      numCores <- MAX_CORES
    cl <- parallel::makeCluster(numCores,setup_strategy = "sequential",outfile="")
    doParallel::registerDoParallel(cl)
    environ = c('decompose_full','decompose_sparse','solveIRWLS.weights','solveOLS','solveWLS','Q_mat','X_vals','K_val', 'SQ_mat')
    results <- foreach::foreach(i = 1:(dim(beads)[1]), .export = environ) %dopar% { #.packages = c("quadprog"),
      #if(i %% 100 == 0)
      #  cat(paste0("Finished sample: ",i,"\n"), file=out_file, append=TRUE)
      assign("Q_mat",Q_mat, envir = globalenv()); assign("X_vals",X_vals, envir = globalenv())
      assign("K_val",K_val, envir = globalenv()); assign("SQ_mat",SQ_mat, envir = globalenv());
      result = process_bead_multi(cell_type_info, gene_list, puck@nUMI[i], beads[i,],
                                  class_df = class_df, constrain = constrain, MIN.CHANGE = MIN.CHANGE, MAX.TYPES = MAX.TYPES,
                                  CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = DOUBLET_THRESHOLD)
      result
    }
    parallel::stopCluster(cl)
  } else {
    #not parallel
    results <- list()
    for(i in 1:(dim(beads)[1])) {
      results[[i]] <- process_bead_multi(cell_type_info, gene_list, puck@nUMI[i],
                                         beads[i,], class_df = class_df,
                                         constrain = constrain, MIN.CHANGE = MIN.CHANGE, MAX.TYPES = MAX.TYPES,
                                         CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = DOUBLET_THRESHOLD)
    }
  }
  return(results)
}

#' Runs the RCTD algorithm
#'
#' If in doublet mode, fits at most two cell types per pixel. It classifies each pixel as 'singlet' or 'doublet' and searches for the cell types
#' on the pixel. If in full mode, can fit any number of cell types on each pixel. In multi mode, cell types are added using a greedy algorithm,
#' up to a fixed number.
#'
#' @param RCTD an \code{\linkS4class{RCTD}} object after running the \code{\link{choose_sigma_c}} function.
#' @param doublet_mode \code{character string}, either "doublet", "multi", or "full" on which mode to run RCTD. Please see above description.
#' @return an \code{\linkS4class{RCTD}} object containing the results of the RCTD algorithm.
#' @export
fitPixels <- function(RCTD, doublet_mode = "doublet") {
  RCTD@internal_vars$cell_types_assigned <- TRUE
  RCTD@config$RCTDmode <- doublet_mode
  set_likelihood_vars(RCTD@internal_vars$Q_mat, RCTD@internal_vars$X_vals)
  cell_type_info <- RCTD@cell_type_info$renorm
  if(doublet_mode == "doublet") {
    results = process_beads_batch(cell_type_info, RCTD@internal_vars$gene_list_reg, RCTD@spatialRNA, class_df = RCTD@internal_vars$class_df,
                                  constrain = F, MAX_CORES = RCTD@config$max_cores, MIN.CHANGE = RCTD@config$MIN_CHANGE_REG,
                                  CONFIDENCE_THRESHOLD = RCTD@config$CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = RCTD@config$DOUBLET_THRESHOLD)
    return(gather_results(RCTD, results))
  } else if(doublet_mode == "full") {
    beads = t(as.matrix(RCTD@spatialRNA@counts[RCTD@internal_vars$gene_list_reg,]))
    results = decompose_batch(RCTD@spatialRNA@nUMI, cell_type_info[[1]], beads, RCTD@internal_vars$gene_list_reg, constrain = F,
                              max_cores = RCTD@config$max_cores, MIN.CHANGE = RCTD@config$MIN_CHANGE_REG, as_matrix = TRUE)
    weights = results$weights
    rownames(weights) = colnames(RCTD@spatialRNA@counts); colnames(weights) = RCTD@cell_type_info$renorm[[2]];
    weights = Matrix(weights)
    RCTD@results <- list(weights = weights)
    return(RCTD)
  } else if(doublet_mode == "multi") {
    RCTD@results = process_beads_multi(cell_type_info, RCTD@internal_vars$gene_list_reg, RCTD@spatialRNA, class_df = RCTD@internal_vars$class_df,
                                  constrain = F, MAX_CORES = RCTD@config$max_cores,
                                  MIN.CHANGE = RCTD@config$MIN_CHANGE_REG, MAX.TYPES = RCTD@config$MAX_MULTI_TYPES,
                                  CONFIDENCE_THRESHOLD = RCTD@config$CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD = RCTD@config$DOUBLET_THRESHOLD)
    return(RCTD)
  } else {
    stop(paste0("fitPixels: doublet_mode=",doublet_mode, " is not a valid choice. Please set doublet_mode=doublet, multi, or full."))
  }
}

# Applies `one` to 1..n on forked workers (unix only): the likelihood tables and profile matrices
# are inherited copy-on-write instead of being serialized to a cluster, and beads are handed out in
# small chunks (dynamic scheduling) so one slow bead does not leave workers idle.
fork_lapply <- function(n, max_cores, one, chunk_cap = 200L) {
  numCores <- max(1L, min(max_cores, parallel::detectCores(), n))
  chunk_size <- max(1L, min(chunk_cap, ceiling(n / (numCores * 8))))
  chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  out <- parallel::mclapply(chunks, function(ix) lapply(ix, one), mc.cores = numCores, mc.preschedule = FALSE)
  failed <- vapply(out, function(x) inherits(x, "try-error") || is.null(x), logical(1))
  if(any(failed))
    stop("fork_lapply: ", sum(failed), " worker chunk(s) failed: ",
         paste(unique(unlist(lapply(out[failed], as.character))), collapse = "; "))
  unlist(out, recursive = FALSE, use.names = FALSE)
}

decompose_batch <- function(nUMI, cell_type_means, beads, gene_list, constrain = T, OLS = F, max_cores = 8, MIN.CHANGE = 0.001,
                            as_matrix = FALSE) {
  res <- decompose_batch_list(nUMI, cell_type_means, beads, gene_list, constrain = constrain, OLS = OLS,
                              max_cores = max_cores, MIN.CHANGE = MIN.CHANGE, as_matrix = as_matrix)
  if(as_matrix && is.list(res) && is.null(res$weights))
    res <- list(weights = do.call(rbind, lapply(res, function(r) r$weights)),
                converged = vapply(res, function(r) r$converged, logical(1)))
  res
}

decompose_batch_list <- function(nUMI, cell_type_means, beads, gene_list, constrain = T, OLS = F, max_cores = 8, MIN.CHANGE = 0.001,
                                 as_matrix = FALSE) {
  #out_file = "logs/decompose_batch_log.txt"
  #if (file.exists(out_file))
  #  file.remove(out_file)
  # Everything that does not depend on the bead is computed once: the profile matrix restricted to
  # gene_list, and the pairwise-product matrix S_mat. For bead i, S = S_base * nUMI[i] and
  # S_mat = S_mat_base * nUMI[i]^2.
  S_base <- data.matrix(cell_type_means[gene_list,])
  S_mat_base <- if(OLS) NULL else build_S_mat(S_base)
  n_beads <- dim(beads)[1]
  decompose_one <- function(i) {
    decompose_full(S_base * nUMI[i], nUMI[i], beads[i,], constrain = constrain, OLS = OLS, MIN_CHANGE = MIN.CHANGE,
                   S_mat = if(is.null(S_mat_base)) NULL else S_mat_base * nUMI[i]^2)
  }
  decompose_chunk <- function(ix) lapply(ix, decompose_one)
  if(!OLS && !constrain && n_beads > 0 && isTRUE(getOption("fastspacexr.use_cpp", TRUE))) {
    # Compiled solver (OpenMP over beads). Beads it cannot solve (non-finite values, eigen/QP failure)
    # are re-run below with the R implementation.
    beads_num <- beads; storage.mode(beads_num) <- "double"
    nUMI_num <- as.numeric(nUMI)
    cpp <- tryCatch({
                      parts <- cpp_shard_apply(n_beads, max_cores, function(ix, threads)
                        irwls_batch_cpp(S_base, nUMI_num[ix], beads_num[ix, , drop = FALSE], Q_mat, SQ_mat, X_vals,
                                        K_val, MIN.CHANGE, 50L, threads))
                      list(weights = do.call(rbind, lapply(parts, `[[`, "weights")),
                           converged = unlist(lapply(parts, `[[`, "converged"), use.names = FALSE),
                           status = unlist(lapply(parts, `[[`, "status"), use.names = FALSE))
                    },
                    error = function(e) {
                      warning("decompose_batch: compiled solver rejected its inputs (", conditionMessage(e),
                              "); using the R implementation")
                      NULL
                    })
    if(!is.null(cpp)) {
      W <- cpp$weights; colnames(W) <- colnames(S_base); conv <- cpp$converged
      bad <- which(cpp$status != 0)
      if(length(bad) > 0) {
        warning("decompose_batch: compiled solver failed for ", length(bad), " bead(s); re-running them in R")
        # beads with invalid data (NaN / negative counts, ...) fail in R too; they get NA weights instead of
        # aborting the whole batch
        fb <- lapply(bad, function(i) tryCatch(decompose_one(i),
                                               error = function(e) list(weights = rep(NA_real_, ncol(W)), converged = FALSE)))
        for(k in seq_along(bad)) { W[bad[k], ] <- fb[[k]]$weights; conv[bad[k]] <- fb[[k]]$converged }
      }
      if(as_matrix)
        return(list(weights = W, converged = conv))
      return(lapply(seq_len(n_beads), function(i) list(weights = W[i, ], converged = conv[i])))
    }
  }
  if(max_cores > 1 && n_beads > 1 && .Platform$OS.type == "unix") {
    # Forked workers inherit Q_mat, X_vals, K_val, SQ_mat and the profile matrices copy-on-write, so
    # nothing is serialized and no cluster has to be started. Beads are handed out in many small
    # chunks (dynamic scheduling) rather than one task per bead.
    weights <- fork_lapply(n_beads, max_cores, decompose_one)
  } else if(max_cores > 1) {
    # no fork on this platform: one task per worker, not per bead
    numCores <- min(max_cores, parallel::detectCores(), n_beads)
    cl <- parallel::makeCluster(numCores, setup_strategy = "sequential", outfile = "")
    doParallel::registerDoParallel(cl)
    environ = c('decompose_full','solveIRWLS.weights','solveOLS','solveWLS', 'Q_mat', 'K_val','X_vals', 'SQ_mat',
                'build_S_mat', 'get_hess_index', 'get_der_fast', 'calc_Q_all', 'get_d1_d2', '.hess_index_cache')
    chunks <- split(seq_len(n_beads), cut(seq_len(n_beads), numCores, labels = FALSE))
    out <- foreach::foreach(ix = chunks, .packages = c("quadprog"), .export = environ) %dopar% {
      assign("Q_mat",Q_mat, envir = globalenv()); assign("X_vals",X_vals, envir = globalenv())
      assign("K_val",K_val, envir = globalenv()); assign("SQ_mat",SQ_mat, envir = globalenv());
      decompose_chunk(ix)
    }
    parallel::stopCluster(cl)
    weights <- unlist(out, recursive = FALSE, use.names = FALSE)
  } else {
    weights <- decompose_chunk(seq_len(n_beads))
  }
  return(weights)
}



# Doublet search for all beads in compiled code (see doublet_batch_cpp). Returns the same list of per-bead
# results as process_beads_batch's R path, so gather_results() is unchanged. Beads the compiled code cannot
# handle (status != 0) are computed with `fallback(i)` (process_bead_doublet).
process_beads_doublet_cpp <- function(cell_type_info, gene_list, nUMI, beads, class_df, MIN.CHANGE,
                                      CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD, max_cores, fallback) {
  S_base <- data.matrix(cell_type_info[[1]][gene_list,])
  type_names <- cell_type_info[[2]]
  cls_id <- as.integer(factor(as.character(class_df[type_names, "class"])))
  beads_num <- beads; storage.mode(beads_num) <- "double"
  nUMI_num <- as.numeric(nUMI)
  cpp <- tryCatch({
                    parts <- cpp_shard_apply(dim(beads)[1], max_cores, function(ix, threads)
                      doublet_batch_cpp(S_base, nUMI_num[ix], beads_num[ix, , drop = FALSE], Q_mat, SQ_mat, X_vals, K_val,
                                        MIN.CHANGE, cls_id, CONFIDENCE_THRESHOLD, DOUBLET_THRESHOLD, threads))
                    merged <- list()
                    for(f in c("all_weights", "doublet_weights"))
                      merged[[f]] <- do.call(rbind, lapply(parts, `[[`, f))
                    for(f in c("first", "second", "spot", "min_score", "singlet_score", "conv_all", "conv_doublet",
                               "first_class", "second_class", "status"))
                      merged[[f]] <- unlist(lapply(parts, `[[`, f), use.names = FALSE)
                    for(f in c("cand", "score_mat", "singlet_scores"))
                      merged[[f]] <- unlist(lapply(parts, `[[`, f), recursive = FALSE, use.names = FALSE)
                    merged
                  },
                  error = function(e) {
                    warning("process_beads_batch: compiled doublet search rejected its inputs (", conditionMessage(e),
                            "); using the R implementation")
                    NULL
                  })
  n <- dim(beads)[1]
  if(is.null(cpp)) {
    if(max_cores > 1 && .Platform$OS.type == "unix" && n > 1)
      return(fork_lapply(n, max_cores, fallback))
    return(lapply(seq_len(n), fallback))
  }
  spot_levels <- c("reject", "singlet", "doublet_certain", "doublet_uncertain")
  bad <- which(cpp$status != 0)
  if(length(bad) > 0) {
    warning("process_beads_batch: compiled doublet search failed for ", length(bad), " bead(s); re-running them in R")
    fb <- if(max_cores > 1 && .Platform$OS.type == "unix" && length(bad) > 1)
      fork_lapply(length(bad), max_cores, function(k) fallback(bad[k])) else lapply(bad, fallback)
  }
  fb_pos <- integer(n); fb_pos[bad] <- seq_along(bad)
  lapply(seq_len(n), function(i) {
    if(fb_pos[i] > 0) return(fb[[fb_pos[i]]])
    all_w <- cpp$all_weights[i, ]; names(all_w) <- type_names
    first <- type_names[cpp$first[i]]; second <- type_names[cpp$second[i]]
    dw <- cpp$doublet_weights[i, ]; names(dw) <- c(first, second)
    cn <- type_names[cpp$cand[[i]]]
    sm <- cpp$score_mat[[i]]; dimnames(sm) <- list(cn, cn)
    ss <- as.vector(cpp$singlet_scores[[i]]); names(ss) <- cn
    list(all_weights = all_w, spot_class = factor(spot_levels[cpp$spot[i]], spot_levels), first_type = first,
         second_type = second, doublet_weights = dw, min_score = cpp$min_score[i], singlet_score = cpp$singlet_score[i],
         conv_all = cpp$conv_all[i], conv_doublet = cpp$conv_doublet[i], score_mat = as(sm, "CsparseMatrix"), singlet_scores = ss,
         first_class = cpp$first_class[i], second_class = cpp$second_class[i])
  })
}

# Runs fn(ix, threads) over shards of 1..n and returns the list of shard results in order.
#
# On unix the shards are processed by separate forked worker processes, each running the compiled solver
# single-threaded. Independent processes share no locks, which scales close to linearly with the number of
# physical cores; running the same solver on many OpenMP threads inside one process did not scale (it got
# slower beyond ~4 threads). Shards are small (about n / (4 * workers) beads) and handed out dynamically
# so that unevenly expensive beads do not leave workers idle. The inputs are inherited copy-on-write.
# Elsewhere (no fork) the whole batch runs in this process with `max_cores` OpenMP threads.
# Set options(fastspacexr.parallel = "thread") to force in-process threading on unix as well.
cpp_shard_apply <- function(n, max_cores, fn) {
  max_cores <- as.integer(max(1, max_cores))
  use_proc <- .Platform$OS.type == "unix" && max_cores > 1 && n > 1 && !identical(getOption("fastspacexr.parallel"), "thread")
  if(!use_proc)
    return(list(fn(seq_len(n), max_cores)))
  workers <- max(1L, min(max_cores, parallel::detectCores(), n))
  shard <- max(1L, ceiling(n / (workers * 4)))
  shards <- split(seq_len(n), ceiling(seq_len(n) / shard))
  parts <- parallel::mclapply(shards, fn, 1L, mc.cores = workers, mc.preschedule = FALSE)
  failed <- vapply(parts, function(x) inherits(x, "try-error") || is.null(x), logical(1))
  if(any(failed))
    stop("cpp_shard_apply: ", sum(failed), " worker shard(s) failed: ",
         paste(unique(unlist(lapply(parts[failed], as.character))), collapse = "; "))
  unname(parts)
}
