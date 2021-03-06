# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' Logistic regression based on MLE
#'
#' @param X a n x p matrix of regressor
#' @param y a n-vector of response
#' @export
logistic_mle <- function(X, y) {
    .Call('_fpl_logistic_mle', PACKAGE = 'fpl', X, y)
}

#'Cross-validation for logistic regression with l2-norm error
#'
#'@param X a n x p matrix of regressor
#'@param y a n-vector of response
#'@param seed an integer for setting the seed (reproducibility)
#'@param K number of splits; 10 by default
#'@param M number of repetitions; 10 by default
#'@export
cross_validation_logistic_l2 <- function(X, y, seed, K = 10L, M = 10L) {
    .Call('_fpl_cross_validation_logistic_l2', PACKAGE = 'fpl', X, y, seed, K, M)
}

#'Cross-validation for logistic regression with counting error
#'
#'@param X a n x p matrix of regressor
#'@param y a n-vector of response
#'@param seed an integer for setting the seed (reproducibility)
#'@param K number of splits; 10 by default
#'@param M number of repetitions; 10 by default
#'@export
cross_validation_logistic_count <- function(X, y, seed, K = 10L, M = 10L) {
    .Call('_fpl_cross_validation_logistic_count', PACKAGE = 'fpl', X, y, seed, K, M)
}

#'Fast Panning for logistic
#'
#'@param X a n x p matrix of regressor
#'@param y a n-vector of response
#'@param var_mat a matrix of indices for subsetting X
#'@param seed an integer for setting the seed (reproducibility)
#'@param ncores number of cores for parallel computing (OpenMP)
#'@param K number of splits; 10 by default
#'@param M number of repetitions; 10 by default
#'@export
fpl <- function(X, y, var_mat, seed, ncores, K = 10L, M = 10L) {
    .Call('_fpl_fpl', PACKAGE = 'fpl', X, y, var_mat, seed, ncores, K, M)
}

