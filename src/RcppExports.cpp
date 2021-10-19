// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// logistic_mle
Eigen::VectorXd logistic_mle(Eigen::MatrixXd& X, Eigen::VectorXd& y);
RcppExport SEXP _fpl_logistic_mle(SEXP XSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(logistic_mle(X, y));
    return rcpp_result_gen;
END_RCPP
}
// test_logistic_count
double test_logistic_count(Eigen::MatrixXd& X_train, Eigen::VectorXd& y_train, Eigen::MatrixXd& X_test, Eigen::VectorXd& y_test);
RcppExport SEXP _fpl_test_logistic_count(SEXP X_trainSEXP, SEXP y_trainSEXP, SEXP X_testSEXP, SEXP y_testSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X_train(X_trainSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y_train(y_trainSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y_test(y_testSEXP);
    rcpp_result_gen = Rcpp::wrap(test_logistic_count(X_train, y_train, X_test, y_test));
    return rcpp_result_gen;
END_RCPP
}
// cross_validation_logistic_l2
double cross_validation_logistic_l2(Eigen::MatrixXd& X, Eigen::VectorXd& y, unsigned int seed, unsigned int K, unsigned int M);
RcppExport SEXP _fpl_cross_validation_logistic_l2(SEXP XSEXP, SEXP ySEXP, SEXP seedSEXP, SEXP KSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type K(KSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(cross_validation_logistic_l2(X, y, seed, K, M));
    return rcpp_result_gen;
END_RCPP
}
// cross_validation_logistic_count
double cross_validation_logistic_count(Eigen::MatrixXd& X, Eigen::VectorXd& y, unsigned int seed, unsigned int K, unsigned int M);
RcppExport SEXP _fpl_cross_validation_logistic_count(SEXP XSEXP, SEXP ySEXP, SEXP seedSEXP, SEXP KSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type K(KSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(cross_validation_logistic_count(X, y, seed, K, M));
    return rcpp_result_gen;
END_RCPP
}
// cross_validation_logistic_auc
double cross_validation_logistic_auc(Eigen::MatrixXd& X, Eigen::VectorXd& y, unsigned int seed, unsigned int K, unsigned int M);
RcppExport SEXP _fpl_cross_validation_logistic_auc(SEXP XSEXP, SEXP ySEXP, SEXP seedSEXP, SEXP KSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type K(KSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(cross_validation_logistic_auc(X, y, seed, K, M));
    return rcpp_result_gen;
END_RCPP
}
// fpl
Eigen::VectorXd fpl(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::MatrixXd& var_mat, unsigned int seed, unsigned int ncores, unsigned int K, unsigned int M);
RcppExport SEXP _fpl_fpl(SEXP XSEXP, SEXP ySEXP, SEXP var_matSEXP, SEXP seedSEXP, SEXP ncoresSEXP, SEXP KSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type var_mat(var_matSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type ncores(ncoresSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type K(KSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(fpl(X, y, var_mat, seed, ncores, K, M));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_fpl_logistic_mle", (DL_FUNC) &_fpl_logistic_mle, 2},
    {"_fpl_test_logistic_count", (DL_FUNC) &_fpl_test_logistic_count, 4},
    {"_fpl_cross_validation_logistic_l2", (DL_FUNC) &_fpl_cross_validation_logistic_l2, 5},
    {"_fpl_cross_validation_logistic_count", (DL_FUNC) &_fpl_cross_validation_logistic_count, 5},
    {"_fpl_cross_validation_logistic_auc", (DL_FUNC) &_fpl_cross_validation_logistic_auc, 5},
    {"_fpl_fpl", (DL_FUNC) &_fpl_fpl, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_fpl(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
