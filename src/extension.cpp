// [[Rcpp::depends(RcppEigen,RcppNumerical)]]
#include <RcppEigen.h>
#include <RcppNumerical.h>
#include <numeric>
#include <math.h>
#include <random>
#include <algorithm>
#include <vector>

// Add a flag to enable OpenMP at compile time
// [[Rcpp::plugins(openmp)]]

// Protect against compilers without OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

// --------------
// Misc
// --------------
struct Sigmoid {
  Sigmoid(){}
  const double operator()(const double& x) const {return 0.1e1 / (0.1e1 + std::exp(-x));}
};

struct Indicator {
  Indicator(){}
  const double operator()(const double &t) const {return (t <= 0.5) ? 0.0 : 1.0;}
};

// https://stackoverflow.com/questions/25921706/creating-a-vector-of-indices-of-a-sorted-vector
Eigen::VectorXi sort_order(Eigen::VectorXd& x){
  std::vector<int> ind(x.size());
  std::size_t n(0);
  std::generate(std::begin(ind), std::end(ind), [&]{ return n++; });
  std::sort(std::begin(ind), std::end(ind), [&](int i1, int i2) { return x[i1] > x[i2]; } );
  int *ptr_data = &ind[0];
  return Eigen::Map<Eigen::VectorXi>(ptr_data, ind.size());
}

// adapted from R pROC::auc
// response is supposed to be 0-1 vector
double auc(
    Eigen::VectorXd& response,
    Eigen::VectorXd& predictor
){
  unsigned int n = response.size();
  double ncases = response.sum();
  double ncontrols = n - ncases;
  
  // order predictor and response
  Eigen::VectorXi predictor_order = sort_order(predictor);
  // Eigen::VectorXd predictor_sorted = predictor(predictor_order);
  Eigen::VectorXd response_sorted = predictor_order.unaryExpr(response);
  
  // compute specificity and sensibility
  Eigen::VectorXd sp(n+1),se(n+1);
  sp(n) = 1.0;
  se(n) = 0.0;
  
  double j(0.0),k(0.0);
  for(unsigned int i=0; i<n; ++i){
    response_sorted(i)==1.0 ? j++ : k++;
    sp(n-i-1) = 1.0 - k / ncontrols;
    se(n-i-1) = j / ncases;
  }
  
  // compute AUC
  Eigen::ArrayXd diffs_x = sp.tail(n) - sp.head(n);
  Eigen::ArrayXd means_vert = 0.5 * se.tail(n) + 0.5 * se.head(n);
  
  return (diffs_x * means_vert).sum();
}

// --------------
// Logistic regression
// --------------
class mle_logistic: public Numer::MFuncGrad
{
private:
  const Eigen::VectorXd y;
  const Eigen::MatrixXd X;
  const unsigned int n = y.size();
  const unsigned int p = X.cols();

public:
  mle_logistic(const Eigen::VectorXd& y_,const Eigen::MatrixXd& X_) : y(y_), X(X_) {}
  double f_grad(Numer::Constvec& beta, Numer::Refvec grad);
};

double mle_logistic::f_grad(
    Numer::Constvec& beta,
    Numer::Refvec grad
){
  // data storage
  Eigen::ArrayXd sig(n);
  Eigen::VectorXd v(n);
  Eigen::MatrixXd x1(n,p+1);

  // pre-computation
  x1.rightCols(p) = X;
  x1.col(0) = Eigen::VectorXd::Constant(n,1.0);
  v = x1 * beta;
  sig = v.unaryExpr(Sigmoid());

  // computation
  // objective function
  const double f = -(0.1e1 - sig).log().sum() / n - v.dot(y) / n;

  // gradient
  grad = X.transpose() * (sig.matrix() - y) / n;
  return f;
}

//' Logistic regression based on MLE
//'
//' @param X a n x p matrix of regressor
//' @param y a n-vector of response
//' @export
// [[Rcpp::export]]
Eigen::VectorXd logistic_mle(
  Eigen::MatrixXd& X,
  Eigen::VectorXd& y
){
  // Regress
  unsigned int p(X.cols());
  double fopt;
  Eigen::VectorXd beta(p);
  beta.setZero();
  double prop = y.mean();
  beta(0) = std::log(prop) - std::log(1.0 - prop);
  mle_logistic f(y,X);
  Numer::optim_lbfgs(f,beta,fopt);
  return beta;
}

//'Test error for logistic regression with counting error
//'
//'@param X_train a n x p matrix of training regressor
//'@param y_train a n-vector of training response
//'@param X_test a m x p matrix of test regressor
//'@param y_test a m-vector of test response
//'@export
// [[Rcpp::export]]
double test_logistic_count(
    Eigen::MatrixXd& X_train,
    Eigen::VectorXd& y_train,
    Eigen::MatrixXd& X_test,
    Eigen::VectorXd& y_test
){
  double err;
  unsigned int p = X_train.cols();
  unsigned int n_test = y_test.size();
  Eigen::ArrayXd pred(n_test),diff_pred(n_test);
  
  // Regress
  Eigen::VectorXd beta(p);
  beta = logistic_mle(X_train,y_train);
  
  // Get the predictions
  pred = (X_test * beta).unaryExpr(Sigmoid());
  
  // Classification error
  diff_pred = y_test.array() - pred.unaryExpr(Indicator());
  err = diff_pred.abs().matrix().sum() / n_test;
  return err;
}

//'Test error for logistic regression with AUC performance
//'
//'@param X_train a n x p matrix of training regressor
//'@param y_train a n-vector of training response
//'@param X_test a m x p matrix of test regressor
//'@param y_test a m-vector of test response
//'@export
// [[Rcpp::export]]
double test_logistic_auc(
    Eigen::MatrixXd& X_train,
    Eigen::VectorXd& y_train,
    Eigen::MatrixXd& X_test,
    Eigen::VectorXd& y_test
){
  double err;
  unsigned int p = X_train.cols();
  unsigned int n_test = y_test.size();
  Eigen::VectorXd pred(n_test);
  
  // Regress
  Eigen::VectorXd beta(p);
  beta = logistic_mle(X_train,y_train);
  
  // Get the predictions
  pred = (X_test * beta).unaryExpr(Sigmoid());
  
  // Classification error
  err = auc(y_test, pred);
  return err;
}

// --------------
// Cross-validation for logistic regression
// --------------

//'Cross-validation for logistic regression with l2-norm error
//'
//'@param X a n x p matrix of regressor
//'@param y a n-vector of response
//'@param seed an integer for setting the seed (reproducibility)
//'@param K number of splits; 10 by default
//'@param M number of repetitions; 10 by default
//'@export
// [[Rcpp::export]]
double cross_validation_logistic_l2(
    Eigen::MatrixXd& X,
    Eigen::VectorXd& y,
    unsigned int seed,
    unsigned int K = 10,
    unsigned int M = 10
){
  // Storage
  double err(0.0);
  unsigned int n = X.rows();
  unsigned int p = X.cols();
  unsigned int nn = n;
  unsigned int n_train, n_test;
  std::vector<int> ivec(n);
  std::iota(ivec.begin(),ivec.end(),0);
  std::vector<int> n_fold(K);

  std::mt19937_64 engine(seed);  // Mersenne twister random number engine

  for(unsigned int i = 0; i<K; i++){
    n_fold[i] = std::ceil(nn/(K-i));
    nn -= n_fold[i];
  }

  for(unsigned int m = 0; m < M; m++){
    // Shuffle the index
    std::shuffle(ivec.begin(),ivec.end(),engine);

    // K-fold CV on logitstic classification
    for(unsigned int k = 0; k < K; k++){
      // Seperate training/test sets
      n_train = n-n_fold[k];
      n_test = n_fold[k];
      Eigen::MatrixXd X_train(n_train,p),X_test(n_test,p);
      Eigen::VectorXd y_train(n_train),y_test(n_test),pred(n_test);

      unsigned int ii(0),jj(0),ind;

      for(unsigned int i = k+1; i < n+k+1; i++){
        if(i%K == 0){
          ind = ivec[i-k-1];
          X_test.row(ii) = X.row(ind);
          y_test(ii) = y(ind);
          ii++;
        }else{
          ind = ivec[i-k-1];
          X_train.row(jj) = X.row(ind);
          y_train(jj) = y(ind);
          jj++;
        }
      }

      // Regress
      Eigen::VectorXd beta(p);
      beta = logistic_mle(X_train,y_train);
     
      // Get the predictions
      pred = (X_test * beta).unaryExpr(Sigmoid());

      // Classification error
      y_test -= pred;
      ii++;
      err += y_test.dot(y_test) / ii;
    }
  }

  return err / K / M;
}

//'Cross-validation for logistic regression with counting error
//'
//'@param X a n x p matrix of regressor
//'@param y a n-vector of response
//'@param seed an integer for setting the seed (reproducibility)
//'@param K number of splits; 10 by default
//'@param M number of repetitions; 10 by default
//'@export
// [[Rcpp::export]]
double cross_validation_logistic_count(
    Eigen::MatrixXd& X,
    Eigen::VectorXd& y,
    unsigned int seed,
    unsigned int K = 10,
    unsigned int M = 10
){
  // Storage
  double err(0.0);
  unsigned int n = X.rows();
  unsigned int p = X.cols();
  unsigned int nn = n;
  unsigned int n_train, n_test;
  std::vector<int> ivec(n);
  std::iota(ivec.begin(),ivec.end(),0);
  std::vector<int> n_fold(K);

  std::mt19937_64 engine(seed);  // Mersenne twister random number engine

  for(unsigned int i = 0; i<K; i++){
    n_fold[i] = std::ceil(nn/(K-i));
    nn -= n_fold[i];
  }

  for(unsigned int m = 0; m < M; m++){
    // Shuffle the index
    std::shuffle(ivec.begin(),ivec.end(),engine);

    // K-fold CV on logitstic classification
    for(unsigned int k = 0; k < K; k++){
      // Seperate training/test sets
      n_train = n-n_fold[k];
      n_test = n_fold[k];
      Eigen::MatrixXd X_train(n_train,p),X_test(n_test,p);
      Eigen::VectorXd y_train(n_train);
      Eigen::ArrayXd y_test(n_test),pred(n_test);

      unsigned int ii(0),jj(0),ind;

      for(unsigned int i = k+1; i < n+k+1; i++){
        if(i%K == 0){
          ind = ivec[i-k-1];
          X_test.row(ii) = X.row(ind);
          y_test(ii) = y(ind);
          ii++;
        }else{
          ind = ivec[i-k-1];
          X_train.row(jj) = X.row(ind);
          y_train(jj) = y(ind);
          jj++;
        }
      }

      // Regress
      Eigen::VectorXd beta(p);
      beta = logistic_mle(X_train,y_train);

      // Get the predictions
      pred = (X_test * beta).unaryExpr(Sigmoid());

      // Classification error
      y_test -= pred.unaryExpr(Indicator());
      ii++;
      err += y_test.abs().matrix().sum() / ii;
    }
  }

  return err / K / M;
}

//'Cross-validation for logistic regression with AUC
//'
//'@param X a n x p matrix of regressor
//'@param y a n-vector of response
//'@param seed an integer for setting the seed (reproducibility)
//'@param K number of splits; 10 by default
//'@param M number of repetitions; 10 by default
//'@export
// [[Rcpp::export]]
double cross_validation_logistic_auc(
    Eigen::MatrixXd& X,
    Eigen::VectorXd& y,
    unsigned int seed,
    unsigned int K = 10,
    unsigned int M = 10
){
  // Storage
  double err(0.0);
  unsigned int n = X.rows();
  unsigned int p = X.cols();
  unsigned int nn = n;
  unsigned int n_train, n_test;
  std::vector<int> ivec(n);
  std::iota(ivec.begin(),ivec.end(),0);
  std::vector<int> n_fold(K);
  
  std::mt19937_64 engine(seed);  // Mersenne twister random number engine
  
  for(unsigned int i = 0; i<K; i++){
    n_fold[i] = std::ceil(nn/(K-i));
    nn -= n_fold[i];
  }
  
  for(unsigned int m = 0; m < M; m++){
    // Shuffle the index
    std::shuffle(ivec.begin(),ivec.end(),engine);
    
    // K-fold CV on logitstic classification
    for(unsigned int k = 0; k < K; k++){
      // Seperate training/test sets
      n_train = n-n_fold[k];
      n_test = n_fold[k];
      Eigen::MatrixXd X_train(n_train,p),X_test(n_test,p);
      Eigen::VectorXd y_train(n_train);
      Eigen::VectorXd y_test(n_test),pred(n_test);
      
      unsigned int ii(0),jj(0),ind;
      
      for(unsigned int i = k+1; i < n+k+1; i++){
        if(i%K == 0){
          ind = ivec[i-k-1];
          X_test.row(ii) = X.row(ind);
          y_test(ii) = y(ind);
          ii++;
        }else{
          ind = ivec[i-k-1];
          X_train.row(jj) = X.row(ind);
          y_train(jj) = y(ind);
          jj++;
        }
      }
      
      // Regress
      Eigen::VectorXd beta(p);
      beta = logistic_mle(X_train,y_train);
      
      // Get the predictions
      pred = (X_test * beta).unaryExpr(Sigmoid());
      
      // Classification error
      err += auc(y_test, pred);
    }
  }
  
  return err / K / M;
}


// --------------
// Model exploration: fast panning for logistic
// --------------
//'Fast Panning for logistic
//'
//'@param X a n x p matrix of regressor
//'@param y a n-vector of response
//'@param var_mat a matrix of indices for subsetting X
//'@param seed an integer for setting the seed (reproducibility)
//'@param ncores number of cores for parallel computing (OpenMP)
//'@param K number of splits; 10 by default
//'@param M number of repetitions; 10 by default
//'@export
// [[Rcpp::export]]
Eigen::VectorXd fpl(
    Eigen::MatrixXd& X,
    Eigen::VectorXd& y,
    Eigen::MatrixXd& var_mat,
    unsigned int seed,
    unsigned int ncores,
    unsigned int K = 10,
    unsigned int M = 10
){
  unsigned int k = var_mat.cols();
  unsigned int p = var_mat.rows();
  unsigned int n = y.size();
  Eigen::VectorXd cv_errors(k);
  
  #pragma omp parallel for num_threads(ncores)
  for(unsigned int i=0; i<k; ++i){
    unsigned int se = seed + i;
    Eigen::MatrixXd x(n,p);
    for(unsigned int j(0);j<p;++j){
      unsigned int ind = var_mat(j,i);
      x.col(j) = X.col(ind);
      }
    cv_errors(i) = cross_validation_logistic_l2(x,y,se,K,M);
  }
  
  return cv_errors;
}
