# pkgList <- c("RcppEigen","RcppNumerical","Rcpp")
# lapply(pkgList, require, character.only=T)
# Sys.setenv("PKG_CXXFLAGS"="-std=c++14")
# Sys.setenv("PKG_CXXFLAGS"="-Wno-ignored-attributes")
# sourceCpp("./test/estimation_logistic.cpp")


n <- 1e2
p <- 5
beta <- seq_len(p)
X <- matrix(rnorm(n*p,sd=4/sqrt(n)),nc=p)
set.seed(123)
u <- runif(n)
y <- ifelse(X%*%beta + log(u) - log(1-u)>=0,1,0)
var_mat <- matrix(1:p,nr=1,nc=p)
fpl(X,y,var_mat,123,2)

check(y,X,beta)
optim_mle_logistic(y,X,beta+rnorm(p,sd=.05))
optim_mle_logistic(y,X,beta)

grad <- vector("numeric",p)
eps <- 1e-8
for(i in seq_len(p)){
  z <- rep(0,p)
  z[i] <- eps
  grad[i] <- (check(y,X,beta+z)$of - check(y,X,beta-z)$of) / eps / 0.2e1
}
check(y,X,beta)$grad

cross_validation_logistic_l2(X,y,beta,124)
cross_validation_logistic_count(X,y,124)
