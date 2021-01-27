#---------------------
## Control
#---------------------
control <- list(pmax = 15,
                m = 5e4,
                alpha = 0.05,
                seed = 98573459L,
                ncores = 16L,
                verbose = TRUE
                )

#---------------------
## Function
#---------------------
# build all possible combinations
model_combination <- function(
  id_screening,
  var_mat
){
  # Generate all combinations of var_mat and id_screening
  A <- rbind(
    matrix(rep(var_mat,length(id_screening)),nrow=nrow(var_mat)),
    rep(id_screening,each=ncol(var_mat))
  )
  
  # Remove duplicates:
  # remove same model
  A <- unique(apply(A,2,sort), MARGIN = 2)
  
  # remove same attributes
  id <- apply(A,2,anyDuplicated)>0
  if(sum(id)==0){
    return(A)
  }else{
    return(subset(A,select=!id))
  }
}

#---------------------
## Panning/SWAG
#---------------------
## Seed
set.seed(control$seed)
graine <- sample.int(1e6,control$pmax)

## Object storage
CVs <- vector("list",control$pmax)
IDs <- vector("list",control$pmax)
VarMat <- vector("list",control$pmax)
cv_alpha <- rep(NA_real_,control$pmax)

#---------------------
## SWAG
#---------------------
p <- ncol(X)
for(d in seq_len(control$pmax)){
  # Build all combinations
  if(d == 1){
    var_mat <- seq_len(p)
    dim(var_mat) <- c(1,p)
  }else{
    var_mat <- model_combination(id_screening,subset(VarMat[[d-1L]],select=IDs[[d-1]]))
  }
  
  # Reduce number of model if exceeding `m`
  if(d>1 && ncol(var_mat)>control$m){
    set.seed(graine[d]-1)
    var_mat <- var_mat[,sample.int(ncol(var_mat),control$m)]
  }
  
  # Compute CV errors
  cv_errors <- fpl(X,y,var_mat,graine[d],control$ncores)
    
  # Store results
  CVs[[d]] <- cv_errors
  VarMat[[d]] <- var_mat
  cv_alpha[d] <- quantile(cv_errors,control$alpha,na.rm=T)
  IDs[[d]] <- which(cv_errors <= cv_alpha[d])
  if(d == 1) id_screening <- IDs[[d]]
  if(control$verbose) print(paste0("Dimension explored: ",d," - CV errors at alpha: ",round(cv_alpha[d],4)))
}


