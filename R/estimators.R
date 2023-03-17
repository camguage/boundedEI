
chen_estimator <- function(outcome, prob_feat_t, prob_feat_f, weight=NULL){
  #' A function that implements the Chen estimator (citation?)
  #'
  #' @param outcome A vector of outcome values
  #' @param prob_feat_t
  #' @param prob_feat_f
  #' @param weight
  #' @keywords internal
  if(is.null(weight)){
    est <- sum(outcome * prob_feat_t) / sum(prob_feat_t) - sum(outcome * prob_feat_f) / sum(prob_feat_f)
  }
  else{
    est <- sum(outcome * prob_feat_t * weight) / sum(prob_feat_t * weight) - sum(outcome * prob_feat_f * weight) / sum(prob_feat_f * weight)
  }
  return(est)
}


reg_estimator <- function(formula, data, weight=NULL, returnSE=FALSE){
  # TODO: have check that formula only has one value on RHS or that data only has two columns
  model <- lm(formula,data, weights=weights)
  coefficient <- as.numeric(coef(model)[2])
  std_err <- NULL
  if(returnSE){
    std_err <- sqrt(diag(vcov(model)))[2]
  }
  return(list(coeff=coefficient, std_err=std_err)
}

mob_by_unit <- function(total_var, protect_feat_t, protect_feat_f){
  # TODO: test that we get same results as the Python code
  max_protect_feat_f <- pmin(protect_feat_f, total_var)
  min_feat_t <- pmax(0, total_var - max_protect_feat_f)
  max_protect_feat_t <- pmin(protect_feat_t, total_var)
  min_feat_f <- pmax(0, total_var - max_protect_feat_t)
  diff_lb = sum(min_feat_t)/ sum(protect_feat_t) - sum(min_feat_f)/sum(protect_feat_f)
  diff_ub = sum(max_protect_feat_t)/ sum(protect_feat_t) - sum(max_protect_feat_f)/sum(protect_feat_f)
  return(c(lb=diff_lb, ub=diff_ub))
}

