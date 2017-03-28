#' @title sample_wwr
#' @description Weighted sampling without replacement
#' @param weights

sample_wwr <- function(weights){
  .Call("Rsample_wwr", sample(1e6, 1), as.numeric(weights))
}