
# Generates random LHS training/testing sets in [0, 1]^2 for the Schaffer function
# Writes results to csv files

library(lhs)

d <- 2
n_pred <- 500
reps <- 20

f <- function(x) {
  z <- 4 * x - 2
  return(0.5 + (cos(sin(abs(z[,1]^2-z[,2]^2)))^2 - 0.5) / 
           (1 + 0.001*(z[,1]^2 + z[,2]^2))^2)
}

for (n in c(100, 500, 1000)) {
  for (seed in 1:reps) {
    set.seed(seed)
    x <- lhs::randomLHS(n, d)
    y <- f(x) + rnorm(n, sd = sqrt(1e-8))
  
    train <- data.frame(cbind(x, y))
    colnames(train) <- c(paste0("X", 1:d), "Y")
    write.csv(train, file = paste0("train_d", d, "_n", n, "_seed", seed, ".csv"), 
              row.names = FALSE)
    if (n == 500) {
      xx <- lhs::randomLHS(n_pred, d)
      yy <- f(xx)
      test <- data.frame(cbind(xx, yy))
      colnames(test) <- c(paste0("X", 1:d), "Y")
      write.csv(test, file = paste0("test_d", d, "_seed", seed, ".csv"), 
                row.names = FALSE)
    }
  }
}
