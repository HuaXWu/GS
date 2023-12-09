
library(BGLR)
library(foreach)

# read data
DT <- read.table("chapter4/Body_Length.txt", sep = "\t", header = TRUE)
GT <- read.table("chapter4/gwas_genotype.vcf", sep="\t", header = TRUE)

# GT <- t(GT)
# convert to matrix
y <- as.matrix(DT$Body_Length)
X <- as.matrix(GT)
X <- apply(X, 2, as.numeric)
# # check NA
which(is.na(GT))
which(is.na(DT))




# Split the data into training (80%) and testing (20%) sets
num <- 1
all_corr <- c()
all_mse <- c()

set.seed(num)
num <- num+1
n <- nrow(X)
train_idx <- sample(n, 0.8 * n)
test_idx <- setdiff(1:n, train_idx)
X_train <- X[train_idx, ]
X_test <- X[test_idx, ]
y_train <- y[train_idx]
y_test <- y[test_idx]
# model= 'BRR', 'BayesA','BayesB', 'BayesC', 'BL'
model <- BGLR(y=y_train, ETA=list(list(X=X_train, model='BL', R2=1.0)), verbose=FALSE,nIter=200,burnIn=10)
y_pred=model$mu+as.vector(X_test%*%model$ETA[[1]]$b)
all_corr = append(all_corr, cor(y_pred, y_test))
all_mse = append(all_mse,  mean((y_pred - y_test)^2))

cat("correlation mean is :", mean(all_corr), "\n")
cat("correlation std is :", sd(all_corr), "\n")
cat("Mean Squared Error (MSE) mean is:", mean(all_mse), "\n")
cat("MSE std is :", sd(all_mse), "\n")

