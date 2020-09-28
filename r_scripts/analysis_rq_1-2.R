# load the necessary packages 
library(tidyverse)
library(broom)
library(purrr)
library(dplyr)
library(dominanceanalysis)
library(sjPlot)
library(rstatix)
#remove any environment variables
rm(list=ls())

# load the functions and variables used in the analysis

scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}

fx <- function(x,l){
  p <- l*exp(-l*x)
  return(p)
}

gx <- function(x,l0,l1,alpha){
  p <- (fx(x,l0)*alpha)+(fx(x,l1)*(1-alpha))
  return(p)
} 

search_out <- data.frame(alpha = numeric(), loglik = numeric())
alpha_vals <- seq(0, 1, by = 0.001)


# Load result files into dataset
file_list <- list.files()
for (file in file_list){
  
  # if the merged dataset doesn't exist, create it
  if (!exists("dataset")){
    dataset <- read.table(file, header=TRUE, sep=",")
  }
  
  # if the merged dataset does exist, append to it
  if (exists("dataset")){
    temp_dataset <-read.csv(file, header=TRUE, sep=",")
    dataset<-rbind(dataset, temp_dataset)
    rm(temp_dataset)
  }
  
}

# assign small  ~zero values to zero values for Gamma glm fitting
library(data.table)
setDT(dataset)[ON_EQ == 0, ON_EQ := runif(.N, min=0.000000001, max=0.0000001)]

# prepare the labels for the models to make them more readable and consistent

dataset$METAMODEL <- "oth"
dataset$L1RESP <- "oth"
dataset$L2RESP <- "oth"
dataset$L2SAMPLING <- "oth"
dataset$EQ_TYPE_LABEL <- "oth"
dataset$METAMODEL <- as.character(dataset$METAMODEL)
dataset$L1RESP <- as.character(dataset$L1RESP)
dataset$L2RESP <- as.character(dataset$L2RESP)
dataset$EQ_TYPE_LABEL <- as.character(dataset$EQ_TYPE_LABEL)
dataset$L2SAMPLING <- as.character(dataset$L2SAMPLING)

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="BR|BASELINE_ONLY"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="BR|BASELINE_ONLY"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="BR|BASELINE_ONLY"] <- "BASELINE")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="BR|BASELINE_ONLY"] <- "S(1)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="BR|BASELINE_ONLY"] <- "Ql0:BR_S(1)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1BR|BASELINE_ONLY"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1BR|BASELINE_ONLY"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1BR|BASELINE_ONLY"] <- "BASELINE")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1BR|BASELINE_ONLY"] <- "S(1)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1BR|BASELINE_ONLY"] <- "Ql1:BR_S(1)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="MAXMIN|BASELINE_ONLY"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="MAXMIN|BASELINE_ONLY"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="MAXMIN|BASELINE_ONLY"] <- "BASELINE")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="MAXMIN|BASELINE_ONLY"] <- "S(1)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="MAXMIN|BASELINE_ONLY"] <- "Ql0:MM_S(1)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1MAXMIN|BASELINE_ONLY"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1MAXMIN|BASELINE_ONLY"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1MAXMIN|BASELINE_ONLY"] <- "BASELINE")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1MAXMIN|BASELINE_ONLY"] <- "S(1)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1MAXMIN|BASELINE_ONLY"] <- "Ql1:MM_S(1)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="NASH|BASELINE_ONLY"] <- "PNE-QE")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="NASH|BASELINE_ONLY"] <- "PNE-QE")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="NASH|BASELINE_ONLY"] <- "BASELINE")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="NASH|BASELINE_ONLY"] <- "S(1)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="NASH|BASELINE_ONLY"] <- "PNE-QE_S(1)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="BR|BR|BOUNDARY"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="BR|BR|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="BR|BR|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="BR|BR|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="BR|BR|BOUNDARY"] <- "Ql0:BR:BR_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="BR|BR|GAUSSIAN"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="BR|BR|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="BR|BR|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="BR|BR|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="BR|BR|GAUSSIAN"] <- "Ql0:BR:BR_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="BR|MAXMIN|BOUNDARY"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="BR|MAXMIN|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="BR|MAXMIN|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="BR|MAXMIN|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="BR|MAXMIN|BOUNDARY"] <- "Ql0:BR:MM_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="BR|MAXMIN|GAUSSIAN"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="BR|MAXMIN|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="BR|MAXMIN|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="BR|MAXMIN|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="BR|MAXMIN|GAUSSIAN"] <- "Ql0:BR:MM_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="MAXMIN|BR|BOUNDARY"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="MAXMIN|BR|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="MAXMIN|BR|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="MAXMIN|BR|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="MAXMIN|BR|BOUNDARY"] <- "Ql0:MM:BR_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="MAXMIN|BR|GAUSSIAN"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="MAXMIN|BR|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="MAXMIN|BR|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="MAXMIN|BR|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="MAXMIN|BR|GAUSSIAN"] <- "Ql0:MM:BR_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="MAXMIN|MAXMIN|BOUNDARY"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="MAXMIN|MAXMIN|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="MAXMIN|MAXMIN|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="MAXMIN|MAXMIN|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="MAXMIN|MAXMIN|BOUNDARY"] <- "Ql0:MM:MM_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="MAXMIN|MAXMIN|GAUSSIAN"] <- "Ql0")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="MAXMIN|MAXMIN|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="MAXMIN|MAXMIN|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="MAXMIN|MAXMIN|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="MAXMIN|MAXMIN|GAUSSIAN"] <- "Ql0:MM:MM_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="NASH|BR|BOUNDARY"] <- "PNE-QE")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="NASH|BR|BOUNDARY"] <- "PNE-QE")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="NASH|BR|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="NASH|BR|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="NASH|BR|BOUNDARY"] <- "PNE-QE:BR_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="NASH|BR|GAUSSIAN"] <- "PNE-QE")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="NASH|BR|GAUSSIAN"] <- "PNE-QE")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="NASH|BR|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="NASH|BR|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="NASH|BR|GAUSSIAN"] <- "PNE-QE:BR_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="NASH|MAXMIN|BOUNDARY"] <- "PNE-QE")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="NASH|MAXMIN|BOUNDARY"] <- "PNE-QE")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="NASH|MAXMIN|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="NASH|MAXMIN|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="NASH|MAXMIN|BOUNDARY"] <- "PNE-QE:MM_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="NASH|MAXMIN|GAUSSIAN"] <- "PNE-QE")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="NASH|MAXMIN|GAUSSIAN"] <- "PNE-QE")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="NASH|MAXMIN|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="NASH|MAXMIN|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="NASH|MAXMIN|GAUSSIAN"] <- "PNE-QE:MM_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1BR|BR|BOUNDARY"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1BR|BR|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1BR|BR|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1BR|BR|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1BR|BR|BOUNDARY"] <- "Ql1:BR:BR_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1BR|BR|GAUSSIAN"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1BR|BR|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1BR|BR|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1BR|BR|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1BR|BR|GAUSSIAN"] <- "Ql1:BR:BR_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1BR|MAXMIN|BOUNDARY"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1BR|MAXMIN|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1BR|MAXMIN|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1BR|MAXMIN|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1BR|MAXMIN|BOUNDARY"] <- "Ql1:BR:MM_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1BR|MAXMIN|GAUSSIAN"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1BR|MAXMIN|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1BR|MAXMIN|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1BR|MAXMIN|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1BR|MAXMIN|GAUSSIAN"] <- "Ql1:BR:MM_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1MAXMIN|BR|BOUNDARY"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1MAXMIN|BR|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1MAXMIN|BR|BOUNDARY"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1MAXMIN|BR|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1MAXMIN|BR|BOUNDARY"] <- "Ql1:MM:BR_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1MAXMIN|BR|GAUSSIAN"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1MAXMIN|BR|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1MAXMIN|BR|GAUSSIAN"] <- "BR")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1MAXMIN|BR|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1MAXMIN|BR|GAUSSIAN"] <- "Ql1:MM:BR_S(1+G)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|BOUNDARY"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|BOUNDARY"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|BOUNDARY"] <- "S(1+B)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|BOUNDARY"] <- "Ql1:MM:MM_S(1+B)")

dataset <- within(dataset, METAMODEL[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|GAUSSIAN"] <- "Ql1")
dataset <- within(dataset, L1RESP[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2RESP[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|GAUSSIAN"] <- "MAXMIN")
dataset <- within(dataset, L2SAMPLING[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|GAUSSIAN"] <- "S(1+G)")
dataset <- within(dataset, EQ_TYPE_LABEL[dataset$EQ_TYPE=="L1MAXMIN|MAXMIN|GAUSSIAN"] <- "Ql1:MM:MM_S(1+G)")

dataset$METAMODEL <- as.factor(dataset$METAMODEL)
dataset$L1RESP <- as.factor(dataset$L1RESP)
dataset$L2RESP <- as.factor(dataset$L2RESP)
dataset$EQ_TYPE_LABEL <- as.factor(dataset$EQ_TYPE_LABEL)
dataset$L2SAMPLING <- as.factor(dataset$L2SAMPLING)
dataset$EQ_TYPE_LABEL <- relevel(dataset$EQ_TYPE_LABEL,"Ql0:BR_S(1)")
dataset$TASK <- as.factor(dataset$TASK)
dataset$EQ_TYPE <- as.factor(dataset$EQ_TYPE)
dataset$NEXT_CHANGE <- as.factor(dataset$NEXT_CHANGE)
dataset$SEGMENT <- as.factor(dataset$SEGMENT)
dataset$SPEED <- as.factor(dataset$SPEED)
dataset$PEDESTRIAN <- as.factor(dataset$PEDESTRIAN)
dataset$RELEV_VEHICLE <- as.factor(dataset$RELEV_VEHICLE)
dataset$ACTIONS <- as.factor(dataset$ACTIONS)
dataset$SPEED <- relevel(dataset$SPEED,"LOW  SPEED")
dataset$AGGRESSIVE <- as.factor(dataset$AGGRESSIVE)
dataset$TASK <- as.character(dataset$TASK)
dataset$TASK[dataset$TASK=="S_W" | dataset$TASK == "E_S" | dataset$TASK == "N_E" | dataset$TASK == "W_N"] <- "LEFT_TURN"
dataset$TASK[dataset$TASK=="W_S" | dataset$TASK == "S_E"] <- "RIGHT_TURN"
dataset$SEGMENT <- as.character(dataset$SEGMENT)
dataset$SEGMENT[dataset$TASK=="LEFT_TURN" & dataset$SEGMENT == "OTHER  LANES"] <- "LEFT_TURN ENTRY OR EXIT"
dataset$SEGMENT[dataset$TASK=="RIGHT_TURN" & dataset$SEGMENT == "OTHER  LANES"] <- "RIGHT_TURN ENTRY OR EXIT"
dataset$TASK <- as.factor(dataset$TASK)
dataset$SEGMENT <- as.factor(dataset$SEGMENT)
dataset$SPEED <- as.character(dataset$SPEED)
dataset$SPEED[dataset$SPEED=="HIGH  SPEED"] <- "MEDIUM  SPEED"
dataset$SPEED <- as.factor(dataset$SPEED)

#fit the glm over the entire model to check the results

res.glm <- glm(ON_EQ ~ EQ_TYPE_LABEL + SEGMENT + NEXT_CHANGE + SPEED + PEDESTRIAN + RELEV_VEHICLE + AGGRESSIVE, family = Gamma(), data=dataset, start=rep(1,38))
summary(res.glm, dispersion = 1)


# Get the results of the EQ TYPE differences in a dataframe
t <- as.data.frame(summary(res.glm, dispersion = 1)$coefficients)
row.names(t)[1] <- "EQ_TYPE_LABELQl0:BR_S(1)"

for (r in row.names(t)) {
  if (!startsWith(r,"EQ_TYPE_LABEL")) {
    t <- t[row.names(t) != r, ]
  } else {
    row.names(t)[row.names(t) == r] <- substr(r,14, nchar(r))
  }
}
intercept_estimate <- t$Estimate[rownames(t) == "Ql0:BR_S(1)"]
t$Estimate[rownames(t) != "Ql0:BR_S(1)"] <- intercept_estimate + t$Estimate[rownames(t) != "Ql0:BR_S(1)"]

# split the dataset based on the metamodel

dataset.split.eq_type <- split(dataset, dataset$EQ_TYPE_LABEL)


# estimate alpha values for QL1 models into alpha_list, and check the significance of difference between QL0 and QL1 at the same time

alpha_list <- list() 
mmodels_cl0 <- c("Ql0:BR_S(1)","Ql0:MM_S(1)","Ql0:BR:BR_S(1+B)","Ql0:BR:BR_S(1+G)","Ql0:BR:MM_S(1+B)","Ql0:BR:MM_S(1+G)", "Ql0:MM:BR_S(1+B)","Ql0:MM:BR_S(1+G)","Ql0:MM:MM_S(1+B)","Ql0:MM:MM_S(1+G)")
for (m in mmodels_cl0){
  mcl1 <- m
  substr(mcl1, 3, 3) <- "1"
  CL0 <- dataset.split.eq_type[[m]]
  CL1 <- dataset.split.eq_type[[mcl1]]
  m_data <- CL0
  m_data <- rbind(m_data,CL1)
  res.kruskal.clk <- kruskal.test(ON_EQ ~ EQ_TYPE, data = m_data)
  cn <- c("SEGMENT", "TASK", "NEXT_CHANGE", "SPEED", "PEDESTRIAN", "RELEV_VEHICLE", "AGGRESSIVE", "FILE_ID", "TRACK_ID", "TIME")
  merged_data <- merge(select(CL0, append(cn,"ON_EQ")), select(CL1, append(cn,"ON_EQ")), by=cn)
  merged_samples <- data.frame(cbind(merged_data$ON_EQ.x,merged_data$ON_EQ.y))
  search_out <- data.frame(alpha = numeric(), loglik = numeric())
  alpha_vals <- seq(0, 1, by = 0.001)
  X_cl0 <- data.frame(merged_samples[,1])
  X_cl1 <- data.frame(merged_samples[,2])
  for (alpha in alpha_vals) {
    X_temp_cl0 <- sapply(X_cl0, function(x) (fx(x,1/x)))
    X_temp_cl1 <- sapply(X_cl1, function(x) (fx(x,1/x)))
    X_temp_prob <- (alpha*X_temp_cl0) + ((1-alpha)*X_temp_cl1)
    loglik_df <- data.frame(alpha=alpha,loglik=sum(log(X_temp_prob)))
    search_out <- rbind(search_out,loglik_df)
  }
  this_alpha <- search_out[which.max(search_out$loglik),1]
  print(c(mcl1,"alpha",this_alpha))
  alpha_list[[ mcl1 ]] <- this_alpha
  df <- data.frame(x = merged_samples[,1], y = merged_samples[,2]) %>% 
    gather(key, value)
  
  if (res.kruskal.clk$p.value < 0.05){
    res.dunn <- dunn_test(m_data, ON_EQ ~ EQ_TYPE, p.adjust.method="holm", detailed=TRUE)
    print(res.dunn)
  }
  
}

# Calculate prediction accuracy
# Use the model to find prediction accuracy

# fix the same test set size, since the evaluation is based on the log likelihood.
test_size <- 100000000000000
for (eq in levels(dataset$EQ_TYPE_LABEL)){
  m_test_size <- floor(.25*nrow(dataset.split.eq_type[[eq]]))
  if (m_test_size < test_size) {
    test_size <- m_test_size
  }
}



eq <- "Ql0:BR_S(1)"
prediction_accuracy = data.frame("EQ_TYPE"=NA)
prediction_accuracy <- prediction_accuracy[-c(1),]
sample <- sample.int(n = nrow(dataset.split.eq_type[["Ql0:BR_S(1)"]]), size = test_size, replace = F)
dataset_train <- dataset.split.eq_type[["Ql0:BR_S(1)"]][-sample, ]
dataset_test <- dataset.split.eq_type[["Ql0:BR_S(1)"]][sample, ]
#dataset_train$TASK <- NULL
dataset_train$ACTIONS <- NULL
res.glm.train <- glm(ON_EQ ~ SEGMENT + NEXT_CHANGE + SPEED + PEDESTRIAN + RELEV_VEHICLE + AGGRESSIVE , family = Gamma(), data=dataset_train, start=rep(1,14))
maineff.coeff <- res.glm.train$coefficients
names(maineff.coeff) <- NULL
res.glm.predict <- as.data.frame(predict(res.glm.train,newdata=dataset_test,type="response",dispersion=1))
res.glm.predict.accuracy <- (dataset_test$ON_EQ-res.glm.predict)^2
obs_pred <- data.frame(cbind(dataset_test$ON_EQ,res.glm.predict))
obs_pred <- cbind(obs_pred,loglik = mapply(fx, obs_pred$dataset_test.ON_EQ, 1/obs_pred$predict.res.glm.train..newdata...dataset_test..type....response...dispersion...1.) )
res.glm.predict.loglik <- sum(log(obs_pred$loglik))
this_acc <- sqrt(mean(res.glm.predict.accuracy[,1]))
this_loglik <- res.glm.predict.loglik


for (rn in 1:30) {
  tempc = data.frame("EQ_TYPE"=NA,"LOGLIK"=NA)
  tempc <- tempc[-c(1),]
  for (eq in levels(dataset$EQ_TYPE_LABEL)){
    if (substr(eq,1,3) == "Ql1"){
      ql0_lab <- eq
      substr(ql0_lab, 3, 3) <- "0"
      alpha <- alpha_list[[ eq ]]
      CL0 <- dataset.split.eq_type[[ql0_lab]]
      CL1 <- dataset.split.eq_type[[eq]]
      cn <- c("SEGMENT", "TASK", "NEXT_CHANGE", "SPEED", "PEDESTRIAN", "RELEV_VEHICLE", "AGGRESSIVE", "FILE_ID", "TRACK_ID", "TIME")
      merged_data <- merge(select(CL0, append(cn,"ON_EQ")), select(CL1, append(cn,"ON_EQ")), by=cn)
      #merged_samples <- data.frame(cbind(merged_data$ON_EQ.x,merged_data$ON_EQ.y))
      sample <- sample.int(n = nrow(dataset.split.eq_type[[eq]]), size = test_size, replace = F)
      test_set_size <- test_size
      ql0_data <- merged_data
      ql0_data$ON_EQ <- ql0_data$ON_EQ.x
      ql0_data$ON_EQ.x <- NULL
      ql0_data$ON_EQ.y <- NULL
      ql1_data <- merged_data
      ql1_data$ON_EQ <- ql1_data$ON_EQ.y
      ql1_data$ON_EQ.x <- NULL
      ql1_data$ON_EQ.y <- NULL
      sample <- sample.int(n = nrow(dataset.split.eq_type[[eq]]), size = test_size, replace = F)
      test_set_size <- test_size
      dataset_train <- ql1_data[-sample, ]
      dataset_test <- ql1_data[sample, ]
      ql0.dataset_train <- ql0_data[-sample, ]
      ql0.dataset_test <- ql0_data[sample, ]
    } else {
      sample <- sample.int(n = nrow(dataset.split.eq_type[[eq]]), size = test_size, replace = F)
      test_set_size <- test_size
      dataset_train <- dataset.split.eq_type[[eq]][-sample, ]
      dataset_test <- dataset.split.eq_type[[eq]][sample, ]
      
    }
    
    
    #dataset_train$TASK <- NULL
    #dataset_test$TASK <- NULL
    dataset_train$ACTIONS <- NULL
    dataset_test$ACTIONS <- NULL
    coeff_len <- length(unique(dataset_train$SEGMENT)) + length(unique(dataset_train$NEXT_CHANGE)) + length(unique(dataset_train$SPEED)) + length(unique(dataset_train$PEDESTRIAN)) + length(unique(dataset_train$RELEV_VEHICLE)) + length(unique(dataset_train$AGGRESSIVE)) - 6
    res.glm.train <- glm(ON_EQ ~ SEGMENT + NEXT_CHANGE + SPEED + PEDESTRIAN + RELEV_VEHICLE + AGGRESSIVE , family = Gamma(), data=dataset_train, start=rep(1,coeff_len+1))
    maineff.coeff <- res.glm.train$coefficients
    names(maineff.coeff) <- NULL
    res.glm.predict <- as.data.frame(predict(res.glm.train,newdata=dataset_test,type="response",dispersion=1))
    obs_pred <- data.frame(cbind(dataset_test$ON_EQ,res.glm.predict))
    obs_pred <- cbind(obs_pred,loglik = mapply(fx, obs_pred$dataset_test.ON_EQ, 1/obs_pred$predict.res.glm.train..newdata...dataset_test..type....response...dispersion...1.) )
    if (substr(eq,1,3) == "Ql1"){
      ql0.res.glm.train <- glm(ON_EQ ~ SEGMENT + NEXT_CHANGE + SPEED + PEDESTRIAN + RELEV_VEHICLE + AGGRESSIVE , family = Gamma(), data=ql0.dataset_train, start=rep(1,14))
      ql0.maineff.coeff <- ql0.res.glm.train$coefficients
      names(ql0.maineff.coeff) <- NULL
      ql0.res.glm.predict <- as.data.frame(predict(ql0.res.glm.train,newdata=ql0.dataset_test,type="response",dispersion=1))
      ql0.obs_pred <- data.frame(cbind(ql0.dataset_test$ON_EQ,ql0.res.glm.predict))
      ql0.obs_pred <- cbind(ql0.obs_pred,loglik = mapply(fx, ql0.obs_pred$ql0.dataset_test.ON_EQ, 1/ql0.obs_pred$predict.ql0.res.glm.train..newdata...ql0.dataset_test..type....response...dispersion...1.) )
      l0.loglik <- sum(log((ql0.obs_pred$loglik)))
      l1.loglik <- sum(log((obs_pred$loglik)))
      res.glm.predict.loglik <- sum(log((alpha*ql0.obs_pred$loglik) + ((1-alpha)*obs_pred$loglik)))
      this_loglik <- res.glm.predict.loglik
      tempd <- data.frame(eq,this_loglik)
      names(tempd)<-c("EQ_TYPE",paste("LOGLIK",rn))
      tempc <- rbind(tempc,tempd)
    } else {
      res.glm.predict.loglik <- sum(log(obs_pred$loglik))
      this_loglik <- res.glm.predict.loglik
      tempd <- data.frame(eq,this_loglik)
      names(tempd)<-c("EQ_TYPE",paste("LOGLIK",rn))
      tempc <- rbind(tempc,tempd)
    }
    print(c("run",rn,"model",eq))
  } 
  if(rn==1){
    prediction_accuracy <- tempc
  } else {
    col_n <- paste("LOGLIK",rn)
    #prediction_accuracy <- cbind(prediction_accuracy, tempc[,2])
    prediction_accuracy[,paste("LOGLIK",rn)] <- tempc[,2]
  }
}





res.glm.list <- list()
#dataset.split.eq_type <- split(dataset, dataset$EQ_TYPE)
dataset.split.eq_type <- split(dataset, dataset$EQ_TYPE_LABEL)
temp.dataset.split.eq_type <- dataset.split.eq_type

for (eq in levels(dataset$EQ_TYPE_LABEL)){
  temp.dataset.split.eq_type[[eq]]$EQ_TYPE <- NULL
  temp.dataset.split.eq_type[[eq]]$EQ_TYPE_LABEL <- NULL
  temp.dataset.split.eq_type[[eq]]$METAMODEL <- NULL
  temp.dataset.split.eq_type[[eq]]$L1RESP <- NULL
  temp.dataset.split.eq_type[[eq]]$L2RESP <- NULL
  temp.dataset.split.eq_type[[eq]]$L2SAMPLING <- NULL
  temp.dataset.split.eq_type[[eq]]$FILE_ID <- NULL
  temp.dataset.split.eq_type[[eq]]$TRACK_ID <- NULL
  temp.dataset.split.eq_type[[eq]]$TIME <- NULL
}

temp.glm.res <- glm(ON_EQ ~ . -TASK -ACTIONS , family = Gamma(), data=temp.dataset.split.eq_type[["Ql0:BR_S(1)"]], start=rep(1,14))

for (eq in levels(dataset$EQ_TYPE_LABEL)){
  print(eq)
  maineff.coeff <- temp.glm.res$coefficients
  names(maineff.coeff) <- NULL
  temp.glm.res <- glm(ON_EQ ~ . -TASK -ACTIONS , family = Gamma(), data=temp.dataset.split.eq_type[[eq]], start=rep(1,14))
  print(summary(temp.glm.res,dispersion=1))
  res.glm.list[[eq]] <- temp.glm.res
}

aic_data <- data.frame(EQ_TYPE_LABEL=character(),
                       AIC=integer(),
                       stringsAsFactors=FALSE) 
for (eq in levels(dataset$EQ_TYPE_LABEL)){
  if (substr(eq,1,3) == "Ql1"){
    ql0_lab <- eq
    substr(ql0_lab, 3, 3) <- "0"
    alpha <- alpha_list[[ eq ]]
  }
  aic <- res.glm.list[[eq]]$aic
  ql0.aic <- res.glm.list[[eq]]$aic
  if (substr(eq,1,3) == "Ql1"){
    edf <- extractAIC(res.glm.list[[ql0_lab]])[[1]]
    ql0.ll <- -(extractAIC(res.glm.list[[ql0_lab]])[[2]]-(2*edf))/2
    ql1.ll <- -(extractAIC(res.glm.list[[eq]])[[2]]-(2*edf))/2
    aic <- 2*(edf+1) - 2*((alpha*ql0.ll)+(1-alpha)*ql1.ll)
    }
  aic_data<-rbind(aic_data, data.frame(EQ_TYPE_LABEL=eq,AIC=aic))
  print(eq)
  print(aic)
  print(summary(res.glm.list[[eq]], dispersion = 1))
}
rownames(aic_data) <- aic_data[,1]
aic_data[,1] <- NULL

# Since AIC values are relative, we can scale the AIC value of every model by a constant factor. We do this for easier comparison.

aic_data[,1] <- aic_data[,1]/1000

dat.m <- melt(prediction_accuracy, id.vars = "EQ_TYPE")
res.prediction_accuracy <- cbind(data.frame(ID=prediction_accuracy[,1], MEAN=apply(prediction_accuracy[,-1], 1, mean, na.rm = TRUE)),
                                 SD=data.frame(ID=prediction_accuracy[,1], SD=apply(prediction_accuracy[,-1], 1, sd, na.rm = TRUE))[,2])
#ggplot(dat.m, aes(EQ_TYPE, value)) + geom_boxplot(aes(fill=substr(EQ_TYPE,1,3)), show.legend=FALSE) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + xlab("Behavior model") + ylab("log likelihood")

#ggplot(t, aes(x=rownames(t), y=t$Estimate)) + geom_point(color=c(rep( "red",10),rep( "green4",10),rep( "blue",5))) + geom_errorbar(aes(x=rownames(t), ymin=t$Estimate-t$`Std. Error`, ymax=t$Estimate+t$`Std. Error`), color=c(rep( "red",10),rep( "green4",10),rep( "blue",5))) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + xlab("Behavior models") + xlab("Behavior model") + ylab(expression(lambda["g"])) + geom_text(aes(label=paste("(",sprintf("%0.2f", round(aic_data[,1], digits = 2)),")")),hjust=c(rep(0.3,24),0.5), vjust=c(rep(2,1),-0.5,rep(2,23)), size=3) + ylim(17,26.5) + coord_flip()
#ggplot(dat.m, aes(EQ_TYPE, value)) + geom_boxplot(aes(fill=substr(EQ_TYPE,1,3)), show.legend=FALSE) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + xlab("Behavior models") + ylab("log likelihood") + coord_flip() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + theme(legend.position = "none", axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank())

p1 <- ggplot(t, aes(x=rownames(t), y=t$Estimate)) + geom_point(aes(color=substr(rownames(t),1,3)), show.legend=FALSE) + geom_errorbar(aes(x=rownames(t), ymin=t$Estimate-t$`Std. Error`, ymax=t$Estimate+t$`Std. Error`, color=substr(rownames(t),1,3)), show.legend=FALSE)  + xlab("Behavior models") + ylab(expression(lambda["g"])) + geom_text(aes(label=paste("(",sprintf("%0.2f", round(aic_data[,1], digits = 2)),")")),hjust=c(rep(.3,1),.5,rep(.3,23)), vjust=c(rep(1,1),-0.5,rep(1,23)), size=3) + coord_flip()
p2 <- ggplot(dat.m, aes(EQ_TYPE, value)) + geom_boxplot(aes(fill=substr(EQ_TYPE,1,3)), show.legend=FALSE) + ylab("log likelihood") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + theme(legend.position = "none", axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank())  +  coord_flip()
library(grid)
grid.newpage()
#pushViewport(viewport(angle=-90, width = unit(6, "inches"), height = unit(5, "inches")))
grid.draw(cbind(ggplotGrob(p1), ggplotGrob(p2), size = "last"))
