library(tidyverse)
library(broom)
library(purrr)
library(dplyr)
library(dominanceanalysis)
library(sjPlot)
library(rstatix)

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

em_max <- function(merged_samples) {
  #initial params
  alpha_lastiter <- 0.5
  
  l1_last <- mean(merged_samples[,1])
  l0_last <- mean(merged_samples[,2])
  est <- c(alpha_lastiter,l0_last,l1_last)
  print(est)
  # E-step (calculate mixing proportions)
  for (iter in 1:100){
    X_cl0 <- data.frame(merged_samples[,1])
    X_cl1 <- data.frame(merged_samples[,2])
    X = data.frame((alpha_lastiter*merged_samples[,1]) + ((1-alpha_lastiter)* merged_samples[,2]))
    n_sum <- sapply(X, function(x) ((fx(x,1/l0_last)*alpha_lastiter))/gx(x,1/l0_last,1/l1_last,alpha_lastiter))
    alpha_iter = sum(n_sum)/nrow(X)
    
    l_n_comp <- sapply(X, function(x) (fx(x,1/l0_last)*x)/(gx(x,1/l0_last,1/l1_last,alpha_lastiter)))
    l_d_comp <- sapply(X, function(x) (fx(x,1/l0_last)*1)/(gx(x,1/l0_last,1/l1_last,alpha_lastiter)))
    l0 <- sum(l_n_comp)/sum(l_d_comp)
    
    l_n_comp <- sapply(X, function(x) (fx(x,1/l1_last)*x)/(gx(x,1/l0_last,1/l1_last,alpha_lastiter)))
    l_d_comp <- sapply(X, function(x) (fx(x,1/l1_last)*1)/(gx(x,1/l0_last,1/l1_last,alpha_lastiter)))
    l1 <- sum(l_n_comp)/sum(l_d_comp)
    
    
    est <- c(iter,alpha_iter,1/l0,1/l1)
    print(est)
    alpha_lastiter <- alpha_iter
    l0_last <- l0
    l1_last <- l1
    }
}

em_max1 <- function(merged_samples) {
  #initial params
  alpha_lastiter <- 0.5
  
  l1_last <- mean(merged_samples[,1])
  l0_last <- mean(merged_samples[,2])
  est <- c(alpha_lastiter,l0_last,l1_last)
  print(est)
  # E-step (calculate mixing proportions)
  for (iter in 1:100){
    X_cl0 <- data.frame(merged_samples[,1])
    X_cl1 <- data.frame(merged_samples[,2])
    X = data.frame((alpha_lastiter*merged_samples[,1]) + ((1-alpha_lastiter)* merged_samples[,2]))
    n_sum1 <- sapply(X_cl0, function(x) (fx(x,1/x)))
    n_sum <- sapply(X, function(x) ((fx(x,1/l0_last)*alpha_lastiter))/gx(x,1/l0_last,1/l1_last,alpha_lastiter))
    
    alpha_iter = sum(n_sum)/nrow(X)
    
    est <- c(iter,alpha_iter,1/l0,1/l1)
    print(est)
    alpha_lastiter <- alpha_iter
    l0_last <- l0
    l1_last <- l1
  }
}

search_opt <- function(merged_samples) {
  search_out <- data.frame(alpha = numeric(), loglik = numeric())
  alpha_vals <- seq(0, 1, by = 0.001)
  X_cl0 <- data.frame(merged_samples[,1])
  X_cl1 <- data.frame(merged_samples[,2])
  for (alpha in alpha_vals) {
    X_temp_cl0 <- sapply(X_cl0, function(x) (fx(x,1/x)))
    X_temp_cl1 <- sapply(X_cl1, function(x) (fx(x,1/x)))
    X_temp_prob <- (alpha*X_temp_cl0) + ((1-alpha)*X_temp_cl1)
    loglik_df <- data.frame(alpha=alpha,loglik=sum(log(X_temp_prob)))
    print(loglik_df)
    search_out <- rbind(search_out,loglik_df)
    print(alpha)
  }
  
}
search_out <- data.frame(alpha = numeric(), loglik = numeric())
alpha_vals <- seq(0, 1, by = 0.001)
X_cl0 <- data.frame(merged_samples[,1])
X_cl1 <- data.frame(merged_samples[,2])
for (alpha in alpha_vals) {
  X_temp_cl0 <- sapply(X_cl0, function(x) (fx(x,1/x)))
  X_temp_cl1 <- sapply(X_cl1, function(x) (fx(x,1/x)))
  X_temp_prob <- (alpha*X_temp_cl0) + ((1-alpha)*X_temp_cl1)
  loglik_df <- data.frame(alpha=alpha,loglik=sum(log(X_temp_prob)))
  print(loglik_df)
  search_out <- rbind(search_out,loglik_df)
  print(alpha)
}
ggplot(search_out, aes(x=search_out$alpha, y=search_out$loglik)) +
geom_line()



# Load result files into dataset
setwd("F:\\Spring2017\\workspaces\\game_theoretic_planner\\results_all_2907")
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

# assign epsilon values to zero values for Gamma glm fitting
library(data.table)
setDT(dataset)[ON_EQ == 0, ON_EQ := runif(.N, min=0.000000001, max=0.0000001)]




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

# RQ1: Which solution concept better predicts empirical driving behavior.
# Run Kruskal Walis test to check that the difference in EQ_TYPE group is significant

res.kruskal <- kruskal.test(ON_EQ ~ EQ_TYPE , data = dataset)
print(res.kruskal)

# Run Dunn's post hoc test on significant Kruskal Walis result to test the significant between pairwise comparison of the group factors.

library(rstatix)
res.dunn <- dunn_test(dataset, ON_EQ ~ EQ_TYPE, p.adjust.method="holm", detailed=TRUE)
View(res.dunn)


res.eqtype_only.glm <- glm(ON_EQ ~ EQ_TYPE_LABEL , family = Gamma(), data=dataset)
summary(res.eqtype_only.glm, dispersion=1)
plot_model(res.eqtype_only.glm, transform = NULL, show.p = TRUE, show.values = TRUE, value.size = 3, value.offset = .3, digits = 1, dot.size = 1.5, p.style = "asterisk", axis.labels = levels(dataset$EQ_TYPE_LABEL), axis.title = c("lambda estimates wrt Ql0:BR_S(1)","lambda-29.84"))


t.res.glm <- glm(ON_EQ ~ METAMODEL + L1RESP + L2RESP + L2SAMPLING, family = Gamma(), data=dataset)
summary(t.res.glm, dispersion = 1)

res.glm <- glm(ON_EQ ~ EQ_TYPE_LABEL + SEGMENT + NEXT_CHANGE + SPEED + PEDESTRIAN + RELEV_VEHICLE + AGGRESSIVE, family = Gamma(), data=dataset)
summary(res.glm, dispersion = 1)
plot_model(res.glm, transform = NULL, show.p = TRUE, show.values = TRUE, value.size = 3, value.offset = .3, digits = 1, dot.size = 1.5, p.style = "asterisk", rm.terms=c("SPEED"))

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
#qplot(rownames(t),t$Estimate) + geom_errorbar(aes(x=rownames(t), ymin=t$Estimate-t$`Std. Error`, ymax=t$Estimate+t$`Std. Error`), color=c(rep(1,10),rep(2,10),rep(3,5))) + theme(axis.text.x = element_text(angle = 90)) + xlab("Behavior model") + ylab(expression(lambda["g"])) 
ggplot(t, aes(x=rownames(t), y=t$Estimate)) + geom_point(color=c(rep( "red",10),rep( "green4",10),rep( "blue",5))) + geom_errorbar(aes(x=rownames(t), ymin=t$Estimate-t$`Std. Error`, ymax=t$Estimate+t$`Std. Error`), color=c(rep( "red",10),rep( "green4",10),rep( "blue",5))) + theme(axis.text.x = element_text(angle = 90), legend.position = "none") + xlab("Behavior model") + ylab(expression(lambda["g"]))

# RQ2

# Change the task directions to task types
dataset$TASK <- as.character(dataset$TASK)
dataset$TASK[dataset$TASK=="S_W" | dataset$TASK == "E_S" | dataset$TASK == "N_E" | dataset$TASK == "W_N"] <- "LEFT_TURN"
dataset$TASK[dataset$TASK=="W_S" | dataset$TASK == "S_E"] <- "RIGHT_TURN"
dataset$SEGMENT <- as.character(dataset$SEGMENT)
dataset$SEGMENT[dataset$TASK=="LEFT_TURN" & dataset$SEGMENT == "OTHER  LANES"] <- "LEFT_TURN ENTRY OR EXIT"
dataset$SEGMENT[dataset$TASK=="RIGHT_TURN" & dataset$SEGMENT == "OTHER  LANES"] <- "RIGHT_TURN ENTRY OR EXIT"
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



maineff.coeff <- res.glm$coefficients
names(maineff.coeff) <- NULL
#res.glm.interactions <- glm(ON_EQ ~ . - TASK - ACTIONS + SEGMENT:NEXT_CHANGE + SEGMENT:SPEED + SEGMENT:PEDESTRIAN + SEGMENT:RELEV_VEHICLE, family = Gamma(), data=dataset, start=c(maineff.coeff,rep(1,40)))

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
temp.glm.res <- glm(ON_EQ ~ . -TASK -ACTIONS , family = Gamma(), data=temp.dataset.split.eq_type[[eq]], , start=rep(1,14))
print(summary(temp.glm.res,dispersion=1))
res.glm.list[[eq]] <- temp.glm.res
}

aic_data <- data.frame(EQ_TYPE_LABEL=character(),
                 AIC=integer(),
                 stringsAsFactors=FALSE) 
for (eq in levels(dataset$EQ_TYPE_LABEL)){
  aic <- res.glm.list[[eq]]$aic
  aic_data<-rbind(aic_data, data.frame(EQ_TYPE_LABEL=eq,AIC=aic))
  print(eq)
  print(aic)
  print(summary(res.glm.list[[eq]], dispersion = 1))
}
rownames(aic_data) <- aic_data[,1]
aic_data[,1] <- NULL

aic_data[,1] <- aic_data[,1]/1000

ggplot(t, aes(x=rownames(t), y=t$Estimate)) + geom_point(color=c(rep( "red",10),rep( "green4",10),rep( "blue",5))) + geom_errorbar(aes(x=rownames(t), ymin=t$Estimate-t$`Std. Error`, ymax=t$Estimate+t$`Std. Error`), color=c(rep( "red",10),rep( "green4",10),rep( "blue",5))) + theme(axis.text.x = element_text(angle = 90), legend.position = "none") + xlab("Behavior model") + ylab(expression(lambda["g"])) + geom_text(aes(label=paste("(",sprintf("%0.2f", round(aic_data[,1], digits = 2)),")")),hjust=0.3, vjust=3, size=3)

print(summary(res.glm.list[["BR|BASELINE_ONLY"]]))
print(summary(res.glm.list[["BR|BR|BOUNDARY"]]))

for (eq in levels(dataset$EQ_TYPE_LABEL)){
  print(summary(res.glm.list[[eq]]))
  dapres <- try(dominanceAnalysis(res.glm.list[[eq]]))
  if(inherits(dapres, "try-error")) {
    print("error. continuing")
    next
  }
  temp.c <- averageContribution(dapres,fit.functions = "r2.e")
  if (!exists("dominance.data")){
    dominance.data <- data.frame(as.list(temp.c$r2.e))
    dominance.data$EQ_TYPE <- eq
  } else {
    temp.d <- data.frame(as.list(temp.c$r2.e))
    temp.d$EQ_TYPE <- eq
    dominance.data <- rbind(dominance.data,temp.d)
  }
}

dominance.rank <- data.frame(dominance.data[,7], t(apply(-dominance.data[,1:6], 1, rank, ties.method='min')))
names(dominance.rank)
names(dominance.rank)[1] <- "EQ_TYPE_LABEL"
test_data_long <- melt(dominance.rank, id="EQ_TYPE_LABEL")
ggplot(test_data_long, aes(x=EQ_TYPE_LABEL, y=value, group=variable)) +
geom_line(aes(color=variable))+
geom_point(aes(color=variable)) + theme(axis.text.x = element_text(angle = 90, vjust = 1, hjust=1))


# RQ3

# Split the dataset based on the l1l2 eq_type
dataset.split.eq_type <- split(dataset, dataset$EQ_TYPE)
l2_levels <- c("BR","MAXMIN","NASH","L1BR","L1MAXMIN")
dataset.split.l1l2 <- list()
for (l in l2_levels){
dataset.split.l1l2[[l]] <- subset(dataset, grepl(paste("^",l,"\\|*", sep = ""), EQ_TYPE))
dataset.split.l1l2[[l]]$TRAJ_TYPE <- NA
dataset.split.l1l2[[l]]$L3_EQ <- NA
}

dataset.split.l1l2$MAXMIN$L3_EQ[dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|BR|BOUNDARY" |
dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|BR|GAUSSIAN"] <- "BR"
dataset.split.l1l2$MAXMIN$L3_EQ[dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|MAXMIN|BOUNDARY" |
dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|MAXMIN|GAUSSIAN"] <- "MAXMIN"
dataset.split.l1l2$MAXMIN$TRAJ_TYPE[dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|BR|GAUSSIAN" |
dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|MAXMIN|GAUSSIAN"] <- "GAUSSIAN"
dataset.split.l1l2$MAXMIN$TRAJ_TYPE[dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|BR|BOUNDARY" |
dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|MAXMIN|BOUNDARY"] <- "BOUNDARY"
dataset.split.l1l2$MAXMIN$TRAJ_TYPE[dataset.split.l1l2$MAXMIN$EQ_TYPE=="MAXMIN|BASELINE_ONLY"] <- "BASELINE"

dataset.split.l1l2$L1MAXMIN$L3_EQ[dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|BR|BOUNDARY" |
                                  dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|BR|GAUSSIAN"] <- "BR"
dataset.split.l1l2$L1MAXMIN$L3_EQ[dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|MAXMIN|BOUNDARY" |
                                  dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|MAXMIN|GAUSSIAN"] <- "MAXMIN"
dataset.split.l1l2$L1MAXMIN$TRAJ_TYPE[dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|BR|GAUSSIAN" |
                                      dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|MAXMIN|GAUSSIAN"] <- "GAUSSIAN"
dataset.split.l1l2$L1MAXMIN$TRAJ_TYPE[dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|BR|BOUNDARY" |
                                      dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|MAXMIN|BOUNDARY"] <- "BOUNDARY"
dataset.split.l1l2$L1MAXMIN$TRAJ_TYPE[dataset.split.l1l2$L1MAXMIN$EQ_TYPE=="L1MAXMIN|BASELINE_ONLY"] <- "BASELINE"

dataset.split.l1l2$NASH$L3_EQ[dataset.split.l1l2$NASH$EQ_TYPE=="NASH|BR|BOUNDARY" |
dataset.split.l1l2$NASH$EQ_TYPE=="NASH|BR|GAUSSIAN"] <- "BR"
dataset.split.l1l2$NASH$L3_EQ[dataset.split.l1l2$NASH$EQ_TYPE=="NASH|MAXMIN|BOUNDARY" |
dataset.split.l1l2$NASH$EQ_TYPE=="NASH|MAXMIN|GAUSSIAN"] <- "MAXMIN"
dataset.split.l1l2$NASH$TRAJ_TYPE[dataset.split.l1l2$NASH$EQ_TYPE=="NASH|BR|GAUSSIAN" |
dataset.split.l1l2$NASH$EQ_TYPE=="NASH|MAXMIN|GAUSSIAN"] <- "GAUSSIAN"
dataset.split.l1l2$NASH$TRAJ_TYPE[dataset.split.l1l2$NASH$EQ_TYPE=="NASH|BR|BOUNDARY" |
dataset.split.l1l2$NASH$EQ_TYPE=="NASH|MAXMIN|BOUNDARY"] <- "BOUNDARY"
dataset.split.l1l2$NASH$TRAJ_TYPE[dataset.split.l1l2$NASH$EQ_TYPE=="NASH|BASELINE_ONLY"] <- "BASELINE"

dataset.split.l1l2$BR$L3_EQ[dataset.split.l1l2$BR$EQ_TYPE=="BR|BR|BOUNDARY" |
dataset.split.l1l2$BR$EQ_TYPE=="BR|BR|GAUSSIAN"] <- "BR"
dataset.split.l1l2$BR$L3_EQ[dataset.split.l1l2$BR$EQ_TYPE=="BR|MAXMIN|BOUNDARY" |
dataset.split.l1l2$BR$EQ_TYPE=="BR|MAXMIN|GAUSSIAN"] <- "MAXMIN"
dataset.split.l1l2$BR$TRAJ_TYPE[dataset.split.l1l2$BR$EQ_TYPE=="BR|BR|GAUSSIAN" |
dataset.split.l1l2$BR$EQ_TYPE=="BR|MAXMIN|GAUSSIAN"] <- "GAUSSIAN"
dataset.split.l1l2$BR$TRAJ_TYPE[dataset.split.l1l2$BR$EQ_TYPE=="BR|BR|BOUNDARY" |
dataset.split.l1l2$BR$EQ_TYPE=="BR|MAXMIN|BOUNDARY"] <- "BOUNDARY"
dataset.split.l1l2$BR$TRAJ_TYPE[dataset.split.l1l2$BR$EQ_TYPE=="BR|BASELINE_ONLY"] <- "BASELINE"

dataset.split.l1l2$L1BR$L3_EQ[dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|BR|BOUNDARY" |
                              dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|BR|GAUSSIAN"] <- "BR"
dataset.split.l1l2$L1BR$L3_EQ[dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|MAXMIN|BOUNDARY" |
                              dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|MAXMIN|GAUSSIAN"] <- "MAXMIN"
dataset.split.l1l2$L1BR$TRAJ_TYPE[dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|BR|GAUSSIAN" |
                                  dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|MAXMIN|GAUSSIAN"] <- "GAUSSIAN"
dataset.split.l1l2$L1BR$TRAJ_TYPE[dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|BR|BOUNDARY" |
                                  dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|MAXMIN|BOUNDARY"] <- "BOUNDARY"
dataset.split.l1l2$L1BR$TRAJ_TYPE[dataset.split.l1l2$L1BR$EQ_TYPE=="L1BR|BASELINE_ONLY"] <- "BASELINE"

dataset.split.l1l2$BR$TRAJ_TYPE <- as.factor(dataset.split.l1l2$BR$TRAJ_TYPE)
dataset.split.l1l2$BR$EQ_TYPE <- as.factor(dataset.split.l1l2$BR$EQ_TYPE)
dataset.split.l1l2$NASH$TRAJ_TYPE <- as.factor(dataset.split.l1l2$NASH$TRAJ_TYPE)
dataset.split.l1l2$NASH$EQ_TYPE <- as.factor(dataset.split.l1l2$NASH$EQ_TYPE)
dataset.split.l1l2$MAXMIN$TRAJ_TYPE <- as.factor(dataset.split.l1l2$MAXMIN$TRAJ_TYPE)
dataset.split.l1l2$MAXMIN$EQ_TYPE <- as.factor(dataset.split.l1l2$MAXMIN$EQ_TYPE)
dataset.split.l1l2$L1MAXMIN$TRAJ_TYPE <- as.factor(dataset.split.l1l2$L1MAXMIN$TRAJ_TYPE)
dataset.split.l1l2$L1MAXMIN$EQ_TYPE <- as.factor(dataset.split.l1l2$L1MAXMIN$EQ_TYPE)


data.split.l1 <- dataset.split.l1l2$BR
data.split.l1$EQ_TYPE <- "BR"
temp.d <- dataset.split.l1l2$L1BR
temp.d$EQ_TYPE <- "L1BR"
data.split.l1 <- rbind(data.split.l1,temp.d)
temp.d <- dataset.split.l1l2$MAXMIN
temp.d$EQ_TYPE <- "MAXMIN"
data.split.l1 <- rbind(data.split.l1,temp.d)
temp.d <- dataset.split.l1l2$L1MAXMIN
temp.d$EQ_TYPE <- "L1MAXMIN"
data.split.l1 <- rbind(data.split.l1,temp.d)
temp.d <- dataset.split.l1l2$NASH
temp.d$EQ_TYPE <- "NASH"
data.split.l1 <- rbind(data.split.l1,temp.d)
View(data.split.l1)

# Run Kruskal-Walis test to check for significant difference between groups
res.kruskal.l1 <- kruskal.test(ON_EQ ~ EQ_TYPE, data = data.split.l1)
print(res.kruskal.l1)

# Run post-hoc Dunn's test to check for pairwise significant difference between groups
res.dunn.l1 <- dunn_test(data.split.l1, ON_EQ ~ EQ_TYPE, p.adjust.method="holm", detailed=TRUE)
print(res.dunn.l1)

# Run Kruskal-Walis test to check for significant difference among the levels

for (l in l2_levels){
  print(l)
  res.kruskal.l1l2 <- kruskal.test(ON_EQ ~ EQ_TYPE, data = dataset.split.l1l2[[l]])
  print(res.kruskal.l1l2)
}

# Run post-hoc Dunn's test to check for pairwise significant difference

for (l in l2_levels){
res.dunn.l1l2 <- dunn_test(dataset.split.l1l2[[l]], ON_EQ ~ L3_EQ, p.adjust.method="bonferroni", detailed=TRUE)
print(l)
print(res.dunn.l1l2)
}

# For the l1l2 levels where significant difference in l3_eq type was found, create glm model to analyse the impact of the l3_eq on the rationality parameter. 

res.glm.l1l2.list.rq3 <- list()
for (l in l2_levels){
#res.glm.l1l2.list.rq3[[l]] <- glm(ON_EQ ~ . - TASK - ACTIONS - EQ_TYPE, family = Gamma(), data=dataset.split.l1l2[[l]])
res.glm.l1l2.list.rq3[[l]] <- glm(ON_EQ ~ TRAJ_TYPE + L3_EQ, family = Gamma(), data=dataset.split.l1l2[[l]])
print(l)
print(res.glm.l1l2.list.rq3[[l]])
}

for (l in l2_levels){
print(l)
print(summary(res.glm.l1l2.list.rq3[[l]], dispersion = 1))
}


# Check for statistical significance in cl0 cl1 difference
alpha_list <- list() 
mmodels_cl0 <- c("Ql0:BR_S(1)","Ql0:MM_S(1)","Ql0:BR:BR_S(1+B)","Ql0:BR:BR_S(1+G)","Ql0:BR:MM_S(1+B)","Ql0:BR:MM_S(1+G)", "Ql0:MM:BR_S(1+B)","Ql0:MM:BR_S(1+G)","Ql0:MM:MM_S(1+B)","Ql0:MM:MM_S(1+G)")
for (m in mmodels_cl0){
  mcl1 <- m
  substr(mcl1, 3, 3) <- "1"
  print(c(m,mcl1))
  CL0 <- dataset.split.eq_type[[m]]
  CL1 <- dataset.split.eq_type[[mcl1]]
  m_data <- CL0
  m_data <- rbind(m_data,CL1)
  res.kruskal.clk <- kruskal.test(ON_EQ ~ EQ_TYPE, data = m_data)
  print(res.kruskal.clk$p.value)
  cn <- cn <- c("SEGMENT", "TASK", "NEXT_CHANGE", "SPEED", "PEDESTRIAN", "RELEV_VEHICLE", "AGGRESSIVE", "FILE_ID", "TRACK_ID", "TIME")
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
  print(c("alpha",this_alpha))
  alpha_list[[ mcl1 ]] <- this_alpha
  df <- data.frame(x = merged_samples[,1], y = merged_samples[,2]) %>% 
    gather(key, value)
  print(ggplot(df, aes(value, colour = key)) +geom_density(show.legend = F) +theme_minimal() +scale_color_manual(values = c(x = "red", y = "blue")))
  if (res.kruskal.clk$p.value < 0.05){
    res.dunn <- dunn_test(m_data, ON_EQ ~ EQ_TYPE, p.adjust.method="holm", detailed=TRUE)
    print(res.dunn)
  }
  invisible(readline(prompt="Press [enter] to continue"))
}




# Calculate prediction accuracy

# Use the model to find prediction accuracy

eq <- "Ql0:BR_S(1)"
prediction_accuracy = data.frame("EQ_TYPE"=NA)
prediction_accuracy <- prediction_accuracy[-c(1),]
sample <- sample.int(n = nrow(dataset.split.eq_type[["Ql0:BR_S(1)"]]), size = floor(.75*nrow(dataset.split.eq_type[["Ql0:BR_S(1)"]])), replace = F)
dataset_train <- dataset.split.eq_type[["Ql0:BR_S(1)"]][sample, ]
#dataset_train$TASK <- NULL
dataset_train$ACTIONS <- NULL
res.glm.train <- glm(ON_EQ ~ SEGMENT + NEXT_CHANGE + SPEED + PEDESTRIAN + RELEV_VEHICLE + AGGRESSIVE , family = Gamma(), data=dataset_train)
maineff.coeff <- res.glm.train$coefficients
names(maineff.coeff) <- NULL
res.glm.predict <- as.data.frame(predict(res.glm.train,newdata=dataset_test,type="response",dispersion=1))
res.glm.predict.accuracy <- (dataset_test$ON_EQ-res.glm.predict)^2
obs_pred <- data.frame(cbind(dataset_test$ON_EQ,res.glm.predict))
obs_pred <- cbind(obs_pred,loglik = mapply(fx, obs_pred$dataset_test.ON_EQ, 1/obs_pred$predict.res.glm.train..newdata...dataset_test..type....response...dispersion...1.) )
res.glm.predict.loglik <- sum(log(obs_pred$loglik))
this_acc <- sqrt(mean(res.glm.predict.accuracy[,1]))
this_loglik <- res.glm.predict.loglik
print(eq)
print(this_loglik)

for (rn in 1:30) {
  tempc = data.frame("EQ_TYPE"=NA,"LOGLIK"=NA)
  tempc <- tempc[-c(1),]
  print(rn)
  for (eq in levels(dataset$EQ_TYPE_LABEL)){
    sample <- sample.int(n = nrow(dataset.split.eq_type[[eq]]), size = floor(.75*nrow(dataset.split.eq_type[[eq]])), replace = F)
    dataset_train <- dataset.split.eq_type[[eq]][sample, ]
    dataset_test <- dataset.split.eq_type[[eq]][-sample, ]
    if (substr(eq,1,3) == "Ql1"){
      ql0_lab <- eq
      substr(ql0_lab, 3, 3) <- "0"
      alpha <- alpha_list[[ eq ]]
      ql0.dataset_train <- dataset.split.eq_type[[ql0_lab]][sample, ]
      ql0.dataset_test <- dataset.split.eq_type[[ql0_lab]][-sample, ]
      print(c("alpha",alpha))
    }
    
    
    #dataset_train$TASK <- NULL
    #dataset_test$TASK <- NULL
    dataset_train$ACTIONS <- NULL
    dataset_test$ACTIONS <- NULL
    res.glm.train <- glm(ON_EQ ~ SEGMENT + NEXT_CHANGE + SPEED + PEDESTRIAN + RELEV_VEHICLE + AGGRESSIVE , family = Gamma(), data=dataset_train, start = maineff.coeff)
    maineff.coeff <- res.glm.train$coefficients
    names(maineff.coeff) <- NULL
    res.glm.predict <- as.data.frame(predict(res.glm.train,newdata=dataset_test,type="response",dispersion=1))
    obs_pred <- data.frame(cbind(dataset_test$ON_EQ,res.glm.predict))
    obs_pred <- cbind(obs_pred,loglik = mapply(fx, obs_pred$dataset_test.ON_EQ, 1/obs_pred$predict.res.glm.train..newdata...dataset_test..type....response...dispersion...1.) )
    if (substr(eq,1,3) == "Ql1"){
      ql0.res.glm.train <- glm(ON_EQ ~ SEGMENT + NEXT_CHANGE + SPEED + PEDESTRIAN + RELEV_VEHICLE + AGGRESSIVE , family = Gamma(), data=ql0.dataset_train, start = maineff.coeff)
      ql0.maineff.coeff <- ql0.res.glm.train$coefficients
      names(ql0.maineff.coeff) <- NULL
      ql0.res.glm.predict <- as.data.frame(predict(ql0.res.glm.train,newdata=ql0.dataset_test,type="response",dispersion=1))
      ql0.obs_pred <- data.frame(cbind(ql0.dataset_test$ON_EQ,ql0.res.glm.predict))
      ql0.obs_pred <- cbind(ql0.obs_pred,loglik = mapply(fx, ql0.obs_pred$ql0.dataset_test.ON_EQ, 1/ql0.obs_pred$predict.ql0.res.glm.train..newdata...ql0.dataset_test..type....response...dispersion...1.) )
      res.glm.predict.loglik <- sum(log((alpha*ql0.obs_pred$loglik) + ((1-alpha)*obs_pred$loglik)))
      this_loglik <- res.glm.predict.loglik
      print(eq)
      print(this_loglik)
      tempd <- data.frame(eq,this_loglik)
      names(tempd)<-c("EQ_TYPE",paste("LOGLIK",rn))
      tempc <- rbind(tempc,tempd)
      
    } else {
    res.glm.predict.loglik <- sum(log(obs_pred$loglik))
    this_loglik <- res.glm.predict.loglik
    print(eq)
    print(this_loglik)
    tempd <- data.frame(eq,this_loglik)
    names(tempd)<-c("EQ_TYPE",paste("LOGLIK",rn))
    tempc <- rbind(tempc,tempd)
    }
  } 
  if(rn==1){
    prediction_accuracy <- tempc
  } else {
    col_n <- paste("LOGLIK",rn)
    #prediction_accuracy <- cbind(prediction_accuracy, tempc[,2])
    prediction_accuracy[,paste("LOGLIK",rn)] <- tempc[,2]
  }
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
  aic_data<-rbind(aic_data, data.frame(EQ_TYPE_LABEL=eq,AIC=aic))
  print(eq)
  print(aic)
  print(summary(res.glm.list[[eq]], dispersion = 1))
}
rownames(aic_data) <- aic_data[,1]
aic_data[,1] <- NULL

aic_data[,1] <- aic_data[,1]/1000

dat.m <- melt(prediction_accuracy, id.vars = "EQ_TYPE")
res.prediction_accuracy <- cbind(data.frame(ID=prediction_accuracy[,1], MEAN=apply(prediction_accuracy[,-1], 1, mean)),
                                 SD=data.frame(ID=prediction_accuracy[,1], SD=apply(prediction_accuracy[,-1], 1, sd))[,2])
ggplot(dat.m, aes(EQ_TYPE, value)) + geom_boxplot(aes(fill=substr(EQ_TYPE,1,3)), show.legend=FALSE) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + xlab("Behavior model") + ylab("log likelihood")

ggplot(t, aes(x=rownames(t), y=t$Estimate)) + geom_point(color=c(rep( "red",10),rep( "green4",10),rep( "blue",5))) + geom_errorbar(aes(x=rownames(t), ymin=t$Estimate-t$`Std. Error`, ymax=t$Estimate+t$`Std. Error`), color=c(rep( "red",10),rep( "green4",10),rep( "blue",5))) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + xlab("Behavior models") + xlab("Behavior model") + ylab(expression(lambda["g"])) + geom_text(aes(label=paste("(",sprintf("%0.2f", round(aic_data[,1], digits = 2)),")")),hjust=c(rep(0.3,24),0.5), vjust=c(rep(2,1),-0.5,rep(2,23)), size=3) + ylim(17,26.5) + coord_flip()
ggplot(dat.m, aes(EQ_TYPE, value)) + geom_boxplot(aes(fill=substr(EQ_TYPE,1,3)), show.legend=FALSE) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + xlab("Behavior models") + ylab("log likelihood") + ylim(6500,8700) +  coord_flip() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + theme(legend.position = "none", axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank())

p1 <- ggplot(t, aes(x=rownames(t), y=t$Estimate)) + geom_point(aes(color=substr(rownames(t),1,3)), show.legend=FALSE) + geom_errorbar(aes(x=rownames(t), ymin=t$Estimate-t$`Std. Error`, ymax=t$Estimate+t$`Std. Error`, color=substr(rownames(t),1,3)), show.legend=FALSE)  + xlab("Behavior models") + ylab(expression(lambda["g"])) + geom_text(aes(label=paste("(",sprintf("%0.2f", round(aic_data[,1], digits = 2)),")")),hjust=c(rep(0.3,24),0.5), vjust=c(rep(2,1),-0.5,rep(2,23)), size=3) + ylim(17,26.5) + coord_flip()
p2 <- ggplot(dat.m, aes(EQ_TYPE, value)) + geom_boxplot(aes(fill=substr(EQ_TYPE,1,3)), show.legend=FALSE) + ylab("log likelihood") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + theme(legend.position = "none", axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank())   + ylim(6500,8700) +  coord_flip()
library(grid)
grid.newpage()
#pushViewport(viewport(angle=-90, width = unit(6, "inches"), height = unit(5, "inches")))
grid.draw(cbind(ggplotGrob(p1), ggplotGrob(p2), size = "last"))

cn <- c("SEGMENT", "TASK", "NEXT_CHANGE", "SPEED", "PEDESTRIAN", "RELEV_VEHICLE", "AGGRESSIVE", "FILE_ID", "TRACK_ID", "TIME")
merged_data <- merge(select(CL0, append(cn,"ON_EQ")), select(CL1, append(cn,"ON_EQ")), by=cn)
merged_samples <- data.frame(cbind(merged_data$ON_EQ.x,merged_data$ON_EQ.y))
df <- data.frame(x = merged_samples[,1], y = merged_samples[,2]) %>% 
  +     gather(key, value)
ggplot(df, aes(value, colour = key)) +
  +     geom_density(show.legend = F) +
  +     theme_minimal() +
  +     scale_color_manual(values = c(x = "red", y = "blue"))
