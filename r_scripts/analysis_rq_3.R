# load the necessary packages 
library(tidyverse)
library(broom)
library(purrr)
library(dplyr)
library(dominanceanalysis)
library(sjPlot)
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

# assign small  ~zero values for Gamma glm fitting
library(data.table)
setDT(dataset)[ON_EQ == 0, ON_EQ := runif(.N, min=0.000000001, max=0.0000001)]

# prepare the labels for the models

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
dataset$SPEED <- as.character(dataset$SPEED)
dataset$SPEED[dataset$SPEED=="HIGH  SPEED"] <- "MEDIUM  SPEED"
dataset$SPEED <- as.factor(dataset$SPEED)


# Split the dataset based on the l1l2 eq_type
dataset.split.eq_type <- split(dataset, dataset$EQ_TYPE)
l2_levels <- c("BR","MAXMIN","NASH","L1BR","L1MAXMIN")
dataset.split.l1l2 <- list()
for (l in l2_levels){
  dataset.split.l1l2[[l]] <- subset(dataset, grepl(paste("^",l,"\\|*", sep = ""), EQ_TYPE))
  dataset.split.l1l2[[l]]$TRAJ_TYPE <- NA
  dataset.split.l1l2[[l]]$L3_EQ <- NA
}

# Assign the correct level-3 metamodel and sampling scheme to the dataset rows.

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

# For the l1l2 levels where significant difference in l3_eq type was found (Dunn's test), create glm model to analyse the impact of the l3_eq on the precision parameter.
# Observe that the estimates are negative when response is L3_EQMAXMIN

res.glm.l1l2.list.rq3 <- list()
for (l in l2_levels){
  #res.glm.l1l2.list.rq3[[l]] <- glm(ON_EQ ~ . - TASK - ACTIONS - EQ_TYPE, family = Gamma(), data=dataset.split.l1l2[[l]])
  res.glm.l1l2.list.rq3[[l]] <- glm(ON_EQ ~ TRAJ_TYPE + L3_EQ, family = Gamma(), data=dataset.split.l1l2[[l]])
}

for (l in l2_levels){
  print(l)
  print(summary(res.glm.l1l2.list.rq3[[l]], dispersion = 1))
}
