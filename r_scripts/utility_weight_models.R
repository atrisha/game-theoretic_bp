library(tidyverse)
library(broom)
library(purrr)
library(dplyr)
library(sjPlot)
library(rstatix)
library(Matrix)
library(ggplot2)


dataset <- read.table("F:\\Spring2017\\workspaces\\game_theoretic_planner\\r_scripts\\input_files\\769\\UTILITY_WEIGHTS.csv", header=TRUE, sep=",")
dataset$SEGMENT <- as.factor(dataset$SEGMENT)
dataset$SIGNAL <- as.factor(dataset$SIGNAL)
dataset$NEXT_SIGNAL <- as.factor(dataset$NEXT_SIGNAL)
dataset$LEAD_VEHICLE <- as.factor(dataset$LEAD_VEHICLE)
dataset$RELEV_VEHICLE <- as.factor(dataset$RELEV_VEHICLE)
dataset$PEDESTRIAN <- as.factor(dataset$PEDESTRIAN)
dataset$SPEED_MPS <- as.factor(dataset$SPEED_MPS)
dataset$TIME_TO_NEXT_SIGNAL <- as.factor(dataset$TIME_TO_NEXT_SIGNAL)

veh_inh_low_model_data <- select(dataset, SIGNAL , NEXT_SIGNAL , TIME_TO_NEXT_SIGNAL , SEGMENT , SPEED_MPS , LEAD_VEHICLE , RELEV_VEHICLE , PEDESTRIAN, VEH_INH_LOW)
veh_inh_low_model <- glm(VEH_INH_LOW ~ SIGNAL + NEXT_SIGNAL + TIME_TO_NEXT_SIGNAL + SEGMENT + SPEED_MPS + LEAD_VEHICLE + RELEV_VEHICLE + PEDESTRIAN, family = binomial, data=veh_inh_low_model_data)
summary(veh_inh_low_model)
veh_inh_low_model_data[1870:1892,]
p<- predict(veh_inh_low_model, newdata = veh_inh_low_model_data[1870:1892,], type="response")
print(data.frame(p),)
s <- data.frame(veh_inh_low_model_data[1870:1892,]$VEH_INH_LOW)
s$predicted <- data.frame(p)
s$err <- abs(s$veh_inh_low_model_data.1870.1892....VEH_INH_LOW - s$predicted)*100

veh_inh_high_model_data <- select(dataset, SIGNAL , NEXT_SIGNAL , TIME_TO_NEXT_SIGNAL , SEGMENT , SPEED_MPS , LEAD_VEHICLE , RELEV_VEHICLE , PEDESTRIAN, VEH_INH_HIGH)
veh_inh_high_model <- glm(VEH_INH_HIGH ~ SIGNAL + NEXT_SIGNAL + TIME_TO_NEXT_SIGNAL + SEGMENT + SPEED_MPS + LEAD_VEHICLE + RELEV_VEHICLE + PEDESTRIAN, family = binomial, data=veh_inh_high_model_data)
summary(veh_inh_high_model)
veh_inh_high_model_data[1870:1892,]
p<- predict(veh_inh_high_model, newdata = veh_inh_high_model_data[1870:1892,], type="response")
s <- data.frame(veh_inh_high_model_data[1870:1892,]$VEH_INH_HIGH)
s$predicted <- data.frame(p)
s$err <- abs(s$veh_inh_high_model_data.1870.1892....VEH_INH_HIGH - s$predicted)*100

rule_model_data <- select(dataset, SIGNAL , NEXT_SIGNAL , TIME_TO_NEXT_SIGNAL , SEGMENT , SPEED_MPS , LEAD_VEHICLE , RELEV_VEHICLE , PEDESTRIAN, RULE_HIGH)
rule_model <- glm(RULE_HIGH ~ SIGNAL + NEXT_SIGNAL + TIME_TO_NEXT_SIGNAL + SEGMENT + SPEED_MPS + LEAD_VEHICLE + RELEV_VEHICLE + PEDESTRIAN, family = binomial, data=rule_model_data)
summary(rule_model)
rule_model_data[1870:1892,]
p<- predict(rule_model, newdata = rule_model_data[1870:1892,], type="response")
s <- data.frame(rule_model_data[1870:1892,]$RULE_HIGH)
s$predicted <- data.frame(p)
s$err <- abs(s$rule_model_data.1870.1892....RULE_HIGH - s$predicted)*100


