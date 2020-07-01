# Load result files into dataset
setwd("F:\\Spring2017\\workspaces\\game_theoretic_planner\\results_all")
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

# RQ1: Which solution concept better predicts empirical driving behavior.
# Run Kruskal Walis test to check that the difference in EQ_TYPE group in significant

res.kruskal <- kruskal.test(ON_EQ ~ EQ_TYPE, data = dataset)

# Run Dunn's post hoc test on significant Kruskal Walis result to test the significant between pairwise comparison of the group factors.

library(rstatix)
res.dunn <- dunn_test(dataset, ON_EQ ~ EQ_TYPE, p.adjust.method="holm", detailed=TRUE)



