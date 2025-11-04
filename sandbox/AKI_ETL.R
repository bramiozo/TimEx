install.packages(c(
  "dplyr", "tidyr", "finalfit", "writexl", "lcmm",  "NormPsy",  "arrow",
  "ggplot2",  "gridExtra",  "nlme",  "splines2",  "zoo",  "doParallel",
  "progressr",  "data.table",  "stringr",  "survival",  "survminer",
  "broom", "conflicted", "nephro", "meta", "rmarkdown", "pander", "knitr", "gt",
  "tidyverse", "sas7bdat", "tableone", "patchwork", "nricens", "stringi", "plotly"))

library(dplyr)
library(tidyr)
library(arrow)
library(data.table)

# load Rdata into dataframe
library(arrow)
library(sas7bdat)
# 'L:\\lab_research\\RES-Folder-UPOD\\NOSTRADAMUS_SALTRO\\E_ResearchData\\2_ResearchData\\CLEANED_for_Methods_paper_longitudinal_analysis\\13032025\\data_preparation_DV_final_for_LCMM_15257AKIpatients_FU7-3650d_SW365d_TR365d_13032025.Rdata'
load("T:\\lab_research\\RES-Folder-UPOD\\NOSTRADAMUS_SALTRO\\E_ResearchData\\2_ResearchData\\CLEANED_for_Methods_paper_longitudinal_analysis\\28102025\\ANALYSIS\\BRAM\\cleaned_DV_LCMM_data.Rdata")
write_parquet(LCMM_data, "T:\\lab_research\\RES-Folder-UPOD\\NOSTRADAMUS_SALTRO\\E_ResearchData\\2_ResearchData\\CLEANED_for_Methods_paper_longitudinal_analysis\\28102025\\ANALYSIS\\BRAM\\cleaned_DV_LCMM_data.parquet")


#load("L:\\lab_research\\RES-Folder-UPOD\\NOSTRADAMUS_SALTRO\\E_ResearchData\\2_ResearchData\\CLEANED_for_Methods_paper_longitudinal_analysis\\13032025\\cleaned_DV_final.Rdata")
#write_parquet(data, "L:\\lab_research\\RES-Folder-UPOD\\NOSTRADAMUS_SALTRO\\E_ResearchData\\2_ResearchData\\CLEANED_for_Methods_paper_longitudinal_analysis\\13032025\\cleaned_DV_final.parquet")
