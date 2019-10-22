# Overview

We analyzed sample of newborn EEG data at week 2 and 5 after birth. There were two aims of the project:
1.	Describing entropy & PSD changes with sleep states and age in newborns
2.	Automatic classification of such states
We observed developmental changes limited to “quiet” NREM and “active” REM sleep. Classification accuracy was of 60% at week-2, 73% at week-5 (while estimated chance level of  33%) using multi-scale permutation entropy (6 EEG and 5 physiological channels).

## References

## Year 
2018

# Main Functionalities

**1_read_raw.py** This function goes through every subject and loads both raw EDF data and corresponding CSV sleep staging. Data are segmented into 30s segments and saved as epochs (staging converted to events). Data is saved in the epoch repository. This function is run with one of the following argument values:

 - 'St_Pr_corrected_27min' 
 - 'St_Pr_corrected_35min'

**2_mspe_psd_from_raw.py** This function goes through every subject and loads both raw EDF data and corresponding CSV sleep staging. Data are segmented into 30s segments. MSPE (multi scale permutation entropy) and PSD (power spectrum) are computed. Data is saved in the mspet1m3 and psd folders. Data is later used for classification. This function is run with two of the following argument values:
 - 'St_Pr_corrected_27min' 
 -  'St_Pr_corrected_35min'
 - 'psd'
 -  'mspet1m3'

 **3_run_mspe_from_epochs.py** This script goes through every subject and loads epochs data. Data is saved in the mspe_from_epochs repository.  Data is later used for statistical comparison and boxplots.
**4_run_plot_mspe.py** This script goes through every subject and loads MSPE data (mspe_from_epochs). Data is aggregated using random re-sampling, converted to pandas data frame, saved as CSV (later used for statistics). Additionally box-plots are created. This script is run with following configuration setups: what MSPE scale to use, what channels to use, whether to  drop outliers.
**5_mixed_anova.R** This R script is used for main statistical analysis (linear mixed model and posthoc test for MSPE data).
**6_run_plot_mspe_across_scales.py** This script plots MSPE values across 5 scales, 3 sleep stages (NREM, REM , wake) and 2 sessions (week2, week5).  This script is run with following configuration setups: what MSPE scale to use, what channels to use, whether to  drop outliers
**7_run_plot_psd.py** This script goes through every subject and loads epochs data. Next following steps are implemented: PSD computation, aggregation epochs by using random re-sampling, plotting PSD with cluster permutation statistics. This script is run with following configuration setups: what channels to use.
**8_main.py** This script classifies within sessions.  This script is run with following configuration setups: what data to use (PSD or MSPE), what sessions to classify, (week-2, week-5 or merged), what MSPE scale to use, whether to search for optimal hyperparameters of the classifier, whether to estimate chance level.
**9_main_crosstime.py** This script classifies across sessions.  This script is run with following configuration setups: what session to use as test (week-2 or week-5), what MSPE scale to use, whether to search for optimal hyperparameters of the classifier, whether to estimate chance level. 
**10_run_compare_mspe_psd_scores.py** Script loads classification scores for PSD and MSPE data, next plot confusion matrix and box plots for both.
**11_run_test_sleep_distribution_changes.py** Script tests statistically whether duration of sleep stages differ across sessions
**12_run_plot_scores.py** Plot classification scores for MSPE data including cross classification and null distribution (accuracy and F1 per sleep stage as upper and lower panel respectively)

- - Auxiliary functions/scripts:
	**config.py** Various configurations (e.g. bad subjects, channels names), also functions to manipulate paths (e.g. saving, loading) 
	**classify_with_shuffling.py** – used by 8_main.py to classify within session
	**classify_with_shuffling_crosstime.py** – used by 9_main_crosstime.py to classify cross sessions
	**functional.py** Various utility functions (e.g. removing 20Hz artifacts, selecting given sleep classes)
	**plot_hd_topos.py** Plot high density plots for PSD data 
**plot_dec_bound.py** Plot decision boundary for the classifer
	

	
	

