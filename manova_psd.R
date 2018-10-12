#install.packages('mvnormtest')
library(mvnormtest)

df = read.csv('H:\\BABY\\results\\stat\\df_psd_new.csv')

#split df to NREM, REM and WAKE
#d_ = as.matrix(df[, c(2:70)])
d_ = as.matrix(df[, c(2:60)])
n_bins = dim(d_)[2]
nremdf = df[df$stag == 'nrem',]
remdf = df[df$stag == 'rem', ]
wakedf = df[df$stag == 'wake',]

seq_sub = seq(1, n_bins, 3) #sample bins to reduce correlations 

# functions accepts df, separatley ss or full data
my_manova <- function(data) {
  #fit <- manova(as.matrix(data[, c(2:70)][seq_sub]) ~ data$time)
  fit <- manova(as.matrix(data[, c(2:60)][seq_sub]) ~ data$time)
  s = summary(fit, test = "Pillai")
  aovs = summary.aov(fit) #1 way anova 
  
  #extract ANOVA p val for each freq bin
  pvals = list()
  for (i in 1: length(aovs)) {
    pvals[i] = unlist(aovs[i])[9]
  }
  return(list(s, aovs, pvals))
}

# adjust for multiple comp, #Benjamini & Hochberg (1995) correction, input to my_correct
# is uncorrected pvs
my_correct <- function(data) {
  #colnames(data) = 'uncor_pval'
  data$corrected = p.adjust(data$uncor_pval, method = 'BH', n = length(data$uncor_pval) + 1) 
  data$signif = data$corrected <= 0.05
  data$sum_sign = sum(data$signif)
  return(data)
}

store_F = list()
i = 0
dd = list(nremdf, remdf, wakedf)
for (d in dd) {
  i = i + 1
  store_F[i] = my_manova(d)[1]
}

#insert manualy (!) pvals from store_F 
pval_manovas = data.frame(c(0.02182, 0.007696, 0.04624))
#correct for m.c.
my_correct(pval_manovas)

my_pvals_stag <- function(data) {
  #get aovs and corrected pv for given ss.
  pvals = my_manova(data)[3]
  pvals = data.frame(unlist(pvals))
  colnames(pvals) = 'uncor_pval'
  res = my_correct(pvals)
  return(res)
}
save_path = 'H:\\BABY\\results\\stat\\'
nrempv = my_pvals_stag(nremdf)
rempv = my_pvals_stag(remdf)
wakepv = my_pvals_stag(wakedf)

write.csv(nrempv, paste(save_path,'nrempv.csv',sep=""))
write.csv(rempv, paste(save_path,'rempv.csv',sep=""))
write.csv(wakepv, paste(save_path,'wakepv.csv',sep=""))





