
acc25 = read.csv('H:\\BABY\\results\\acc_week25_MW_R.csv')
f125 = read.csv('H:\\BABY\\results\\f1_week25_MW_R.csv')

#acc
mw = wilcox.test(acc ~ time, data=acc25)
z = qnorm(mw$p.value) #  score

#f1 scores
mw = wilcox.test(WAKE ~ time, data=f125)
z = qnorm(mw$p.value) #  score
mw
z

