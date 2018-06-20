#The right script! similar to mixed_anova (later version) BUT runs on equally sampled and averged across epochs 
library(lme4)
require(car)
library(nlme)
library(multcomp)
library(plyr)
library(ggpubr)
d_l = read.csv('H:\\BABY\\results\\stat\\df_l_eps.csv')

d_l$time_id = factor(d_l$time_id,
                     levels=unique(d_l$time_id))
d_l$stag = factor(d_l$stag,
                  levels=unique(d_l$stag))

m1 = lmer(value ~ time_id*stag  + (1|name_id_short) + (1|stag), data=d_l)
m2 = lmer(value ~ time_id * stag + (1+time_id|name_id_short), data=d_l) #THE model

# strip plot; random effects crossed
m3 = lmer(value~time_id * stag + (1|name_id_short) + (1|time_id:name_id_short) + (1|stag:name_id_short), data=d_l)  

m4 = lmer(value~time_id * stag + (1|name_id_short) + (1|time_id:name_id_short) , data=d_l)  
m5 = lmer(value~time_id * stag + (1|name_id_short) + (1|stag:name_id_short) , data=d_l)  
summary(m4)

Anova(m2)
anova(m1, m2, m3, m4, m5)

#plot residuals
#plot(fitted(m2), resid(m2))

#post hoc tests
d_l = d_l[complete.cases(d_l), ] #drop nans
d_l$SHD<-interaction(d_l$time, d_l$stag) #add interaction co
#equivalent with m2, nlme library
m2_posthoc <- lme(value~SHD, random=~time_id | name_id_short, data=d_l)
m2_comp<-glht(m2_posthoc,mcp(SHD='Tukey'))
summary(m2_comp)


#Visualisation
d_l$stag = revalue(d_l$stag, c("1"="NREM", "2"="REM", "3"="WAKE"))
d_l$time_id = revalue(d_l$time_id, c("1"="2_weeks", "2"="5_weeks"))
d_l$stag = factor(d_l$stag, levels = c('NREM','REM', 'WAKE'),ordered = TRUE)
p <- ggboxplot(d_l, x = "stag", y = "value",
               color = "time_id", palette = "Dark2",
               size =1.5,
               add='jitter',
               notch=FALSE,  ylim = c(1.3, 1.6))
p  <- p + theme(legend.title=element_blank(),
                axis.title.x=element_blank())
p <- p + ylab('Permutation entropy')
 

p <- p + geom_text(aes(x = 1, y= 1.56, label = "*"), size = 12) 
p <- p + geom_text(aes(x = 2, y= 1.56, label = "*"), size = 12) 
p <- p + geom_text(aes(x = 3, y= 1.57, label = "ns"), size = 6) 
p


# plot observed vs predicted and r2
library(MuMIn) #for R squered
fo_df = data.frame(fitted(m2), d_l$value)
colnames(fo_df) <- c('predicted_PE', 'observed_PE')
with(fo_df, plot(observed_PE, predicted_PE))
abline(a=0, b=1)

legend("topleft", bty="n", legend=paste("R2 =", format(r.squaredGLMM(m2)[2], digits=4)))






