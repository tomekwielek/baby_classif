library(lme4)
require(car)
library(nlme)
library(multcomp)
library(plyr)
library(ggpubr)
library(ggsignif)
library(r2glmm)

library(rlme)
d_l = read.csv('H:\\BABY\\results\\figs\\mspe_final\\factorchannels_mspe_scale5.csv')

d_l = d_l[complete.cases(d_l), ]
#d_l = d_l[d_l$channels == 'front', ]

names(d_l)[names(d_l) == 'time'] <- 'time_id'
names(d_l)[names(d_l) == 'sbj_id'] <- 'name_id_short'

d_l$time_id = factor(d_l$time_id,
                     levels=unique(d_l$time_id))
d_l$stag = factor(d_l$stag,
                  levels=unique(d_l$stag))

d_l$channels = factor(d_l$channels,
                  levels=unique(d_l$channels))

m1 = lmer(value ~ time_id * stag * channels + (1+1|name_id_short), data=d_l) 

m2 = lmer(value ~ time_id * stag * channels + (1+time_id|name_id_short), data=d_l) #LMER channels incuded

#m2 = lmer(value ~ time_id * stag  + (1+time_id|name_id_short), data=d_l) #LMER NO channels
Anova(m2)

#Anova(m1)

#anova(m1, m2) #compare models
 
#m2_lme = lme(value~time_id * stag * channels, random=~time_id | name_id_short, data=d_l) # LME
#Anova(m2_lme)

#post hoc tests for stag
m2_posthoc_stag <- lme(value~stag, random=~time_id | name_id_short, data=d_l)
m2_comp_stag <-glht(m2_posthoc_stag,mcp(stag='Tukey'))
summary(m2_comp_stag)


#post hoc tests for interaction  (sleep stage x time )
d_l = d_l[complete.cases(d_l), ] #drop nans
d_l$SHD<-interaction(d_l$time, d_l$stag) #add interaction co
m2_posthoc <- lme(value~SHD, random=~time_id | name_id_short, data=d_l)
m2_comp<-glht(m2_posthoc,mcp(SHD='Tukey'))
summary(m2_comp)


#post hoc tests for interaction  (sleep stage x location )
d_l = d_l[complete.cases(d_l), ] #drop nans
d_l$SHD2<-interaction(d_l$stag, d_l$channels) #add interaction co
m2_posthoc2 <- lme(value~SHD2, random=~time_id | name_id_short, data=d_l)
m2_comp<-glht(m2_posthoc2,mcp(SHD2='Tukey'))
summary(m2_comp)

#post hoc tests for interaction  (session x location )
d_l = d_l[complete.cases(d_l), ] #drop nans
d_l$SHD3<-interaction(d_l$time_id, d_l$channels) #add interaction co
m2_posthoc3 <- lme(value~SHD3, random=~time_id | name_id_short, data=d_l)
m3_comp<-glht(m2_posthoc3,mcp(SHD3='Tukey'))
summary(m3_comp)

#post hoc tests for interaction  (time x channels )
d_l = d_l[complete.cases(d_l), ] #drop nans
d_l$SHD4<-interaction(d_l$time, d_l$channels) #add interaction co
m2_posthoc4 <- lme(value~SHD4, random=~time_id | name_id_short, data=d_l)
m4_comp<-glht(m2_posthoc4,mcp(SHD4='Tukey'))
summary(m4_comp)

#post hoc tests for channels
m2_posthoc_channels <- lme(value~channels, random=~time_id | name_id_short, data=d_l)
m2_comp_channels <-glht(m2_posthoc_channels,mcp(channels='Tukey'))
summary(m2_comp_channels)

# plot observed vs predicted and r2
library(MuMIn) #for R squered
fo_df = data.frame(fitted(m2), d_l$value)
colnames(fo_df) <- c('predicted_PE', 'observed_PE')
with(fo_df, plot(observed_PE, predicted_PE))
abline(a=0, b=1)

legend("topleft", bty="n", legend=paste("R2 =", format(r.squaredGLMM(m2)[2], digits=4)))





