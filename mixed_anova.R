
#d_l = read.csv('H:\\BABY\\results\\perf_raw\\df_l.csv')
#d_l = read.csv('H:\\BABY\\results\\perf_raw\\df_l.csv')
#d_l = read.csv('H:\\BABY\\results\\stat\\df_l.csv')
d_l = read.csv('H:\\BABY\\results\\old\\stat\\df_l_eps.csv')
d_w = read.csv('H:\\BABY\\results\\stat\\df_w.csv')

if(!require(psych)){install.packages("psych")}
if(!require(nlme)){install.packages("nlme")}
if(!require(car)){install.packages("car")}
if(!require(multcompView)){install.packages("multcompView")}
if(!require(lsmeans)){install.packages("lsmeans")}
if(!require(ggplot2)){install.packages("ggplot2")}
if(!require(rcompanion)){install.packages("rcompanion")}
if(!require(rcompanion)){install.packages("sjPlot")}
library(lme4)
library(effects)
library(ggplot2)
library(ez)
library(nlme)
library(multcomp)
library(car)
library(lattice)

d_l = d_l[complete.cases(d_l), ]

d_l$time_id = factor(d_l$time_id,
                          levels=unique(d_l$time_id))
d_l$stag = factor(d_l$stag,
                     levels=unique(d_l$stag))

ggplot(data=d_l, aes(x=stag, y=value, fill = time_id)) + 
  geom_boxplot() 
 
fit_aov <- aov(value ~ stag + time_id + stag:time_id, data=d_l)
layout(matrix(c(1,2,3,4),2,2)) 
plot(fit_aov) # diagnostic plots
anova(fit_aov) #type I SS
Anova(fit_aov, type="II") 
#MIXED MODEL
#plot data with a plot per person including a regression line for each
xyplot(value ~ time_id|name_id_short, groups=stag, type= c("p", "r"), data=d_l)

m <- lmer(value ~ time_id + stag + time_id:stag + (1 | name_id_short), data=d_l, REML=FALSE)
m = lmer(value ~ time_id*stag + (1|name_id_short) + (1|time_id:name_id_short) + (1|stag:name_id_short), data=d_l)  
m = lmer(value ~ (1|name_id_short) + (1|time_id:name_id_short) + (1|stag:name_id_short), data=d_l)  
summary(m)
require(car)
Anova(m)



#ef <- effect("time_id:stag", m)
#summary(ef)
#plot estimates
#x <- as.data.frame(ef)
#ggplot(x, aes(stag, fit, color=time_id)) + geom_point() + geom_errorbar(aes(ymin=fit-se, ymax=fit+se), width=0.4) + theme_bw(base_size=12)


d_l$SHD<-interaction(d_l$time, d_l$stag) #add interaction col
m2 <- lme(value~time_id+stag+time_id:stag, random=~1 | name_id_short, data=d_l) #equivalent with m
#m2 <- lme(value~time_id+stag+time_id:stag, random=~time_id | name_id_short, data=d_l) # random slope for each person
summary(m2)
#check for signinficance of main, save
Anova(m2)
#capture.output(Anova(m2), file='anova.csv')
# posthocs with tukey hsd, takes care of mcp
#s = summary(glht(m2, linfct=mcp(stag="Tukey")))
#capture.output(s, file='time_post.csv')
#posthocs all comprisons
m2_posthoc <- lme(value~SHD, random=~1 | name_id_short, data=d_l)
m2_comp<-glht(m2_posthoc,mcp(SHD='Tukey'))
summary(m2_comp)



library(plyr)
d_l$stag = revalue(d_l$stag, c("1"="NREM", "2"="REM", "3"="WAKE"))
d_l$time_id = revalue(d_l$time_id, c("1"="2_weeks", "2"="5_weeks"))
#VISUALIZATION
library(ggpubr)
compare <- list( c("1", "2"), c("1", "3"), c("2", "3") )
p <- ggboxplot(d_l, x = "stag", y = "value",
               color = "time_id", palette = "Dark2",
               add = "jitter",
               size =1.5,
               notch=FALSE,  ylim = c(1.3, 1.6))
p  <- p + theme(legend.title=element_blank(),
               axis.title.x=element_blank())
p  <- p + ylab('Permutation entropy')

# Add p-value
#path <- data.frame(x=c(1,1,2,2), y=c(1.55,1.56,1.56,1.55))
#path2 <- data.frame(x=c(2,2,3,3), y=c(1.57,1.58,1.58,1.57))
#path3 <- data.frame(x=c(1,1,3,3), y=c(1.59,1.61,1.61,1.59))

#p <- p + geom_path(data=path,aes(x=x, y=y))
#p <- p + geom_path(data=path2,aes(x=x, y=y))
#p <- p + geom_path(data=path3,aes(x=x, y=y))
#p <- p + geom_text(aes(x = 1.5, y= 1.57, label = "*")) 
#p <- p + geom_text(aes(x = 2.5, y= 1.59, label = "ns")) 
#p <- p + geom_text(aes(x = 2, y= 1.62, label = "*")) 
p <- p + geom_text(aes(x = 1, y= 1.56, label = "*"), size = 12) 
p <- p + geom_text(aes(x = 2, y= 1.56, label = "*"), size = 12) 
p


#plot box t1 vs t2 and significance
t <- ggboxplot(d_l, x = "time_id", y = "value",
               color = "time_id", palette = "lancet",
               add = "jitter",
               size =1.5,
               notch=FALSE,  ylim = c(1.3, 1.65))
t  <- t + theme(legend.title=element_blank(),
                axis.title.x=element_blank())
path4 <- data.frame(x=c(1,1,2,2), y=c(1.59,1.6,1.6,1.59))
t <- t + geom_path(data=path4,aes(x=x, y=y))
t <- t + geom_text(aes(x = 1.5, y= 1.62, label = "*")) 
t  <- t + ylab('Permutation entropy')
t