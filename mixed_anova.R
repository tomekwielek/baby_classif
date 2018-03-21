
d_l = read.csv('H:\\BABY\\results\\perf_raw\\df_l.csv')

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
#rANOVA for unbalanced 

#plot data with a plot per person including a regression line for each
xyplot(value ~ time_id|name_id_short, groups=stag, type= c("p", "r"), data=d_l)

#Mixed effects modelling

m <- lmer(value ~ time_id + stag + time_id:stag + (1 | name_id_short), data=d_l, REML=FALSE)
coef(summary(m))
ef <- effect("time_id:stag", m)
summary(ef)
#plot estimates
x <- as.data.frame(ef)
ggplot(x, aes(stag, fit, color=time_id)) + geom_point() + geom_errorbar(aes(ymin=fit-se, ymax=fit+se), width=0.4) + theme_bw(base_size=12)

#compare differet package
d_l$SHD<-interaction(d_l$time, d_l$stag) #add interaction col, not used though
m2 <- lme(value~time_id+stag+time_id:stag, random=~1 | name_id_short, data=d_l) #equivalent with m
summary(m2)
#check for signinficance of main, save
Anova(m2)
capture.output(Anova(m2), file='anova.csv')
# posthocs

s = summary(glht(m2, linfct=mcp(time_id="Tukey")))
capture.output(s, file='time_post.csv')



#fit model without inter and compare 
m3 <- lme(value~time_id+stag, random= ~1 | name_id_short,data=d_l) 
summary(m3)


anova(m2, m3)
m4 <- lmer(value ~ time_id + stag  + (1 | name_id_short), data=d_l)
AIC(m2, m4) #intereaction gets lower AIC












#VISUALIZATION
library(ggpubr)
compare <- list( c("1", "2"), c("1", "3"), c("2", "3") )
p <- ggboxplot(d_l, x = "stag", y = "value",
               color = "time_id", palette = "jco",
               add = "jitter",
               notch=FALSE,  ylim = c(1.3, 1.65))
# Add p-value
path <- data.frame(x=c(1,1,2,2), y=c(1.55,1.56,1.56,1.55))
path2 <- data.frame(x=c(2,2,3,3), y=c(1.57,1.58,1.58,1.57))
path3 <- data.frame(x=c(1,1,3,3), y=c(1.59,1.61,1.61,1.59))

p <- p + geom_path(data=path,aes(x=x, y=y))
p <- p + geom_path(data=path2,aes(x=x, y=y))
p <- p + geom_path(data=path3,aes(x=x, y=y))
p <- p + geom_text(aes(x = 1.5, y= 1.57, label = "*")) 
p <- p + geom_text(aes(x = 2.5, y= 1.59, label = "ns")) 
p <- p + geom_text(aes(x = 2, y= 1.62, label = "*")) 
p


