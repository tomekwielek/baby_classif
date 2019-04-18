#The right script! similar to mixed_anova (later version) BUT runs on equally sampled and averged across epochs 
# csv is na output of wrap_mspe_psd_v2.py
library(lme4)
require(car)
library(nlme)
library(multcomp)
library(plyr)
library(ggpubr)
library(ggsignif)
d_l = read.csv('H:\\BABY\\results\\mspe_allsbjs_alleeg_10epochs.csv')

d_l = d_l[complete.cases(d_l), ]

d_l = subset(d_l, variable == 4) #if mspe select scale

#aggregate by time and stage
aggregate(value~stag*time, data=d_l, FUN=sd)

#d_s0 = subset(d_l, variable == 0) #scale as factor
#d_s5 = subset(d_l, variable == 4) #scale as factor
#d_l = rbind(d_s0, d_s5) #scale as factor
  
names(d_l)[names(d_l) == 'time'] <- 'time_id'
names(d_l)[names(d_l) == 'sbj_id'] <- 'name_id_short'

# Frequency count
xtabs(~ stag + name_id_short, d_l)

d_l$time_id = factor(d_l$time_id,
                     levels=unique(d_l$time_id))
d_l$stag = factor(d_l$stag,
                  levels=unique(d_l$stag))
d_l$variable = factor(d_l$variable,
                  levels=unique(d_l$variable))

boxplot(value ~ time_id * stag * variable,
        col=c("white","lightgray"),d_l)

m1 = lmer(value ~ time_id * stag + (1+1|name_id_short), data=d_l) 

m2 = lmer(value ~ time_id * stag + (1+time_id|name_id_short), data=d_l) 

#m2 = lmer(value ~ time_id * stag * variable + (1+time_id|name_id_short), data=d_l)#scale as factor

# strip plot; random effects crossed
#m3 = lmer(value~time_id * stag + (1|name_id_short) + (1|time_id:name_id_short) + (1|stag:name_id_short), data=d_l)  

#m4 = lmer(value~time_id * stag + (1|name_id_short) + (1|time_id:name_id_short) , data=d_l)  

#m5 = lmer(value~time_id * stag + (1|name_id_short) + (1|stag:name_id_short) , data=d_l)  
#summary(m4)

Anova(m2)
anova(m1, m2)

#plot residuals
#plot(fitted(m2), resid(m2))

#post hoc tests
d_l = d_l[complete.cases(d_l), ] #drop nans
d_l$SHD<-interaction(d_l$time, d_l$stag) #add interaction co
#d_l$SHD<-interaction( d_l$stag) 
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


#plot significnat contrast with 'ggsignif' 
ggCustom <- function(d_l, stag, i, pvalues, pvalues1, pvalues2, pvalues3)
{
  require(ggplot2)
  
  g <- ggplot(ScatterMatrix, aes(x=Module, y=as.numeric(ScatterMatrix[,i]))) +
    
    #Add a boxplot
    #Control how outliers are managed using extra parameters
    geom_boxplot(position=position_dodge(width=0.5), outlier.shape=17, outlier.colour="red", outlier.size=0.1, aes(fill=Module)) +
    
    #Choose which colours to use; otherwise, ggplot2 choose automatically
    #scale_color_manual(values=c("red3", "white", "blue")) + #for scatter plot dots
    #scale_fill_manual(values=c("royalblue", "pink", "red4")) + #for boxplot
    scale_fill_grey(start=0.5, end=0.9) +
    
    stat_summary(geom="crossbar", width=0.8, fatten=2, color="black", fun.data=function(x){return(c(y=median(x), ymin=median(x), ymax=median(x)))}) +
    
    #Add the scatter points (treats outliers same as 'inliers')
    geom_jitter(position=position_jitter(width=0.3), size=0.25, colour="black") +
    
    #Set the size of the plotting window
    theme_bw(base_size=24) +
    
    #Modify various aspects of the plot text and legend
    theme(
      legend.position="none",
      legend.background=element_rect(),
      plot.title=element_text(angle=0, size=12, face="bold", vjust=1),
      
      axis.text.x=element_text(angle=0, size=12, face="bold", vjust=0.5),
      axis.text.y=element_text(angle=0, size=12, vjust=0.5),
      axis.title=element_text(size=12),
      
      #Legend
      legend.key=element_blank(),     #removes the border
      legend.key.size=unit(1, "cm"),  #Sets overall area/size of the legend
      legend.text=element_text(size=8),   #Text size
      title=element_text(size=8)) +       #Title text size
    
    #Change the size of the icons/symbols in the legend
    guides(colour=guide_legend(override.aes=list(size=2.5))) +
    
    #Set x- and y-axes labels
    xlab("Cluster group") +
    ylab("Vitamin D (ng/ml)") +
    
    ylim(0, 100.0) +
    
    geom_segment(aes(x=1, y=83, xend=2, yend=83), size=0.7, data=ScatterMatrix) +
    geom_segment(aes(x=1, y=83, xend=1, yend=79.0), size=0.7, data=ScatterMatrix) +
    geom_segment(aes(x=2, y=83, xend=2, yend=79.0), size=0.7, data=ScatterMatrix) +
    geom_text(x=1.5, y=87, size=3.0, family="mono", label=pvalues1[i]) +
    
    geom_segment(aes(x=1, y=2.5, xend=3, yend=2.5), size=0.7, data=ScatterMatrix) +
    geom_segment(aes(x=1, y=2.5, xend=1, yend=6.5), size=0.7, data=ScatterMatrix) +
    geom_segment(aes(x=3, y=2.5, xend=3, yend=6.5), size=0.7, data=ScatterMatrix) +
    geom_text(x=2, y=-1.5, size=3.0, family="mono", label=pvalues2[i]) +
    
    geom_segment(aes(x=2, y=96, xend=3, yend=96), size=0.7, data=ScatterMatrix) +
    geom_segment(aes(x=2, y=96, xend=2, yend=92.0), size=0.7, data=ScatterMatrix) +
    geom_segment(aes(x=3, y=96, xend=3, yend=92.0), size=0.7, data=ScatterMatrix) +
    geom_text(x=2.5, y=100, size=3.0, family="mono", label=pvalues3[i]) +
    
    ggtitle(colnames(ScatterMatrix)[i])
  return(g)
}

