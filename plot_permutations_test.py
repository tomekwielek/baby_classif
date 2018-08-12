import matplotlib.pyplot as plt
from functional import read_pickle
data_path = 'E:\\BABY\\results\\perf_raw\\'
from matplotlib.patches import Rectangle

f1 = 'sign_t1_mspe_weightedF1.txt'
f2 = 'sign_t2_mspe_weightedF1.txt'
f3 = 'sign_t1t2_mspe_weightedF1.txt'
f4 = 'sign_t2t1_mspe_weightedF1.txt'

#t1, t2, t1t2, t2t1 = [read_pickle(data_path+fname) for fname in [f1, f2, f3, f4]]


t1, t2, t1t2, t2t1 = [read_pickle(data_path+fname) for fname in [f1, f2, f3, f4]]
grs = ['week2', 'week5', 'week2->week5', 'week5->week2']


def plot_perf_stat(data, title=None): #data[0] is perf, data[1] is null dist
    plt.figure()
    plt.hist(data[1], bins=30, edgecolor='black', linewidth=0.8, color='white')
    av_perf = data[0].mean()
    plt.axvline(x=av_perf, linewidth=2.2, color='red')
    plt.title(title)
    plt.xlim(0.2,0.8)
    plt.ylabel('N bootstraps')
    plt.xlabel('Classifer performance [f1-score]')
    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['black','red']]
    labels= ["randomized data","'real' data"]
    plt.legend(handles, labels)
    plt.show()
#plot_perf_stat(t1, 'Week2')

dd= zip([t1, t2, t1t2, t2t1], grs)


[plot_perf_stat(data=data, title=title) for data, title in dd]
