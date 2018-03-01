import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
path = 'H:\\BABY\\results\\perf_raw\\'
perf1 = pickle.load(open(path+'perf_t1.txt', 'rb'))
perf2 = pickle.load(open(path+'perf_t2.txt', 'rb'))
perf_dummy1 = pickle.load(open(path+'perf_dummy_t1.txt', 'rb'))
perf_dummy2 = pickle.load(open(path+'perf_dummy_t2.txt', 'rb'))

def get_df(dict_):
    df1_ = pd.DataFrame({'corr': dict_.values()[0]})
    df2_ = pd.DataFrame({'uncorr':dict_.values()[1]})
    df = pd.concat([df1_, df2_], ignore_index=False,axis=1 )
    df =  df.melt(value_vars =['corr', 'uncorr'], var_name='if_corr')
    return df

df1 = get_df(perf1)
df2 = get_df(perf2)
df1['time'] = ['t1'] * len(df1)
df2['time'] = ['t2'] * len(df2)

df_dummy1 = get_df(perf_dummy1)
df_dummy2 = get_df(perf_dummy2)

def get_box(my_df, my_dummy_df, ax, t):
    sns.boxplot(my_df['if_corr'], my_df['value'], ax = ax)
    sns.stripplot(my_df['if_corr'], my_df['value'], jitter=True, color='black', ax=ax)
    x_ = np.unique(my_dummy_df['if_corr'])
    median = my_dummy_df.groupby('if_corr').median().values
    ax.plot( median, 'r*', markersize=8 )
    ax.set_ylabel('Accuracy [%]')
    ax.set_title(t)
    plt.show()
fig, ax = plt.subplots(1,2, sharey= True)
subtit = ['t1', 't2']
for d, dm, ax_, t in zip([df1, df2], [df_dummy1, df_dummy2], ax, subtit):
    get_box(d, dm, ax_, t)


#plot t1, t2 and corr, uncorr
df12 = pd.concat([df1, df2], ignore_index=True)
sns.factorplot(x='if_corr', hue='time', y='value', data=df12,ci=95, kind='point')
