import mne
import os
import pandas as pd
import numpy as np
import matplotlib
from config import myload
from IPython.core.debugger import set_trace
from config import paths,  bad_sbjs_1, bad_sbjs_2, mysave
import matplotlib.pyplot as plt
import os
from mne.stats import permutation_cluster_test
import seaborn as sns
from matplotlib.patches import Patch

matplotlib.rcParams.update({'font.size': 12,'xtick.labelsize':8, 'ytick.labelsize':8})
np.random.seed(12)

path = 'H:\\BABY\\working\\subjects'
fnames =  os.listdir(path)
fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

events_id = ['N', 'R', 'W']
events_id_map = {'N':1, 'R':2, 'W':3}
scale_idx = 0

# Set config for plotting (e.g what channels)
pick_chann = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2'] # set what channels mspe was computed for
plot_chann = ['F3', 'F4']

def my_bootstraper(data, ch_indices, repetions=1000, n=10):
    '''
    Bootstraped averaging of n epochs (epochs per sleep stage)
    data[i].shape = n_chs x n_epochs where i=sbj index
    '''
    np.random.seed(None) # randomly initialize the RNG from some platform-dependent source
    store_repet = np.zeros([repetions, len(data)])
    for i in range(repetions):
        store = []
        for d_ in data:
            count = d_.shape[1]
            if count == 0:
                store.append(np.nan)
            else:
                sample = np.min([n, count]) # if n<count mean over count
                epoch_indices = np.random.choice(count, sample)
                av = d_[ch_indices, :] #sample channels
                av = av[:, epoch_indices] #sample epochs
                store.append(av.mean())
        store_repet[i,:] = store
    return store_repet

def build_df(mydict, time='week2'):
    '''
    Build pandas data frame from nested dictionary.
    '''
    mydf = pd.DataFrame.from_dict({(i,j,k): mydict[i][j][k]
                                for i in [time]
                                for j in mydict[i].keys()
                                for k in mydict[i][j].keys()},
                                orient='columns')

    sbj_columns = [l for l in mydf.columns.tolist() if 'sbj' in l]
    sbjs = mydf[sbj_columns]
    mydf = mydf.drop(columns=sbj_columns)
    sbjs['new_idx'] = sbjs[time].apply(lambda x: ' '.join(x.dropna().astype(str).unique()), axis=1)
    mydf = mydf.set_index(sbjs['new_idx'])
    return mydf


# LOAD mspe
store = {'week2':[], 'week5':[]}
store_event = {'week2':[], 'week5':[]}
store_sbjnames = {'week2':[], 'week5':[]}

for data, time, bad in zip([fnames1, fnames2], ['week2', 'week5'], [bad_sbjs_1, bad_sbjs_2]):
    for sbj in data:
        if len(os.listdir(os.path.dirname(paths(typ='mspe_from_epochs', sbj=sbj)))) == 0: #emty folder means bad sbj
            continue
        else:
            stag, mspe = myload(typ='mspe_from_epochs', sbj=sbj)
            mspe = mspe[scale_idx, :, :]
            store[time].append(mspe)
            store_event[time].append(stag)
            store_sbjnames[time].append(sbj)


m = mne.channels.read_montage(kind='standard_1020')
epoch = myload(typ='epoch', sbj='108_1') #single sbj-epoch to get chans indexes
epoch.set_montage(m).pick_channels(pick_chann)
ch_indices  = mne.pick_channels(epoch.ch_names, include=plot_chann)
print(ch_indices)

# Init nested dict
mydict = {'week2': {'NREM': {'value':[], 'sbj':[]},
                    'REM': {'value':[], 'sbj':[]},
                    'WAKE':{'value':[], 'sbj':[]}},
        'week5': {'NREM': {'value':[], 'sbj':[]},
                'REM': {'value':[], 'sbj':[]},
                'WAKE':{'value':[], 'sbj':[]}} }
# Populate nested dict
for time in ['week2', 'week5']:
    mspes = store[time]
    stags = store_event[time]
    sbj = store_sbjnames[time]
    for stag_id, stag_name in zip(events_id, ['NREM', 'REM', 'WAKE']):
        finder = [ np.where(this_stag == events_id_map[stag_id])[0] for this_stag in stags ]
        nonempty_sbj = [s if len(f) > 0 else np.nan for s, f in zip(sbj, finder) ]
        mspe_stage  = [ mspes[i][:, finder[i]] for i in range(len(mspes)) ]

        boots = my_bootstraper(mspe_stage, ch_indices=ch_indices, repetions=1000, n=10)
        boots = np.nanmean(boots, 0) #av bootstraped averages
        mydict[time][stag_name]['value'] = boots
        mydict[time][stag_name]['sbj'] = nonempty_sbj

# Convert nested dict  to pandas df
df2 = build_df(mydict, time='week2').reset_index()
df5 = build_df(mydict, time='week5').reset_index()

# Reshape pandas df to long format
df2 = df2.melt(var_name = ['time', 'stag'], id_vars = ['new_idx'] )
df5 = df5.melt(var_name = ['time', 'stag'], id_vars = ['new_idx'])

df = pd.concat([df2, df5], axis=0, sort=True)
df['name_id_short'] = [n.split('_')[0] for n in df['new_idx']]

#df.fillna('NaN').to_csv('_'.join(plot_chann)+ 'mspe_scale' + str(scale_idx+1) + '.csv')
'''
df['channels'] = ['front'] * len(df)
df_front = df
DF = pd.concat([df_front, df_cent, df_occip], axis=0)
DF.fillna('NaN').to_csv('_'.join(plot_chann)+ 'channels_mspe_scale' + str(scale_idx+1) + '.csv')
'''

def detect_outlier(data):
    outliers = np.zeros(len(data))
    sorted(data)
    q1, q3= np.percentile(data,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr)
    upper_bound = q3 +(1.5 * iqr)
    for i, d in enumerate(data):
        if d < lower_bound or d > upper_bound:
            outliers[i] = True
        else:
            outliers[i] = False
    return outliers


def remove_outliers_from_df(df):
    new_df = pd.DataFrame(columns=df.columns)
    for (column, data) in df.iteritems():
        if 'new_idx' in column:
            new_df[column] = df[column]
            continue
        data = data.dropna()
        #sorted(data)
        outliers = np.zeros(len(data), dtype=bool)
        q1, q3= np.percentile(data,[25,75])
        iqr = q3 - q1
        lower_bound = q1 -(1.5 * iqr)
        upper_bound = q3 +(1.5 * iqr)
        for i, y in enumerate(data):
            #set_trace()
            if y < lower_bound or y > upper_bound:
                out = np.where(data == y)
                set_trace()
                outliers[i] = True
            else:
                outliers[i] = False
        #set_trace()
        new_df[column] = data[np.logical_not(outliers)]
    return new_df


def remove_outliers_from_df(df):
    df1 = df.copy()
    df = df._get_numeric_data()

    for (column, _) in df.iteritems():
        df[column].fillna(df[column].median())
        set_trace()
        q1 = df[column].quantile(.25)
        q3 = df[column].quantile(.75)
        mask = df[column].between(q1, q3, inclusive=True)
        iqr = df.loc[mask, column]
    return(df1)

dff = remove_outliers_from_df(df5)









dd = df5['week5']['NREM']['value'].dropna()
out = detect_outlier(dd)



# Plot boxes
fig, ax = plt.subplots()
box = sns.boxplot(x='stag', y='value', hue='time',data=df, linewidth=3, \
                ax=ax, whis=[5, 95], showfliers=False, dodge=True, order=['NREM', 'REM', 'WAKE'])
def modif_box(axes, facecolor, edgecolor, idxs):
    for i in idxs:
        mybox = box.artists[i]
        mybox.set_facecolor(facecolor)
        mybox.set_edgecolor(edgecolor)
        # Each box has several associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i*5,i*5+5):
            ax = plt.gca()
            line = ax.lines[j]
            line.set_color(edgecolor)
            line.set_mfc(edgecolor)
            line.set_mec(edgecolor)
    return

modif_box(box, 'white', 'black', [0,2,4])
modif_box(box, 'white', 'red', [1,3,5])

sns.swarmplot(x='stag', y='value', hue='time',data=df, split=True, color='black', \
            size=4, alpha=0.7, ax=ax, order=['NREM', 'REM', 'WAKE'])

legend_elements = [Patch(facecolor='white', edgecolor='black',
                         label='week 2', linewidth=3),
                    Patch(facecolor='white', edgecolor='red',
                        label='week 5', linewidth=3)]
ax.legend(handles=legend_elements, loc='lower right')
ax.set(ylabel= 'MSPE(scale={}) [bit]'.format(scale_idx+1), xlabel='', xticklabels= ['NREM', 'REM', 'WAKE'], ylim=[1.2, 1.7])
plt.suptitle(' '.join(plot_chann))
#plt.savefig('_'.join(plot_chann)+ 'mspe.tif', dpi=300)
'''
#ax.plot([-0.2, -0.2, 0.2, 0.2], [1.63, 1.64, 1.64, 1.63], linewidth=1, color='black')
ax.plot([-0.2, -0.2, 0.2, 0.2], [1.67, 1.68, 1.68, 1.67], linewidth=1, color='black')
#ax.plot([0.8, 0.8, 1.2, 1.2], [1.63, 1.64, 1.64, 1.63], linewidth=1, color='black')
ax.plot([0.8, 0.8, 1.2, 1.2], [1.67, 1.68, 1.68, 1.67], linewidth=1, color='black')
#ax.plot([1.8, 1.8, 2.2, 2.2], [1.63, 1.64, 1.64, 1.63], linewidth=1, color='black')
ax.plot([1.8, 1.8, 2.2, 2.2], [1.67, 1.68, 1.68, 1.67], linewidth=1, color='black')
#ax.text(-0.07, 1.645, 'p<.05', color='black', size=10)
#ax.text(0.93, 1.645, 'p<.05', color='black', size=10)
#ax.text(1.95, 1.645, 'ns', color='black', size=10)
ax.text(-0.07, 1.685, 'p<.05', color='black', size=10)
ax.text(0.93, 1.685, 'p=.06', color='black', size=10)
ax.text(1.95, 1.685, 'ns', color='black', size=10)
'''
plt.rcParams.update({'font.size': 15})
#pyplot.locator_params(axis='y', nbins=8)
#plt.show()
