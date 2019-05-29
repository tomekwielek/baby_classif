import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
s1 = [1.347, 1.424, 1.445]
s2 = [1.642, 1.627, 1.564]

d = np.hstack ([s1, s2])
l = np.hstack([['s1'] *3, ['s2'] *3 ])
s = np.hstack(['NREM', 'REM', 'WAKE'] * 2 )
df = pd.DataFrame({'pe' : d, 'scale' : l, 'stage' : s })

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(5, 2))
sns.pointplot(data=df[df['scale']=='s1'], x ='stage', y='pe', ax = ax1, color='black')
ax1.set(ylim=(1.335, 1.67))
ax1.set( xlabel='', ylabel='PE')
sns.pointplot(data=df[df['scale']=='s2'], x ='stage', y='pe', ax = ax2, color='black')
ax2.set(ylim=(1.335, 1.67))
ax2.set( xlabel='', ylabel='', yticklabels='')



x = np.arange(6)
fig, ax = plt.subplots(1)
ax.bar(x, height= [0.3, 0.1, 0.15, 0.1, 0.15, 0.2], edgecolor='black', color='None', width=0.6, linewidth=2)
ax.set( yticks=[],xticks=x, xticklabels=['S1','S2','S3', 'S4', 'S5', 'S6'], ylabel='Probablity of occurrence (p)')
matplotlib.rcParams.update({'font.size': 16})
plt.box(False)
ax.set(yticks= (0,0.3))
