import pandas as pd
from config import raw_path
scores_path = 'H:\\BABY\\results\\perf_raw\\'
f_name = raw_path + '\\behav\\Bayley_MS_Tomek.xlsx'
scores = pd.read_csv(scores_path + 'cross_time_scores.csv')
bley = pd.read_excel(f_name, sheet_name='Übersicht')
ci = pd.read_excel(f_name, sheetname='CI_Details', skiprows=2)
bley = bley.dropna()
ci = ci.dropna()
bley['VPN'] = bley['VPN'].copy().astype(int).astype(str)
ci['VPN'] = ci['VPN'].copy().astype(str)
scores['name'] = scores['name'].copy().astype(str)
bley_sub = bley[bley['VPN'].isin(scores['name'].tolist())]
ci_sub = ci[ci['VPN'].isin(scores['name'].tolist())]

scores_sub_bley = scores[scores['name'].isin(bley_sub['VPN'].tolist())]
scores_sub_ci = scores[scores['name'].isin(ci_sub['VPN'].tolist())]



fig, ax = plt.subplots()
ax.scatter(ci_sub['sensitiv'], scores_sub_ci['f1_1'])
for i, t in enumerate(ci_sub['VPN']):
    ax.annotate(t, (ci_sub['sensitiv'].iloc[i], \
                    scores_sub_ci['f1_1'].iloc[i]))
fig, ax2 = plt.subplots()
ax2.scatter(bley_sub['Unnamed: 3'], scores_sub_bley['f1_1'])
for i, t in enumerate(bley_sub['VPN']):
    ax2.annotate(t, (bley_sub['Unnamed: 3'].iloc[i], \
                    scores_sub_bley['f1_1'].iloc[i]))

from scipy import stats
stats.spearmanr(ci_sub['sensitiv'], scores_sub_ci['f1_1'])
