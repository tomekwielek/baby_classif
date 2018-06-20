import pandas as pd
from config import raw_path
import matplotlib.pyplot as plt
import numpy as np
scores_path = 'H:\\BABY\\results\\perf_raw\\'
f_name = raw_path + '\\behav\\Bayley_MS_Tomek.xlsx'
#scores = pd.read_csv(scores_path + 'cross_time_scores.csv')
scores = pd.read_csv('cross_time_scores.csv')

bley = pd.read_excel(f_name, sheet_name='Ubersicht')
ci = pd.read_excel(f_name, sheetname='CI_Details', skiprows=2)
bley = bley.dropna()
ci = ci[['VPN', 'sensitiv']].dropna()
bley['VPN'] = bley['VPN'].copy().astype(int).astype(str)
ci['VPN'] = ci['VPN'].copy().astype(str)
scores['name'] = scores['name'].copy().astype(str)
bley_sub = bley[bley['VPN'].isin(scores['name'].tolist())]
ci_sub = ci[ci['VPN'].isin(scores['name'].tolist())].reset_index()

scores_sub_bley = scores[scores['name'].isin(bley_sub['VPN'].tolist())]
scores_sub_ci = scores[scores['name'].isin(ci_sub['VPN'].tolist())].reset_index()
'''
#replace 0 with smal float, transform to normalize
from scipy import stats
ci_sub['sensitiv_trf'] = stats.boxcox(ci_sub['sensitiv'].replace(0, 0.0001))[0]
scores_sub_ci['f1_1_trf'] = stats.boxcox(scores_sub_ci['f1_1'])[0]
'''
fig, ax = plt.subplots()
#ax.scatter(ci_sub['sensitiv'], scores_sub_ci['f1_1'])
ax.scatter(bley_sub['Unnamed: 3'], scores_sub_bley['f1_1'])
# plot sb label
#for i, t in enumerate(ci_sub['VPN']):
    #ax.annotate(t, (ci_sub['sensitiv'].iloc[i], \
    #                scores_sub_ci['f1_1'].iloc[i]))
    #ax.set_title('CI')
plt.show()
'''
fig, ax2 = plt.subplots()
ax2.scatter(bley_sub['Unnamed: 3'], scores_sub_bley['f1_1'])
for i, t in enumerate(bley_sub['VPN']):
    ax2.annotate(t, (bley_sub['Unnamed: 3'].iloc[i], \
                    scores_sub_bley['f1_1'].iloc[i]))
'''
from scipy import stats
from scipy.stats import linregress
import seaborn as sns
ci_zeros_out = np.where(ci_sub['sensitiv'] == 0, False, True)
stats.kendalltau(ci_sub['sensitiv'],
                scores_sub_ci['f1_1'])
linregress(ci_sub['sensitiv'],scores_sub_ci['f1_1'])
plt.scatter(ci_sub['sensitiv'],scores_sub_ci['f1_1'])


sns.regplot(ci_sub['sensitiv'],scores_sub_ci['f1_1'], fit_reg=True, ci=95)

#sns.regplot(ci_sub['sensitiv'][ci_zeros_out],scores_sub_ci['f1_1'][ci_zeros_out])

scores['f1_av'] = scores[['f1_1', 'f1_2']].mean(1)
