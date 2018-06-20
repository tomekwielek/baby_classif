def av_stag_pe_against_beh():
    #get average pe for NREM
    nrem1 = [pe1[i].mean(0)[stag1[i]['numeric'] == 1].mean() for i in range(len(pe1))]
    fn1 = [fnames1[i].split('_')[0] for i in range(len(fnames1))]
    nrem1 = pd.DataFrame({'id' : fn1, 'nrem_pe' : nrem1})
    nrem2 = [pe2[i].mean(0)[stag2[i]['numeric'] == 1].mean() for i in range(len(pe2))]
    fn2 = [fnames2[i].split('_')[0] for i in range(len(fnames2))]
    nrem2 = pd.DataFrame({'id' : fn2, 'nrem_pe' : nrem2})

    nrem1_sub = nrem1[nrem1['id'].astype('int').isin(bley['VPN'].astype('int').tolist())]
    nrem2_sub = nrem2[nrem2['id'].astype('int').isin(bley['VPN'].astype('int').tolist())]
    bley_sub1 = bley['Unnamed: 3'][bley['VPN'].astype(int).isin(nrem1['id'])]
    plt.scatter(bley_sub1, nrem1_sub['nrem_pe'])

    bley_sub2 = bley['Unnamed: 3'][bley['VPN'].astype(int).isin(nrem2['id'])]
    plt.scatter(bley_sub2, nrem2_sub['nrem_pe'])

    nrem1_sub = nrem1[nrem1['id'].astype('int').isin(ci['VPN'].astype('int').tolist())]
    ci_sub1 = ci[ci['VPN'].astype(int).isin(nrem1['id'])]
    nrem2_sub = nrem2[nrem2['id'].astype('int').isin(ci['VPN'].astype('int').tolist())]
    ci_sub2 = ci['sensitiv'][ci['VPN'].astype(int).isin(nrem2['id'])]
    fig, ax = plt.subplots()
    ax.scatter(ci_sub1['sensitiv'], nrem1_sub['nrem_pe'])
    for i, t in enumerate(ci_sub1['VPN']):
        ax.annotate(t, (ci_sub1['sensitiv'].iloc[i], \
                        nrem1_sub['nrem_pe'].iloc[i]))
    sns.regplot(ci_sub1['sensitiv'], nrem1_sub['nrem_pe'])
    plt.scatter(ci_sub2, nrem2_sub['nrem_pe'])
return
