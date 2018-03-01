def read_raw(setup, sheet):
    import mne
    from mne import io
    import os
    import pandas as pd
    from pyentrp import entropy as ent
    import numpy as np
    from functional import map_stag_with_raw, my_rename_col
    from config import raw_path, stag_fname, mysave, pe_par

    raw_fnames_ = sorted(np.array(filter(lambda x: x.endswith('.edf'), os.listdir(raw_path))))
    raw_fnames = [i.split('.edf')[0] for i in raw_fnames_] #dlete .edf from fname

    s = pd.read_excel(stag_fname, sheetname = sheet)
    s = my_rename_col(s)

    embed = pe_par[setup]['embed']; tau = pe_par[setup]['tau']
    #different staging for (un)corrected staging
    if 'corrected' in sheet:
        typ_name = 'pet' + str(tau) + 'm' + str(embed)
    else:
        typ_name = 'pet' + str(tau) + 'm' + str(embed) + '_stag_uncorr'#for 'Staging_vs_Prechtl_27min'

    #bad defined by inspecting excell dat (e.g red annotations by Heidi)
    if sheet == 'St_Pr_corrected_27min':
        bad = ['205_1-S', '206_1-S', '208_1-S', '211_1-S', '211_2-S', '212_2-S', '213_1-S',
        '213_2-S', '214_1-S', '214_2-S', '218_1-S', '220_1-S', '220_2-S', '221_1-S', '226_1-S', '227_2-S',
        '231_2-S', '232_1-S', '232_2-S', '235_1-S', '238_1-S', '239_1-S']
    elif sheet == 'Staging_vs_Prechtl_27min':
        bad = ['205_1', '206_1', '207_1_S', '207_2_S', '208_1', '211_1.1heog?', '212_2,2heogref100',
                '213_1', '213_2,1heog', '213_2.2heogsref100', '214_1,2heogsref100', '214_2,2heogsref100',
                '218_1,1heog', '220_2.2heogref100', '221_12heogref100', '221_1.1heog', '222_1,2heogsref100',
                '223_1,2heogsref100', '226_1.2heogsref100', '226_2,1heog', '226_2,2heogref100', '227_2,1heog',
                '231_2,1heog', '232_1', '232_2,1heog', '235_1,2heogsref100', '235_1,1heog', '238_1', '239_1']
    else: bad = []
    def preproces(raw):
        raw.pick_types(eeg=True)
        raw.filter(l_freq=1, h_freq=30)
        return raw

    def load_raw_and_stag(idf_stag, idf_raw):
        stag = s.filter(like = idf_stag)
        raw = io.read_raw_edf(raw_path + idf_raw + '.edf', preload=True)
        return preproces(raw), stag

    def compute_pe_segm(raw, embed=3, tau=1, window=30):
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        length_dp = len(raw.times)
        window_dp = int(window*sfreq)
        no_epochs = int(length_dp // window_dp)
        chan_len = len(raw.ch_names)
        store_segm = []
        m = np.zeros((chan_len, no_epochs))
        #m = np.zeros((4, chan_len, no_epochs))
        for i in range(no_epochs):
            #e_ = data[...,-window_dp:]
            e_ = data[...,:window_dp]
            print e_.shape
            #data = np.delete(data, np.s_[-window_dp:], axis =1 )
            data = np.delete(data, np.s_[:window_dp], axis =1 )
            for j in range(chan_len):
                m[j,i] = ent.permutation_entropy(e_[j], m=embed, delay=tau)
                #m[:,j,i] = ent.multiscale_permutation_entropy(e_[j], m=3, delay=1, scale=4)
            del e_
        return m

    def encode(stag):
        stages_coding = {'N' : 1, 'R' : 2, 'W' : 3,'X' : 4, 'XW' : 5, 'WR' : 6, 'RW' : 7, 'XR' : 8}
        stag['numeric'] = stag.replace(stages_coding, inplace=False).astype(float)
        return stag

    idfs = map_stag_with_raw(raw_fnames, s)
    idfs = {i : v for i, v in idfs.items() if len(v) >= 1} # drop empty
    idfs = {i : v for i, v in idfs.items() if 'P' not in i} # drop Prechtls
    ''' select single subject
    sbj = '202_1_S'
    idfs = {k : v for k,v in idfs.items() if k == sbj}
    '''
    for s_, r_ in sorted(idfs.items()):
        if s_ not in bad:
            print s_
            raw, stag = load_raw_and_stag(s_, r_[0])
            stag = encode(stag)
            pe = compute_pe_segm(raw, embed=embed, tau=tau)
            mysave(var = [stag, pe], typ=typ_name, sbj=s_[:5])
        elif s_ in bad:
            print 'Sbj dropped, see red annot. by H.L in excel file'
            continue
