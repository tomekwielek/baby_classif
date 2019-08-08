'''
(1) Read raw edf data and coresponding staging
(2) Segment data into 30s epochs, save as mne object (used for PSD analysis)
(3) Segment data into 30s epochs, compute MSPE, save
Note: there are two excell sheets with staging 'St_Pr_corrected_27min' or 'St_Pr_corrected_35min',
    set as function argument
'''
def read_raw(sheet, setup='mspet1m3'):
    import mne
    from mne import io
    import os
    import pandas as pd
    from pyentrp import entropy as ent
    import numpy as np
    from functional import map_stag_with_raw, my_rename_col, order_channels
    from config import raw_path, stag_fname, mysave, pe_par
    from IPython.core.debugger import set_trace
    from config import paths
    from mne.time_frequency import psd_welch
    from mne.io import RawArray
    from scipy.signal import welch
    from scipy.integrate import simps

    fnames_ = sorted(list(filter(lambda x: x.endswith('.edf'), os.listdir(raw_path))))
    fnames = [i.split('.edf')[0] for i in fnames_] #delete .edf from fname
    ss = pd.read_excel(stag_fname, sheet_name=sheet)

    ss = my_rename_col(ss)
    ss = ss.drop(ss.columns[[0, 1, 2]], axis=1) #drop cols like 'Condition', 'Minuten', etc
    #set pe parameters
    if setup != 'psd':
        embed = pe_par[setup]['embed']
        tau = pe_par[setup]['tau']

    # set bad sbjs, defined by inspecting excell dat (e.g red annotations by Heidi)
    if sheet == 'St_Pr_corrected_27min':
        # all but 213_2 considered as nicht auswertbar.
        # 213_2 schwierig + 'all physio channels flat, ECG reconstructed from EMG - pretty noisy'
        bad = ['205_1_S', '206_1_S', '208_1_S', '213_1_S', '238_1_S', '239_1_S', '213_2_S']

    if sheet == 'St_Pr_corrected_35min':
        bad = ['104_1_S'] #104_1-S missing, pressumbly unscorable

    def preproces(raw):
        raw.pick_types(eeg=True)
        raw.filter(l_freq=1., h_freq=30., fir_design='firwin')
        return raw

    def load_raw_and_stag(idf_stag, idf_raw):
        stag = ss.filter(like = idf_stag)
        raw = io.read_raw_edf(raw_path + idf_raw + '.edf', preload=True, \
                            stim_channel=None)
        #return preproces(raw), stag # apply pp.
        return raw, stag #no filtering


    def compute_psd_segm(raw, window=30):
        from mne.time_frequency import psd_welch
        from mne.io import RawArray

        def get_raw_array(data):
            ch_types  = ['eeg'] * chan_len
            info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types=ch_types)
            raw_array = RawArray(data, info=info)
            return raw_array

        data = raw.get_data()
        sfreq = raw.info['sfreq']
        length_dp = len(raw.times)
        window_dp = int(window*sfreq)
        no_epochs = int(length_dp // window_dp)
        chan_len = len(raw.ch_names)
        store_psd = []
        for i in range(no_epochs):
            e_ = data[...,:window_dp]
            data = np.delete(data, np.s_[:window_dp], axis =1 )
            ra = get_raw_array(e_)
            psd, freqs = psd_welch(ra ,fmin=1,fmax=30)
            #psd = 10 * np.log10(psd) # log is done latter on, in wrap_psd.py
            store_psd.append(psd)
        #psd array of shape [freqs, channs, epochs]
        return freqs, np.asarray(store_psd).transpose(2,1,0)

    def compute_pe_segm(raw, embed=3, tau=1, window=30, mspe=False):
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        length_dp = len(raw.times)
        window_dp = int(window*sfreq)
        no_epochs = int(length_dp // window_dp)
        chan_len = len(raw.ch_names)
        store_segm = []
        if mspe:
            scale = 5
            m = np.zeros((scale, chan_len, no_epochs)) # multiscale entropy
        else:
            m = np.zeros((chan_len, no_epochs)) # PE
        for i in range(no_epochs):
            e_ = data[...,:window_dp]
            print (e_.shape)
            data = np.delete(data, np.s_[:window_dp], axis =1 )
            for j in range(chan_len):
                if mspe:
                    m[:,j,i], _ = ent.multiscale_permutation_entropy(e_[j], m=3, delay=1, scale=scale)

                else:
                    m[j,i] = ent.permutation_entropy(e_[j], m=embed, delay=tau)
            del e_
        return m

    def encode(stag):
        stages_coding = {'N' : 1, 'R' : 2, 'W' : 3,'X' : 4, 'XW' : 5, 'WR' : 6, 'RW' : 7, 'XR' : 8, 'WN' : 9}
        stag['numeric'] = stag.replace(stages_coding, inplace=False).astype(float)
        return stag

    idfs = map_stag_with_raw(fnames, ss, sufx='ref119')
    idfs = {i : v for i, v in idfs.items() if len(v) >= 1} # drop empty
    idfs = {i : v for i, v in idfs.items() if 'P' not in i} # drop Prechtls
    #write idfs df to excel
    #from write_data import mywriter
    #mywriter(idfs, sheet + '.xlsx')

    #select single subject
    #sbj = '113_2_S'
    #idfs = {k : v for k,v in idfs.items() if k == sbj}


    def raw_to_epochs_using_annotations(raw, stag, k):
        if k.startswith('110_2'): #see annot by HL in excel; BL shorter
            stag = stag.iloc[:66]
        if k.startswith('236_2'):
            stag  = stag[1:] #see annot by HL in excel; BL shorter
        stag = stag.dropna()
        sfreq = raw.info['sfreq']
        n_epochs = np.floor(raw.times[-1] / 30).astype(int)
        if len(stag) - n_epochs == 1:# can be shorter due to windowing
            stag = stag[:-1]
        raw.info['meas_date'] = 0
        onset = np.arange(0, raw.times[-1], 30 )[:-1]
        duration = [30] * n_epochs
        description = stag[k].values
        annotations = mne.Annotations(onset, duration, description, orig_time=0)
        raw.set_annotations(annotations)
        event_id  = {'N' : 1,
                    'R' : 2,
                    'W' : 3,
                    'X' : 4,
                    'XW' : 5,
                    'WR' : 6,
                    'RW' : 7,
                    'XR' : 8,
                    'WN' : 9}
        events, event_id_this = mne.events_from_annotations(
                    raw, event_id=event_id, chunk_duration=30.)
        raw.info['meas_date'] = None
        tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
        epochs = mne.Epochs(raw=raw, events=events,
                          event_id=event_id_this, tmin=0., tmax=tmax, baseline=None)
        return epochs

    for k, v in sorted(idfs.items()):
        k_short = k.split('_S')[0]
        p = paths(typ=setup, sbj=k_short) #path to a folder
        if k not in bad:
            print (k)
            raw, stag = load_raw_and_stag(k, v[0])
            raw, _ = order_channels(raw) #reorder channels, add zeros for missings (EMG usually)
            stag = encode(stag)
            epochs = raw_to_epochs_using_annotations(raw,stag,k)
            mysave(var=epochs, typ='epoch', sbj=k[:5])
            pe = compute_pe_segm(raw, embed=embed, tau=tau, mspe=True)
            #mysave(var = [stag, pe], typ='mspet1m3', sbj=k[:5])
            #freqs, psd = compute_psd_segm(raw, window=30)
            #mysave(var = [stag, psd, freqs], typ='psd', sbj=k[:5])
        elif k in bad:
            print ('Sbj dropped, see red annot. by H.L in excel file')
            continue

if __name__ == "__main__":
    import sys
    read_raw(sys.argv[1], sys.argv[2])
