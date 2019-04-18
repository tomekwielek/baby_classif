fnames = ['104_2', '108_2', '110_2', '111_2', '112_2', '118_2', '119_2'] #sbjs where count maches
psd, stag, names, freqs = load_single_append(_, fnames, typ='psd_hd')
sel_idxs = [1,2,3]
psd, stag, _ = select_class_to_classif(psd, stag, sel_idxs=sel_idxs)

psd = [psd[i][:,:-1, :] for i in range(len(psd))] #delete las ch, zeros

rel_psd = [psd[i] / np.abs(np.sum(psd[i], 0)) for i in range(len(psd))]


def av_channels_epochs(data, stag, what_stag, names):
    '''
    Function agragates across epochs and/or channels
    '''
    #select data corr with given sleep stage s
    np.random.seed(123)
    av_stag = [data[i][:,:, np.where(stag[i]['numeric'] == what_stag)] for i in range(len(data))]
    names_ss = [ names[i] for i in range(len(av_stag)) if av_stag[i].size != 0 ]
    if k != 'all': #average k number of epochs
        ep_i = [ av_stag[i].shape[3] for i in range(len(av_stag)) ] #count how many epochs we have for given ss
        #randmoly sample epochs

        ep_sample_i = [np.random.choice(ep_i[i], size=k, replace=False) if ep_i[i] > k  else [] if ep_i[i] == 0 else np.arange(ep_i[i]) for i in range(len(ep_i)) ]
        #average epochs using random index
        av_stag =[ np.where(len(ep_sample_i[i]) > 0 , np.median(av_stag[i][:,:,:,ep_sample_i[i]], 3), np.nan) for i in range(len(av_stag)) ]
    else: #average all epochs
        av_stag =[ np.where(av_stag[i].size != 0, np.nanmedian(av_stag[i], 3), np.full(av_stag[i].shape[:3], np.nan)) for i in range(len(av_stag))]

    return av_stag


k= 10 #4 # epochs to average
psd_ = av_channels_epochs(rel_psd, stag, what_stag=3, names=names)
psd_data = [ psd_[i] for i in range(len(psd_)) if not np.isnan(psd_[i][0]).any() ]
psd_data = np.asarray(psd_data).mean(0) #mean sbj
psd_data = psd_data.transpose([1,0, 2])

m = mne.channels.read_montage('GSN-HydroCel-129')
raw = mne.io.read_raw_egi(mff_path+fnames_mff[0], preload=True, include=markers).pick_types(eeg=True)
raw.ch_names.pop() #del last
info = mne.create_info(ch_names = raw.ch_names, sfreq=125., ch_types='eeg', montage=m)

tfr = mne.time_frequency.AverageTFR(info, psd_data, times=[1], freqs=freqs, nave=1)
tfr.plot_topomap(fmin=24, fmax=26)




###################################################################################
mff_fname = 'D:\\baby_mff\\eeg_mff_raw_correctedMarkers\\203_2_20150812.mff'
edf_fname = 'H:\\BABY\\data\\203_2_correctFilter_2heogs_ref100.edf'
stag, psd, freqs = myload(typ='psd_hd', sbj='203_2')
raw_mff_mark = mne.io.read_raw_egi(mff_fname, preload=True, include=markers27)
#raw_mff_no = mne.io.read_raw_egi(mff_fname, preload=True)
sfreq = raw_mff_mark.info['sfreq']

raw_edf =  mne.io.read_raw_edf(edf_fname, preload=True)

def my_crop(raw):
    events = mne.find_events(raw, stim_channel = 'STI 014')
    tmin = events[0][0] / sfreq
    tmax = (events[-1][0] + 180000) / sfreq
    raw_last_ts = raw.times.shape[0] / sfreq
    #set_trace()
    if tmax > raw_last_ts:
        tmax = int(raw_last_ts)
        print('Short file')
    raw.crop(tmin, tmax)
    return raw

mff_crop = my_crop(raw_mff_mark)

print('len edf in [30s segments]:{}'.format(raw_edf.times.shape[0]/125./30.))
print('len psd in [30s segments] :{}'.format(psd.shape[-1]))
print('len mff in [30s segments]: {}'.format(mff_crop.times.shape[0]/sfreq/30))
print ('len stag [30s segments]: {}'.format(len(stag)))







############################################
raw = mne.io.read_raw_egi(mff_path + fnames_mff[31], preload=True)
