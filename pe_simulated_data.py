#from https://martinos.org/mne/dev/auto_examples/simulation/plot_simulate_raw_data.html
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import read_source_spaces, find_events, Epochs, compute_covariance
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw
from pyentrp import entropy as ent

from mne import EpochsArray
from mne.time_frequency import tfr_morlet
from mne.time_frequency import psd_multitaper

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
src_fname = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
bem_fname = (data_path +
             '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')
iters = 1
# Load real data as the template (mne data sample)
raw = mne.io.read_raw_fif(raw_fname, preload = True)
raw.set_eeg_reference(projection=True)
raw = raw.crop(0., 30.)  # 30 sec is enough
raw =raw.pick_types(eeg=True, meg=False)
raw = raw.copy().resample(250, npad='auto')
n_dipoles = 4  # number of dipoles to create
epoch_duration = 30.  # duration of each epoch/event

def data_fun(times):
    """Generate time-staggered sinusoids at harmonics of freq Hz"""
    n = 0  # harmonic number
    global n
    n_samp = len(times)
    window = np.zeros(n_samp)
    start, stop = [int(ii * float(n_samp) / (2 * n_dipoles))
                   for ii in (2 * n, 2 * n + 1)]
    window[start:stop] = 1.
    n += 1
    data = 25e-10 * np.sin(freq * np.pi * 2. * n * times)
    data *= window
    return data

def sim_epoch(freq, noise_level):
    times = raw.times[:int(raw.info['sfreq'] * epoch_duration)]
    src = read_source_spaces(src_fname)
    stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                              data_fun=data_fun, random_state=0)
    raw_sim = simulate_raw(raw, stc, trans_fname, src, bem_fname, cov='simple',
                          ecg=True, blink=True,
                           n_jobs=1, verbose=True)
    rd = raw_sim.copy()._data
    noise = np.random.normal(size=rd.shape, scale=rd.std()*noise_level)
    raw_sim._data = rd + noise
    raw_sim.plot_psd(fmax=35, average=True, dB=True)
    return raw_sim

def generate_fake_record(noise_level):
    #iterate iters times generating each time 10 fake concatenated epochs:
    # (each having diff. frequency content.)
    import copy
    store_ep = [] #Unconcatenated
    for freq in [15]:#range(5,35,3):
        global freq
        raw_ = sim_epoch(freq=freq, noise_level=noise_level )
        del freq
        store_ep.append(raw_)
    rawc = mne.concatenate_raws(copy.deepcopy(store_ep), preload=True)
    rawc = rawc._data[np.newaxis, :] #add 1 dim if no iteration

    '''
    store_psd = []
    for i in range(len(store_ep)):
        psd, freqs = psd_welch(store_ep[i], fmin=1, fmax=35, tmin=0.)
        psd = psd.mean(0) #av channels
        store_psd.append(psd)
    '''
        #psds = np.asarray(store_psd)

    #from mne.time_frequency import psd_welch
    #psds, freqs = psd_welch(rawc, fmin=1, fmax=35, tmin=0., n_fft=250*30)

    times = np.arange(0, rawc.shape[2])
    events = [[0,0,1]]
    event_id = dict(fake1=1)

    ep = EpochsArray(rawc, store_ep[0].info, events=events, event_id=event_id)

    freqs_tfr = np.arange(1., 35., 3.)
    n_cycles = freqs/2
    power = tfr_morlet(ep, freqs=freqs_tfr,n_cycles=n_cycles, return_itc=False,
                        average=True)
    #average power across time (within each 30s epoch)
    powav_time = power._data.reshape([no_chs, len(freqs_tfr), -1, 30])
    t_  = power._data.reshape(no_chs, len(freqs), no_times)

    raws[str(noise_level)] = store_raws_freqs
    powers[str(noise_level)] = powav_time
    return

noise_levels = [0.6]
raws = {}
powers = {}
for nl in noise_levels:
    generate_fake_record(nl)



'''
#temp
dl = np.squeeze(store_record['0.1'].mean(1))
dm = np.squeeze(store_record['0.5'].mean(1))
dh = np.squeeze(store_record['0.9'].mean(1))

plt.plot(dl[5,:])
plt.plot(dm[5,:])
plt.plot(dh[5,:])




########################################################################
#Compute pe
def preproces(raw):
    raw.pick_types(eeg=True)
    raw.filter(l_freq=1, h_freq=30)
    return raw

def compute_pe_segm(raw, embed=3, tau=1, window=30):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    length_dp = len(raw.times)
    window_dp = int(window*sfreq)
    no_epochs = int(length_dp // window_dp)
    chan_len = len(raw.ch_names)
    store_segm = []
    m = np.zeros((chan_len, no_epochs))
    #m = np.zeros((1, no_epochs))
    for i in range(no_epochs):
        e_ = data[...,:window_dp]
        data = np.delete(data, np.s_[:window_dp], axis =1 )
        for j in range(chan_len): #single chan
            m[j,i] = ent.permutation_entropy(e_[j], m=embed, delay=tau)
        del e_
    return m

store_pe = []
for d in store_ep:
    raw = preproces(d)
    pe = compute_pe_segm(raw, embed=3, tau=1, window=30)
    store_pe.append(pe)
#save pe list (len=#freqs)
f = open('store_pe.txt',"wb")
pickle.dump(store_pe,f)

d = np.asarray(store_pe)
# plot pe vs mask
fig, axes = plt.subplots()
axes[0].plot(np.squeeze(d))

plt.show()

'''
