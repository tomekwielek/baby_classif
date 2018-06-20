#from https://martinos.org/mne/dev/auto_examples/simulation/plot_simulate_raw_data.html
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import read_source_spaces, find_events, Epochs, compute_covariance
from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw
from pyentrp import entropy as ent

from mne import EpochsArray
from mne.time_frequency import psd_welch

from scipy.signal import welch
from scipy.stats import linregress

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
src_fname = data_path + '/subjects/sample/bem/sample-oct-6-src.fif'
bem_fname = (data_path +
             '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')
# Load real data as the template (mne data sample)
raw = mne.io.read_raw_fif(raw_fname, preload = True)
raw.set_eeg_reference(projection=True)
raw = raw.crop(0., 30.)  # 30 sec is enough
raw =raw.pick_types(eeg=True, meg=False)
raw.pick_channels(raw.info['ch_names'][::5])
raw = raw.copy().resample(250, npad='auto')
n_dipoles = 2  # number of dipoles to create
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
    data = 10e-8 * np.sin(freq * np.pi * 2. * n * times)
    data *= window
    return data

def sim_epoch(freq, noise_level, blink):
    times = raw.times[:int(raw.info['sfreq'] * epoch_duration)]
    src = read_source_spaces(src_fname)
    stc = simulate_sparse_stc(src, n_dipoles=n_dipoles, times=times,
                              data_fun=data_fun, random_state=0)
    raw_sim = simulate_raw(raw, stc, trans_fname, src, bem_fname, cov='simple',
                          ecg=True, blink=blink,
                           n_jobs=1, verbose=True)
    rd = raw_sim.copy()._data
    #noise scale: oscillation amplitute * noise_level
    noise = np.random.normal(size=rd.shape, scale= 10e-8*noise_level)

    #raw_sim._data = rd + noise
    #plot raw psd
    #raw_sim.plot_psd(fmax=50, average=True, dB=True)
    return raw_sim
'''
psd of simulated epoch, averaged across channels
'''
def get_psd(raw, fmin, fmax, plot=False):
    psds, freqs = psd_welch(raw, fmin=fmin, fmax=fmax, tmin=0.)
    #psds = 10 * np.log10(psds)  # scale to dB
    psds_m = psds.mean() # av. channels and freq bins
    if plot:
        fig,ax = plt.subplots()
        ax.plot(psds_m)
        ax.set_xticks(range(1,35, 5))
        plt.show()
    return psds_m
'''
Compute pe; same function as for real data
'''
def compute_pe_segm(raw, embed=3, tau=1, window=30, mspe=True):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    length_dp = len(raw.times)
    window_dp = int(window*sfreq)
    no_epochs = int(length_dp // window_dp)
    chan_len = len(raw.ch_names)
    store_segm = []
    if mspe:
        scale = 4
        m = np.zeros((scale, chan_len, no_epochs)) # multiscale entropy
    else:
        m = np.zeros((chan_len, no_epochs)) # PE
    for i in range(no_epochs):
        e_ = data[...,:window_dp]
        print e_.shape
        data = np.delete(data, np.s_[:window_dp], axis =1 )
        for j in range(chan_len):
            if mspe:
                m[:,j,i] = ent.multiscale_permutation_entropy(e_[j], m=3, delay=1, scale=scale)
            else:
                m[j,i] = ent.permutation_entropy(e_[j], m=embed, delay=tau)
        del e_
    return m



def sim_giv_noise(noise_level, freq_, blink, plot_psd=True, ):
    freq = freq_
    global freq
    raw = sim_epoch(freq=freq, noise_level=noise_level, blink=blink)
    del freq
    # psd around frequency of simulation
    psd_around = get_psd(raw, fmin=freq_-2, fmax=freq_+2, plot=False)

    simd_pe = compute_pe_segm(raw, embed=3, tau=1, window=30, mspe=True)
    #return [simd_psd, simd_pe]
    return raw, simd_pe, psd_around

# freqs. for simulataion, each epoch simulates. diff. freq. component
frequencies = np.arange(3,50,4)
# diffrente noise levels
noise_levels = [ 0.01, 1, 10]
noise_levels =  [1]
blinks = [False, True]
# store simulated data for giv. noise level
stores_noise = []
# iterate over noise levels
#for ns in noise_levels:
for b in blinks:
    stores = []
    for freq_ in frequencies:
        raw, pe, psd_around = sim_giv_noise(noise_level=1, freq_=freq_, plot_psd=False, blink=b)
        stores.append([raw, pe, psd_around])
    stores_noise.append(stores)

'''
compute 5-50Hz psd using scipy welch func., return slope of the spectrum,
'''
def get_psds_slope(raw_arr, show=False):
    raw_arr = raw_arr.filter(0.2,50).copy()
    f, power = welch(raw_arr._data, 250., 'hamming', 256, scaling='density')
    power = power.mean(0) #average channels
    power = np.log10(power) #dB
    f = np.round(f)
    mask = np.where( (f>= 3) & (f <=50.), True, False) #select freq bins
    # get slope of the psd
    f = f[mask]
    power = power[mask]
    slope, intercept, _, _, _ = linregress(f, power)
    if show:
        plt.figure()
        plt.plot(f, intercept + slope*f, 'r', label='fitted line')
        plt.plot(f, power)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()
    return slope

#get slopes; iterate over noise and freqs.
slopes_noise = []
for ns in range(len(blinks)):
    slopes = []
    for i in range(len(stores)):
        s = get_psds_slope(stores_noise[ns][i][0], show=False)
        slopes.append(s)
    slopes_noise.append(slopes)

#pes = [stores[i][1] for i in range(len(stores))]
pes = [[stores_noise[ns][i][1] for i in range(len(stores))]
                                for ns in range(len(stores_noise)) ]
psds_around = [[stores_noise[ns][i][2] for i in range(len(stores))]
                                for ns in range(len(stores_noise)) ]

'''
plot slopes against pes, frequencies as a colorbar
'''
for i, n in enumerate(blinks):
    fig, ax = plt.subplots(figsize=(8, 8))
    slope2, intercept2, rv, _, _ = linregress(pes[i], slopes_noise[i])
    sc = ax.scatter(pes[i], slopes_noise[i], c=frequencies, s=100)
    #ax.plot(pes[i], intercept2 + slope2*np.asarray(pes[i]), 'r', label='fitted line')
    #ax.text(1.78, -0.04, 'r=%0.2f' %rv)
    ax.set_ylim(-0.085, 0.045)
    ax.set_xlim(1.75, 1.8)
    ax.set_xlabel('Permutation entropy', fontsize=25)
    ax.set_ylabel('PSD slope', fontsize=25)
    #for fi, f in enumerate(frequencies):
    #    ax.annotate(f, (pes[i][fi], slopes_noise[i][fi]))
    cbar = fig.colorbar(sc, ticks=frequencies[::3])
    cbar.ax.set_yticklabels(['%sHz' %f for f in frequencies[::3]])
    #cbar.ax.set_title('simulated activ.')
    plt.show()
    #savefig('noise'+str(noise_level)+'.png')
'''
plot both pes and slopes against frequencies, noise as colorbar
'''
fig, ax = plt.subplots(figsize=(8, 10))
for i, b in zip(range(len(pes)), ['NO BLINKS', 'BLINKS']):
    ax.plot(frequencies, pes[i], linewidth=10, label=b)
    ax.set_yticks(np.arange(1.76, 1.79, 0.005))
    ax.scatter(frequencies, [ 1.756]*len(frequencies), c=frequencies, s=100)
    ax.set_ylabel('Permutation entropy', fontsize=25)
    ax.set_xlabel('Frequency [Hz]', fontsize=25)
    ax.legend(fontsize=20)

    #axes[2].plot(frequencies, psds_around[i])

'''
customized plot of psd
raw = stores_noise[0][8][0]
f = raw.plot_psd(fmin = 5, fmax=25, average=True, dB=True)
f.axes[0].set_xlabel('frequency[Hz]', fontsize=23)
f.axes[0].set_ylabel('log10(power)', fontsize=25)
f.axes[0].set_ylim([-35, 20])
f.axes[0].set_xticklabels([])
f.axes[0].set_yticklabels([])
'''
