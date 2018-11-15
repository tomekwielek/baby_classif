import numpy as np
import os
from config import myload, subjects, paths
import matplotlib.pyplot as plt
from functional import select_class_to_classif, load_single_append, plot_pe
from IPython.core.debugger import set_trace

bad_sbjs_1 = []
bad_sbjs_2 = ['112_2', '118_2', '119_2', '202_2', '204_2', '205_2', '206_2', '208_2', '209_2', '216_2',\
                '218_2', '221_2', '222_2', '224_2', '225_2', '226_2', '234_2', '236_2', '238_2', '239_2', \
                    '231_2', '212_2', '214_2', '220_2', '232_2']

sel_idxs = [1,2,3]
path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
#fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

pe, stag, names_pe, _ = load_single_append(path, fnames2, typ = 'mspet1m3')
#pe, stag, _ = select_class_to_classif(pe, stag, sel_idxs=sel_idxs)
psd, stagpsd, names_psd, freqs =  load_single_append(path, fnames2, typ = 'psd')
#psd, stagpsd, _ = select_class_to_classif(psd, stagpsd, sel_idxs=sel_idxs)

assert len(names_psd) == len(names_pe)
assert all([names_pe[i] == names_psd[i] for i in range(len(names_psd))])
assert all([stag[i].shape == stagpsd[i].shape for i in range(len(stagpsd))])

def plot_data(pe, mypsd, stag, sbj_name):
    stag.reset_index(inplace=True)
    print(pe.shape)
    print(mypsd.shape)
    print(len(stag))
    times = range(mypsd.shape[-1])

    mypsd = mypsd / np.abs(np.sum(mypsd, 0))
    mypsd =  np.log10(mypsd)
    mypsd = mypsd.mean(1) # average channels
    pe = pe.mean(1) # average channels
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
    mesh = ax1.pcolormesh(times, freqs, mypsd, cmap='plasma', shading='gouraud')
    ax1.set_xlim(xmin=0, xmax=mypsd.shape[-1])
    ax1.set_yticks(np.rint(freqs[::10]))
    ax1.set_ylabel('Freqs')
    for i, t in enumerate(['scale1', 'scale2', 'scale3', 'scale4', 'scale5']):
        ax2.plot(pe.T[:,i], label = t)
    ax2.legend(loc=1, prop={'size':6})
    ax2.set_xlim(xmin=0, xmax=pe.shape[-1])
    ax2.set_ylabel('Permutation entropy')
    ax3.plot(stag['numeric'])
    ax3.set_xlim(xmin=0, xmax=pe.shape[-1])
    ax3.set_xlabel('time [min]')
    ax3.set_ylabel('Sleep stage')
    ax3.set_yticks([1,2,3])
    ax3.set_yticklabels(['NREM', 'REM', 'WAKE'])
    plt.title('Sbj {}'.format(sbj_name))
    plt.show()

def remove_20hz_artif(pe, psd, stag, names, freqs, bad_sbjs):
    '''
    bad_sbjs - which subject have 20Hz artifacts (inspect plots )
    iterate over mypsd, for bad_sbjs get frequencies in 20Hz, threshold, keep bad_e_idcs and remove epochs,
    use bad_e_idcs to update pe, stag, names
    '''
    store_pe, store_psd, store_stag= [[] for i in range(3)]
    freqs = freqs.astype(int)
    idx_freq = np.where(freqs == 20)
    for idx_sbj, sbj in enumerate(names):
        if sbj in bad_sbjs:
            print sbj
            #plot_data(pe[idx_sbj], psd[idx_sbj], stag[idx_sbj], names_pe[idx_sbj])
            this_data = psd[idx_sbj][idx_freq,:,:][0]
            idx_time = np.where(this_data[0,0,:] > np.percentile(this_data[0,0,:], 88))[0]
            if sbj in ['236_2'] and 0 in idx_time:
                idx_time = idx_time[1:] #see annot by HL in excel and functional.py; BL shorter
            mask_psd = np.ones(psd[idx_sbj].shape,dtype=bool)
            mask_psd[:,:,idx_time] = False
            psd_cor = psd[idx_sbj][mask_psd].reshape(psd[idx_sbj].shape[:2]+(-1,))
            mask_pe = np.ones(pe[idx_sbj].shape,dtype=bool)
            mask_pe[:,:,idx_time] = False
            pe_cor = pe[idx_sbj][mask_pe].reshape(pe[idx_sbj].shape[:2]+(-1,))
            #set_trace()
            stag_cor = stag[idx_sbj].drop(idx_time, axis=0)
            #plot_data(pe_cor, psd_cor, stag_cor, names[idx_sbj])
        else:
            pe_cor = pe[idx_sbj]
            psd_cor = psd[idx_sbj]
            stag_cor = stag[idx_sbj]
        store_pe.append(pe_cor)
        store_psd.append(psd_cor)
        store_stag.append(stag_cor)
    return (store_pe, store_psd, store_stag)

p, ps, s = remove_20hz_artif(pe, psd, stag, names_pe, freqs, bad_sbjs_2)
