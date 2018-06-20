import numpy as np
import os
from config import myload, subjects, paths
import matplotlib.pyplot as plt
from functional import select_class_to_classif, load_single_append, plot_pe

def plot_data(pe, mypsd, stag):
    stag.reset_index(inplace=True)
    print pe.shape
    print mypsd.shape
    print len(stag)
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
    plt.show()

# no mypsd, pe with channels
def plot_data_ver2(pe, mypsd, stag, pred):
    stag.reset_index(inplace=True)
    print pe.shape
    print mypsd.shape
    print len(stag)
    times = range(mypsd.shape[-1])

    mypsd = mypsd / np.abs(np.sum(mypsd, 0))
    mypsd =  np.log10(mypsd)
    #mypsd = mypsd.mean(1) # average channels
    #mypsd = mypsd[:,:-5,:].mean(1) # average EEG channels, physio drop
    mypsd = mypsd[:, 1, :] #take second channel
    #pe = pe.mean(1) # average channels
    fig, axes= plt.subplots(4,1, sharex=True)
    mesh = axes[0].pcolormesh(times, freqs, mypsd, cmap='plasma', shading='gouraud')

    axes[0].set_xlim(xmin=0, xmax=mypsd.shape[-1])
    axes[0].set_yticks(np.rint(freqs[::10]))
    axes[0].set_ylabel('Freqs')

    pe_tau1 = pe[0, ...]
    plot_pe(pe_tau1, stag, axes=axes[1:3], mspe=False)
    axes[-1].plot(pred)
    plt.show()

sel_idxs = [1,2,3]
path = 'H:\\BABY\\working\subjects'
fnames =  os.listdir(path)
#fnames1 = [f for f in fnames if f.endswith('1')]
fnames2 = [f for f in fnames if f.endswith('2')] #filter folders

pe, stag, n1, _ = load_single_append(path, fnames2, typ = 'mspet1m3')
pe, stag = select_class_to_classif(pe, stag, sel_idxs=sel_idxs)
mypsd, stagpsd, n2, freqs =  load_single_append(path, fnames2, typ = 'psd')
mypsd, stagpsd = select_class_to_classif(mypsd, stagpsd, sel_idxs=sel_idxs)

#load predictions
pred2 = []
for sbj in fnames2:
    if not os.path.isfile(paths(typ='pred', sbj=sbj)):
        print 'Folder is empty'
        print sbj
        continue
    pred2.append(myload(typ='pred', sbj=sbj)[0][0] ) #[0][0] cause tuple with pred and stag

print all([all(stagpsd[i] == stag[i]) for i in range(len(stag))])

#ii = range(4)
#for i in ii:
i = 11
plot_data_ver2(pe[i], mypsd[i], stag[i], pred2[i])
