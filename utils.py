from config import raw_path
import os
from mne import io
import re

fnames = os.listdir(raw_path)
fnames = sorted(filter(lambda x : x.endswith('_ref100.edf'), fnames))
chs = []
for fn in fnames:
    chs.append(io.read_raw_edf(raw_path+fn, preload=True).info['ch_names'])

chs_incl = ['F3', 'C3', 'O1', 'O2', 'C4', 'F4', 'ECG', 'VEOG', 'EMG', 'HEOG_l']
regexs = [re.compile(chs_incl[i]) for i in range(len(chs_incl))]

chs_pos = np.zeros((len(chs), len(chs_incl)))#empty arr for storing chs position
for sbj_id in range(len(chs)): #iterate over sbjs
    for i in range(len(chs_incl)): #iterate over channels
        match = np.where([compiled_reg.match(chs[sbj_id][i]) for compiled_reg in regexs])[0]
        chs_pos[sbj_id, match] = match
    print chs[sbj_id]
    print fnames[sbj_id]

zip(range(len(chs)), fnames, [len(chs[i]) for i in range(len(chs))], chs_pos)

        print match
        #chs_pos[sbj_id, i] = np.where([compiled_reg.search(chs[sbj_id][i]) for compiled_reg in regexs])[0]


    if all(compiled_reg.match(chs[i]) for compiled_reg in regexs):
        print("all matched")
    else:
        print '{} does not match'.format(i)


any(compiled_reg.match(mystring) for compiled_reg in reg_lst)



def sort_channels(data_unsorted, ch_names_unsorted):
    permut = np.array(ch_names_unsorted).argsort()
    data = data_unsorted[:,permut,:]
    return data, sorted(ch_names_unsorted)
