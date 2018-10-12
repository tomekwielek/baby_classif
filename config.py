#raw_path ='H:\\BABY\\data\\bv\\exp\\' #brain vision pp. data
raw_path ='H:\\BABY\\data\\'
base_path = 'H:\\BABY\\working\\subjects\\'
results_path='H:\\BABY\\results\\'
stag_fname = 'H:\\BABY\\Stages_inklPrechtl_corrected.xlsx'
report = 'H:\\BABY\\working\\report.html'
from mne.report import Report
report = Report(report)
#chs_incl = ['F3', 'C3', 'O1', 'O2', 'C4', 'F4', 'ECG', 'EMG', 'HEOG', 'VEOG']
chs_incl = ['F3', 'C3', 'O1', 'O2', 'C4', 'F4', 'ECG', 'EMG', 'HEOG_l', 'HEOG_r', 'VEOG']
#subjects - based on edf-s filenames [:5] bads excluded (see read_raw.py)
subjects = ['104_2', '108_1', '108_2', '110_1', '110_2', '111_1',
       '111_2', '112_1', '112_2', '113_1', '113_2', '114_1', '114_2',
       '117_1', '117_2', '118_1', '118_2', '119_1', '119_2', '202_1',
       '202_2', '203_1', '203_2', '204_1', '204_2', '205_2',
       '206_2', '208_2', '209_1', '209_2', '210_1',
       '210_2',  '212_1', '215_1', '215_2', '216_1', '216_2',
       '218_2', '219_1', '219_2', '221_2',
       '222_1', '222_2', '223_1', '223_2', '224_1', '224_2', '225_1',
       '225_2',  '226_2', '227_1', '231_1',
        '234_1', '234_2', '235_2', '236_1',
       '236_2', '238_2',  '239_2', '240_1']

pe_par = {'pet1m3' : {'embed':3, 'tau':1},
           'pet3m3' : {'embed':3, 'tau':3},
           'pet1m4' : {'embed':4, 'tau':1},
           'pet3m4' : {'embed':4, 'tau':3},
           'mspet1m3' : {'embed':3, 'tau':1},
           'mspet_ord1m3' : {'embed':3, 'tau':1},
           'pet1m3_stag_uncorr' : {'embed':3, 'tau':1},
           'pet3m3_stag_uncorr' : {'embed':3, 'tau':3},
           'pet1m4_stag_uncorr' : {'embed':4, 'tau':1},
           'pet1m3_30hz' : {'embed':3, 'tau':1}}

def paths(typ, c=None, sbj='sbj_av'):
    import os
    import os.path as op
    sbj = 'VP'+ str('%02d') % sbj if isinstance(sbj, int) else sbj
    if typ in ['raws']:
        this_path = raw_path
    else:
        this_path = op.join(base_path, sbj, typ)
    path_template = dict(
        raws=this_path,
        base_path=base_path,
        data_path=this_path,
        pet1m3= op.join(this_path, '%s.txt' % sbj),
        mspet1m3= op.join(this_path, '%s.txt' % sbj),
        mspet_ord1m3= op.join(this_path, '%s.txt' % sbj),
        pet3m3= op.join(this_path, '%s.txt' % sbj),
        pet1m4= op.join(this_path, '%s.txt' % sbj),
        pet3m4= op.join(this_path, '%s.txt' % sbj),
        pet1m3_stag_uncorr= op.join(this_path, '%s.txt' % sbj),
        pet3m3_stag_uncorr= op.join(this_path, '%s.txt' % sbj),
        pet1m4_stag_uncorr= op.join(this_path, '%s.txt' % sbj),
        psd = op.join(this_path, '%s.txt' % sbj),
        pet1m3_30hz = op.join(this_path, '%s.txt' % sbj),
        pred = op.join(this_path, '%s.txt' % sbj))

    this_file = path_template[typ]
    # Create subfolder if necessary
    folder = os.path.dirname(this_file)
    if (folder != '') and (not op.exists(folder)):
        os.makedirs(folder)
    return this_file

def myload(typ, sbj, c=None):
    import pickle
    fname = paths(typ=typ, c=c, sbj=sbj)
    if typ in ['pet1m3', 'pet3m3', 'pet1m4', 'pet3m4', 'pet1m3_stag_uncorr',
                'pet3m3_stag_uncorr', 'pet1m4_stag_uncorr', 'mspet1m3', 'mspet_ord1m3',
                'psd', 'pet1m3_30hz', 'pred']:
        with open(fname, 'rb') as f:
            #out = pickle.load(f, encoding='latin1')
            out = pickle.load(f)
    else:
        raise NotImplementedError()
    return out

def mysave(var, typ, sbj='sbj_av',  overwrite=True):
    import os.path as op
    import pickle
    fname = paths(typ, sbj=sbj)
    # check if file exists
    if op.exists(fname) and not overwrite:
        print('%s already exists. Skipped' % fname)
        return False
    elif typ in ['pet1m3', 'pet3m3', 'pet1m4', 'pet3m4', 'pet1m3_stag_uncorr',
                'pet3m3_stag_uncorr', 'pet1m4_stag_uncorr', 'mspet1m3', 'mspet_ord1m3',
                'psd', 'pet1m3_30hz', 'pred']:
        with open(fname, 'wb') as f:
            pickle.dump(var, f)
    else:
        raise NotImplementedError()
        return False
