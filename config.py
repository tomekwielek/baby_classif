#raw_path ='H:\\BABY\\data\\bv\\exp\\' #brain vision pp. data
raw_path ='H:\\BABY\\data\\'
base_path = 'H:\\BABY\\working\\subjects\\'
results_path='H:\\BABY\\results\\'
stag_fname = 'H:\\BABY\\data\staging\\Stages_inklPrechtl_corrected.xlsx'
report = 'H:\\BABY\\working\\report.html'
from mne.report import Report
report = Report(report)

chs_incl = ['F3', 'C3', 'O1', 'O2', 'C4', 'F4', 'ECG', 'EMG', 'HEOG_l', 'HEOG_r', 'VEOG'] #ref100 2heog
#chs_incl = ['F3', 'C3', 'O1', 'O2', 'C4', 'F4', 'ECG', 'EMG', 'HEOG', 'VEOG']

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
            'mspet1m3_nofilt' : {'embed':3, 'tau':1},

           'mspet_ord1m3' : {'embed':3, 'tau':1},
           'pet1m3_stag_uncorr' : {'embed':3, 'tau':1},
           'pet3m3_stag_uncorr' : {'embed':3, 'tau':3},
           'pet1m4_stag_uncorr' : {'embed':4, 'tau':1},
           'pet1m3_30hz' : {'embed':3, 'tau':1}}

#bad_sbjs_1 and bad_sbjs_2 - 20Hz artifacts present
bad_sbjs_1 = ['240_1', '235_1', '232_1', '226_1', '221_1', '220_1', '218_1', '214_1', '211_1', '236_1', \
                '234_1', '231_1', '225_1', '224_1', '223_1', '222_1', '219_1', '216_1', '215_1', '212_1', \
                '210_1', '209_1', '204_1', '203_1', '202_1', '118_1', '117_1']
bad_sbjs_2 = ['111_2','112_2', '113_2', '118_2', '119_2', '202_2', '204_2', '205_2', '206_2', '208_2', '209_2','215_2', '216_2',\
                '218_2','219_2', '221_2', '222_2', '224_2', '225_2', '226_2', '227_2','234_2', '236_2', '238_2', '239_2', \
                    '231_2', '212_2', '214_2', '220_2', '232_2']

# Subjects with recorgins-length-issues (e.g.:first baseline 4.749sec too short). ONLY for hd relevant.
length_issue_sbjs = [110_2, 114_2, 117_1, 201_1, 202_1, 204_1, 205_1, 207_1, 208_1, 220_2, 223_2, 224_2, \
                    226_2, 223_1, 235_1, 236_2]

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
        mspet1m3_nofilt= op.join(this_path, '%s.txt' % sbj), #1eog no filt
        mspet1m3_nofilt_ref100= op.join(this_path, '%s.txt' % sbj), #2eog no filt
        mspet_ord1m3= op.join(this_path, '%s.txt' % sbj),
        pet3m3= op.join(this_path, '%s.txt' % sbj),
        pet1m4= op.join(this_path, '%s.txt' % sbj),
        pet3m4= op.join(this_path, '%s.txt' % sbj),
        pet1m3_stag_uncorr= op.join(this_path, '%s.txt' % sbj),
        pet3m3_stag_uncorr= op.join(this_path, '%s.txt' % sbj),
        pet1m4_stag_uncorr= op.join(this_path, '%s.txt' % sbj),
        psd = op.join(this_path, '%s.txt' % sbj),
        psd_hd = op.join(this_path, '%s.txt' % sbj),
        psd_v2 = op.join(this_path, '%s.txt' % sbj),
        pe_hd = op.join(this_path, '%s.txt' % sbj),
        psd_nofilt = op.join(this_path, '%s.txt' % sbj), #1eog no filt
        psd_nofilt_ref100 = op.join(this_path, '%s.txt' % sbj), #2eog no filt
        psd_notch = op.join(this_path, '%s.txt' % sbj),
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
                'pet3m3_stag_uncorr', 'pet1m4_stag_uncorr', 'mspet1m3','mspet1m3_nofilt', 'mspet_ord1m3',
                'mspet1m3_nofilt_ref100', 'psd_nofilt_ref100', 'psd_v2',
                'psd', 'psd_hd', 'pe_hd', 'psd_nofilt', 'psd_notch', 'pet1m3_30hz', 'pred']:
        with open(fname, 'rb') as f:
            out = pickle.load(f, encoding='latin1')
            #out = pickle.load(f)
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
                'pet3m3_stag_uncorr', 'pet1m4_stag_uncorr', 'mspet1m3','mspet1m3_nofilt', 'mspet_ord1m3',
                'mspet1m3_nofilt_ref100', 'psd_nofilt_ref100', 'psd', 'psd_nofilt', 'psd_notch',
                'psd_hd', 'pe_hd', 'pet1m3_30hz', 'pred', 'psd_v2']:
        with open(fname, 'wb') as f:
            pickle.dump(var, f)
    else:
        raise NotImplementedError()
        return False


markers35 = ['D221', 'D222', 'D223', 'D224', 'D225', 'DI92', 'D201', 'D202',
'D203', 'D204', 'D205', 'DI93', 'D121', 'D122', 'D123', 'D124', 'D125', 'DI91', 'D101',
'D102', 'D103', 'D104', 'D105', 'DI95', 'DI94']

markers27 = ['D101', 'D102', 'D103', 'DI94', 'D221', 'D222', 'D223', 'DI92', 'D201', 'D202',
'D203', 'DI93', 'D121', 'D122', 'D123', 'DI95', 'DI91']

match_names27 = ['202_2_S',
                 '203_1_S',
                 '203_2_S',
                 '204_2_S',
                 '205_2_S',
                 '206_2_S',
                 '208_2_S',
                 '209_2_S',
                 '210_1_S',
                 '210_2_S',
                 '211_1_S',
                 '211_2_S',
                 '212_1_S',
                 '212_2_S',
                 '214_1_S',
                 '214_2_S',
                 '215_1_S',
                 '215_2_S',
                 '216_2_S',
                 '218_1_S',
                 '218_2_S',
                 '219_1_S',
                 '219_2_S',
                 '220_1_S',
                 '220_2_S',
                 '221_1_S',
                 '221_2_S',
                 '223_1_S',
                 '223_2_S',
                 '224_1_S',
                 '224_2_S',
                 '225_1_S',
                 '225_2_S',
                 '226_1_S',
                 '227_1_S',
                 '227_2_S',
                 '231_1_S',
                 '231_2_S',
                 '232_1_S',
                 '232_2_S',
                 '235_2_S',
                 '236_1_S',
                 '238_2_S',
                 '239_2_S',
                 '240_1_S']
match_names35 = ['104_2_S',
                 '108_1_S',
                 '108_2_S',
                 '110_1_S',
                 '110_2_S',
                 '111_1_S',
                 '111_2_S',
                 '112_1_S',
                 '112_2_S',
                 '114_1_S',
                 '118_1_S',
                 '118_2_S',
                 '119_1_S',
                 '119_2_S']
