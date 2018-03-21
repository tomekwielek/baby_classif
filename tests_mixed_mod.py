#fromhttps://github.com/statsmodels/statsmodels/issues/3500

from pandas import DataFrame

num_x = 15
num_s = 6
num_t = 5

x = np.repeat(np.linspace(0,30,num_x)[None], 2*num_s*num_t, axis=0).ravel()
group = np.hstack(( np.zeros(num_x*num_s*num_t, dtype = int),
                    np.ones(num_x*num_s*num_t, dtype = int)))
subject = np.repeat(np.arange(2*num_s), num_x*num_t)
tissue_sample = np.repeat(np.arange(2*num_s*num_t), num_x)

dat = DataFrame({'x':x, 'group':group, 'subject':subject, 'tissue_sample':tissue_sample})
dat.shape
dat.dtypes

factor_s = np.random.normal(scale=7, size=2*num_s)
factor_t = np.random.normal(scale=2, size=2*num_t*num_s)
error    = np.random.normal(scale=.3, size=2*num_x*num_s*num_t)

intercept = 5
group_effect = 6
slope = 0.3

dat['y'] = intercept + group_effect * dat.group + \
    factor_s[dat.subject] + factor_t[dat.tissue_sample] + \
    slope * dat.x + error

dat.shape
dat.dtypes
