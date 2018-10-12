import plotly
import plotly.offline as py
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from config import raw_path
import matplotlib.pyplot as plt
#init_notebook_mode(connected=True)
import plotly.tools as tls
from mne import io
import copy


ch_names = io.read_raw_edf(raw_path + '104_2_correctFilter_2heogs_ref100.edf',
        preload=False).info['ch_names'][:-1]

data = pe[0]
s = copy.deepcopy(stag[0])
s = s.rename(columns = {'numeric':'ground_truth'})
pred = pred_pe[0][0]# 0st  pred_pe is predicted, 1st true
s['predicted'] = pred
s = s.drop(s.columns[0],1 ) # drop first col, nonnumeric


df = pd.DataFrame(data.transpose(), columns=ch_names)
df['average'] = df.mean(1)
df_cat = pd.concat([df, s], 1 )
df_cat = df_cat.dropna(inplace=False)
mapper = {1:'NREM', 2:'REM', 3:'WAKE'}
s = s.replace(to_replace=mapper)
s_match = s.nunique(1) == 1 #find matches
s= s.where(s_match, 'nan') #subste, fill nan

hover_text = s.apply(lambda r: '<br>'.join(['{}: {}'.format(c, r[c])
                                            for c in s.columns]), axis=1)
hover_text

fig, axes = plt.subplots(3,1)
df.plot(ax = axes[0])

plotly_fig = tls.mpl_to_plotly(fig)


axes[0].set_ylabel('Permutation Entropy')
time = range(len(stag))
axes[1].plot(stag['numeric'].values, 'r*', label = 'ground truth (Scholle)')
axes[1].set_xlim(0,len(stag))
axes[1].set_ylim(0.8,3.2)
axes[1].set_yticks([1,2,3])
axes[1].set_yticklabels(['N','R','W'], fontsize='large')
axes[1].legend()



color1 = '#9467bd'
color2 = '#F08B00'

trace1 = go.Scatter(
    x = df.index,
    y = df['stag'],
    name='stag',
    line = dict(
        color = color1
    )
)
trace2 = go.Scatter(
    x= df.index,
    y =df['pe'] ,
    name='PE',
    yaxis='y2',
    mode='markers'

)
data = [trace1, trace2]
layout = go.Layout(
    title= "XY",
    yaxis=dict(
        title='stag',
        titlefont=dict(
            color=color1
        ),
        tickfont=dict(
            color=color1
        )
    ),
    yaxis2=dict(
        title='PE',
        overlaying='y',
        side='right',
        titlefont=dict(
            color=color2
        ),
        tickfont=dict(
            color=color2
        )

    )

)
fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig, sharing = 'public')

#########################################################################
trace_1 = go.Scatter(
    x=df.index,
    y=df['F3'],
    hoverinfo = 'x+y',
    mode = 'markers',
    type ='scatter',
    name = 'F3',
  	xaxis = 'x',
  	yaxis = 'y',
    connectgaps = False,
    marker = dict(
        size = 5,
        color = 'rgb(64, 97, 139)',
        line = dict(
            width = 1,
            color = 'rgb(64, 97, 139)'
        )
    )
)

trace_2 = go.Scatter(
    x=df.index,
    y=df['numeric'],
    hoverinfo = 'x+y',
 	xaxis = 'x2',
    yaxis = 'y2',
    mode = 'lines+markers',
    name = 'stag',
    connectgaps = False,
    marker = dict(
        size = 5,
        color = 'rgb(117, 15, 7)',
        line = dict(
            width = 1,
            color = 'rgb(117, 15, 7)'
        )
    )
)


layout = go.Layout(
    title='Station ABC',
    xaxis = dict(
        rangeselector=dict(
            buttons = list([
                dict(count=1,
                     label='1min',
                     step='minute',
                     stepmode='backward'),
                dict(count=24,
                     label='24h',
                     step='hour',
                     stepmode='backward'),
            ])
        ),
        rangeslider=dict(),
        type='date',
        title='Date and Time'
    ),
    yaxis=dict(
        domain=[1,2]),
    yaxis2=dict(
        domain=[1,2]),

        )


fig = tls.make_subplots(rows=2, cols=1)

fig.append_trace(trace2, 1, 1)
fig.append_trace(trace1, 2, 1)
