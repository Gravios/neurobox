# because I cannot trust ein

import os
import platform
import sys
from neurobox.io.load_xml    import load_xml
from neurobox.io.load_binary import load_binary
from neurobox.io.load_clu_res import load_clu_res
from neurobox.io.load_position_motive_csv import load_position_motive_csv

match platform.system():
    case 'Linux':
        os.environ['NEUROBOX_CODE'] = "/media/nas/code/python"
        os.environ['NEUROBOX_DATA'] = "/media/data/data/"
    case 'Windows':
        os.environ['NEUROBOX_CODE'] = "N:/gravio/code/python"
        os.environ['NEUROBOX_DATA'] = "C:/Users/justi/data"
NEUROBOX_CODE = os.getenv('NEUROBOX_CODE')
NEUROBOX_DATA = os.getenv('NEUROBOX_DATA')
sys.path.insert(0, NEUROBOX_CODE)


# import matplotlib as mpl
# mpl.use("Qt5Agg")
# mpl.use("TkAgg")
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


session_name = "A2929-200711"

fbase = NEUROBOX_DATA + f"/{session_name}/{session_name}"
fname_xml = fbase + ".xml"
fname_dat = fbase + ".dat"
fname_pos = fbase + ".csv"
#fname_wat = fbase + "-temp_wh.dat"

res, clu, cmap = load_clu_res(fbase)


par = load_xml(fname_xml)
print(par.acquisitionSystem.peek())

dat = load_binary(fname_dat,[0,1],par);
#wat = load_binary(fname_wat,[0,1],par);

dat = dat.transpose()

#wat = wat.transpose()


#%matplotlib inline
plt.figure(num=1,figsize=(12,5))
plt.plot(dat[100000:151100])
#plt.plot(wat[100000:100100,1])
plt.show

#import numpy as np
from scipy import signal
#from scipy.fft import fftshift


fs = 2*10e3
win = ('hamming')
nps = 2**14
nlp = int(2**14*0.5)
nfft = 2**15
f, t, Sxx = signal.spectrogram(dat[15000000:20000000,0], fs,win,nps,nlp,nfft)

plt.figure(num=1,figsize=(18,5))
plt.pcolormesh(t, f[0:40], Sxx[0:40,:], shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


fs = 2*10e3
win = ('hamming')
nps = 2**14
nlp = int(2**14*0.5)
nfft = 2**15
f, t, Sxx = signal.spectrogram(wat[15000000:20000000,0], fs,win,nps,nlp,nfft)
plt.figure(num=1,figsize=(18,5))
plt.pcolormesh(t, f[0:40], Sxx[0:40,:], shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

pos = load_position_motive_csv(fname_pos)

fig = px.scatter_3d(pos, x='X', y='Z', z='Y',color="Phi")
fig.show()

import plotly.graph_objects as go
fig = go.Scatter3d(pos, x='X', y='Z', z='Y',color="Phi")
fig = go.scattergl()

fig = go.Figure()
fig.add_trace(
    go.Scatter3d(x=pos['X'], y=pos['Z'], z=pos['Y'])
    )
fig.show()


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pos[:, 5]*1000,pos[:, 7]*1000, pos[:, 6]*1000, label='Position', marker='o')
ax.set_title('Position Data (X vs Y)')
ax.set_xlabel('X Position (meters)')
ax.set_ylabel('Y Position (meters)')
#plt.legend()
#plt.grid(True)
plt.show()

# PYQTGRAPH
pos = load_position_motive_csv(fname_pos)
import pyqtgraph as pg
import numpy as np
plotWidget = pg.opengl.GLViewWidget();
plotWidget = pg.plot(title="Three plot curves")
pg.opengl.GLScatterPlotItem(pos=pos[['X','Z','Y']].values)
pg.mkQApp().exec_()

