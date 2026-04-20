from os.path import dirname, join as pjoin
import scipy.io as sio
import matplotlib as mpl
mpl.use("Qt5Agg")  # call before any other mpl imports/inits
import matplotlib.pyplot as plt



datadir = "/media/Volume/Data/GrosmarkAD/Achilles_10252013"

filename = "Achilles_10252013.behavior.mat"

data = sio.loadmat(pjoin(datadir,filename))


data.keys()


data["behavior"]['timestamps'][0][0].shape
errorPerMarker

data["behavior"][0][0]['timestamps']
data["behavior"][0][0][4]["errorPerMarker"]
data["behavior"][0][0][4]['description']


# ????
data["behavior"][0][0][5][0][0][0][0][2]["x"]
data["behavior"][0][0][5][0][0][0][0][2]["y"]
data["behavior"][0][0][5][0][0][0][0][2]["z"]

#errorPerMarker
#linear
#optitrack
#description
#mapping

#dtype=[('x', 'O'), ('y', 'O'), ('z', 'O'), ('orientation', 'O'), ('errorPerMarker', 'O'), ('timestamps', 'O'), ('mapping', 'O')]),


#data["behavior"]['orientation'][0][0][0][0]['x'].shape
#data["behavior"]['orientation'][0][0][0][0]['y'].shape
#data["behavior"]['orientation'][0][0][0][0]['z'].shape
#data["behavior"]['orientation'][0][0][0][0]['w'].shape
#data["behavior"]["orientation"][0][0][0][0][4][0] 


plt.plot(data["behavior"]["position"][0][0][0][0][1])
position
plt.show




