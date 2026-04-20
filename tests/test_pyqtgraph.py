# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:58:45 2023

@author: justi
"""

# import pyqtgraph as pg
# import numpy as np
# x = np.arange(1000)
# y = np.random.normal(size=(3, 1000))
# plotWidget = pg.plot(title="Three plot curves")
# for i in range(3):
#     plotWidget.plot(x, y[i], pen=(i,3))
# pg.mkQApp().exec_()

def test():
    import os
    import platform
    match platform.system():
        case 'Linux':
            os.environ['PYMTAX_CODE'] = "/media/cloud/code/python"
            os.environ['PYMTAX_DATA'] = "/media/data/data/"
        case 'Windows':
            os.environ['PYMTAX_CODE'] = "Z:/code/python"
            os.environ['PYMTAX_DATA'] = "C:/Users/justi/data"
    PYMTAX_CODE = os.getenv('PYMTAX_CODE')
    PYMTAX_DATA = os.getenv('PYMTAX_DATA')
    os.environ['SESSION_MANAGER'] = "gravio"
     import sys
    sys.path.insert(0, PYMTAX_CODE)
    from pymtax.io.load_position_motive_csv import load_position_motive_csv
    session_name = "A2929-200711"
    fbase = PYMTAX_DATA + f"/{session_name}/{session_name}"
    fname_pos = fbase + ".csv"
    # PYQTGRAPH
    pos = load_position_motive_csv(fname_pos)
    import pyqtgraph as pg
    from pyqtgraph.opengl import GLViewWidget as view
    from pyqtgraph.opengl import GLScatterPlotItem as scatter3
    from pyqtgraph.opengl import GLGridItem as item
    import numpy as np
    QApp = pg.mkQApp()
    window = view();
    window.setObjectName('3D')
    splot = scatter3(pos=pos[['X','Z','Y']].values)
    for item in window.items:
        item._setView(None)
    window.items = []
    window.update()
    window.addItem(splot)
    window.addItem(item())
    window.show()
    QApp.exec_()
    QApp.quit()



test()


