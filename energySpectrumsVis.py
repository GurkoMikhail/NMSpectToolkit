from pyqtgraph.Qt import QtGui
import numpy as np
import pyqtgraph as pg
from hepunits import*

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

energy_spectrum = np.load('Numpy data/lung_cancer_new_sum_spectrum.npy')
energy = energy_spectrum[:, 0]
counts = energy_spectrum[:, 1]

lower_ground = np.searchsorted(energy, 126*keV)
upper_ground = np.searchsorted(energy, 154*keV)
lower_peak = np.searchsorted(energy, 140*keV)
upper_peak = np.searchsorted(energy, 141*keV)

ground_sum = counts[lower_ground:upper_ground].sum()
peak_sum = counts[lower_peak:upper_peak].sum()

print(100*(1 - peak_sum/ground_sum))

pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.resize(400, 300)

p = win.addPlot(title='Energy spectrum')
p.plot(x=energy/eV, y=counts).setPen((0, 0, 255, 255))
# p.setLogMode(y=True)
p.showGrid(x=True, y=True)
p.setLabel('bottom', 'Energy', units='eV')
p.setLabel('left', 'Count')

win.show()
QtGui.QApplication.instance().exec_()