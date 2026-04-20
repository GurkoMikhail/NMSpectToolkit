import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = pg.mkQApp("ImageView Example")

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.setWindowTitle('pyqtgraph example: ImageView')

data = np.loadtxt('Dat phantoms/activity_map.dat').reshape((128, 128, 128), order='F')
print(np.unique(data))

values = {
    200: 40,
    1200: 1000*3,
    700: 550,
    10000: 7000
}

for value, new_value in values.items():
    mask = data == value
    print(f'Number of {value} = {np.count_nonzero(mask)}')
    data[mask] = new_value

np.savetxt('Dat phantoms/activity_map_heart_high_intestines.dat', data.ravel(order='F'),  fmt='%i', delimiter='\t')


# print(data.sum())
# data = np.load('Numpy data/lung_cancer.npy')
# print(data.sum())
# data = np.loadtxt('Dat data/heart_32/8.dat').reshape((128, 128), order='F')

## Display the data and assign each frame a time value from 1.0 to 3.0
imv.setImage(data)
win.show()

if __name__ == '__main__':
    pg.exec()
