from Animation3D import *

data = pd.read_csv('../data/data.csv')

data = data.loc[:, ('ts', 'PositionX', 'PositionY', 'PositionZ', \
                    'rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat')]

anim = Animation3D(data, speed=1., fps=25, view=(45, -90))

plt.show()