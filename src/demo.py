from Animation3D import *

data = pd.read_csv('data.csv')

anim = Animation3D(data, speed=1., fps=25, view=(45, -90))

plt.show()