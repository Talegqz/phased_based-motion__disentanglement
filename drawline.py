
import matplotlib.pyplot as plt
import numpy as np
from  mode_shape import  make_dir
from scipy.interpolate import spline


num = 300


fre = 2
scale = 1
x = np.arange(0,101)
mode1 = np.sin(x*2*np.pi/100)
mode2 = np.sin(x*np.pi/100)
xnew = np.linspace(x.min(),x.max(),300)
#4 0.01


result_path = 'data/1+2_scale_%0.1f_fre_%d'%(scale,fre)
make_dir(result_path)


count = 0
for i in range(0,1):
    for w in range(0,801):
        y = (mode1*np.sin(fre*np.pi*w/100)+mode2*np.sin(fre*w*np.pi/(50)))
        y = y*scale
        xsmoo = spline(x, y, xnew)
        plt.figure()
        plt.xlim(0, 105)
        plt.ylim(-50, 50)
        plt.axis('off')
        plt.plot(xnew, xsmoo, linewidth=15)

        plt.savefig(result_path+'/%d.png'%(count))
        count +=1
        plt.close()




#
# print(y)
#
#
#
#
#
#
# plt.show()
#
