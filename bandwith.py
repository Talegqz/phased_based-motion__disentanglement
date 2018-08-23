from perceptual.filterbank import Steerable
import cv2
from PIL import Image
import numpy as np
import os
import util.util as tool
import math
datapath = r'D:\database\action1-person1-white'
basic = '0'

frame1 = '5'


result_path = 'result/bandwith_frame'+frame1

def get_mask(size,w,d):

    rowcenter = int(size[0] / 2)
    columncenter = int(size[0] / 2)


    mask1 = np.ones(size, np.uint8)

    mask1[rowcenter-d:rowcenter+d,columncenter-d:columncenter+d] = 0

    mask2 = np.zeros(size, np.uint8)

    mask2[rowcenter - d-w:rowcenter + d+w, columncenter - d-w:columncenter + d+w] = 1

    mask = mask1 * mask2

    return mask


def make_dir(result_path):

    if os.path.exists(result_path):
        pass
    else:
        os.mkdir(result_path)

basic = cv2.imread(datapath+'/frame/%s.png'%basic, cv2.IMREAD_GRAYSCALE)
make_dir(result_path)
im1 = cv2.imread(datapath+'/frame/%s.png'%frame1, cv2.IMREAD_GRAYSCALE)
im1 = np.array(im1)
s = Steerable(4,2)


im1pic =Image.fromarray(im1)
im1pic.save(result_path+'/im1_gray.png')

tsize = min(im1.shape)
coeff1= s.buildSCFpyr(im1)

basicco = s.buildSCFpyr(basic)

# coeff2 = s.buildSCFpyr(im2)

def save_pic(array,file):
    array = array.astype(np.uint8)
    pic = Image.fromarray(array,"RGB")
    # pic = pic.convert('RGB')

    pic.save(file)
def show_pic(array):
    min = np.min(array)
    max = np.max(array)
    array = (array-min)/(max-min)
    print(np.array(array))
    array = array*255
    #
    # array = array.astype(np.uint8)
    pic = Image.fromarray(array)
    pic.show()
def visual(A,P,path):
    R =[]
    G =[]
    B =[]
    row =len(A)
    column = len(A[0])
    for i in range(row):
        r=[]
        g=[]
        b=[]
        for j in range(column):
            phase = P[i][j]*180/math.pi
            r1,g1,b1 = hsv2rgb(phase,A[i][j],1)

            r.append(r1)
            g.append(g1)
            b.append(b1)

        R.append(r)
        G.append(g)
        B.append(b)

    R = np.array(R)
    G = np.array(G)
    B = np.array(B)

    all = []
    all.append(R)
    all.append(G)
    all.append(B)
    all = np.array(all)
    all = np.transpose(all,(1,2,0))

    save_pic(all,path)




    pass

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b



for i in range(1,3):

    print("pyramid%d"%i)
    pyresult = result_path+'/pyramid%d'%i
    make_dir(pyresult)


    for j in range(0,2):
        rangepath = pyresult + '/nband%d' % j
        make_dir(rangepath)

        amplitude = ((coeff1[i][j].real)**2+(coeff1[i][j].imag)**2)**0.5
        phase = np.arctan(coeff1[i][j].imag/coeff1[i][j].real)

        basicphase = np.arctan(basicco[i][j].imag/basicco[i][j].real)
        show_pic(phase)
        phase = phase-basicphase
        show_pic(phase)
        for trange in range(1,3):
            widepith = rangepath + '/width'+str(trange)
            make_dir(widepith)
            for dis in range(1,tsize):
                thisshape = np.array(im1.shape)
                # thisshape = thisshape.astype(np.int)
                mask = get_mask((thisshape/(2**(i-1))).astype(np.int),trange,dis)
                phase_f = np.fft.fft2(phase)
                phase_fs = np.fft.fftshift(phase_f)
                # phase_fs = phase_fs*mask
                new_phase = np.fft.ifftshift(phase_fs)
                new_phase = np.fft.ifft2(new_phase)
                # show_pic(amplitude)
                show_pic(new_phase.real)
                print('-----------------------------------------------------------')
                # print(new_phase)
                visual(amplitude,new_phase,widepith+'/%d.png'%dis)


















print('')