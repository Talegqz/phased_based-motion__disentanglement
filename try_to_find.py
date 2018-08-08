from perceptual.filterbank import Steerable
import cv2
from PIL import Image
import numpy as np
import os
import util.util as tool
datapath = r'D:\database\action1-person1-white'
frame1 = '0'
frame2 = '3'
result_path = 'result/frame'+frame1+'and'+frame2

def make_dir(result_path):

    if os.path.exists(result_path):
        pass
    else:
        os.mkdir(result_path)

make_dir(result_path)
im1 = cv2.imread(datapath+'/frame/%s.png'%frame1, cv2.IMREAD_GRAYSCALE)
im1 = np.array(im1)
im2 = cv2.imread(datapath+'/frame/%s.png'%frame2, cv2.IMREAD_GRAYSCALE)
s = Steerable(5)
def get_heatmap(path):


    heatnum = np.load(path)
    heatnumpose = tool.hmp2pose_by_numpy(heatnum)
    heatmap = tool.pose2im_all(heatnumpose)
    return heatmap


im1heat = Image.fromarray(get_heatmap(datapath+'/heatmap/%s.npy'%frame1))



im2heat = Image.fromarray(get_heatmap(datapath+'/heatmap/%s.npy'%frame2))

im1pic =Image.fromarray(im1)
im1pic.save(result_path+'/im1_gray.png')



im2pic =Image.fromarray(im2)
im2pic.save(result_path+'/im2_gray.png')


coeff1 = s.buildSCFpyr(im1)
coeff2 = s.buildSCFpyr(im2)

def save_pic(array,file):
    pic = Image.fromarray(array)
    pic = pic.convert('RGB')

    pic.save(file)

def become_0_255(array):


    return (array-np.min(array))/(np.max(array)-np.min(array))*255

for i in range(1,4):
    pyresult = result_path+'/pyramid%d'%i
    make_dir(pyresult)

    im1heat.save(pyresult+"/%s_heatmap.png"%frame1)
    im2heat.save(pyresult + "/%s_heatmap.png" % frame2)
    for j in range(0,4):
        realpath = pyresult+'/real'
        make_dir(realpath)
        save_pic(become_0_255(coeff1[i][j].real), realpath+'/im1_%d_real.png'%j)
        save_pic(become_0_255(coeff2[i][j].real), realpath + '/im2_%d_real.png'%j)
        delta = become_0_255(coeff2[i][j].real - coeff1[i][j].real)
        save_pic(delta, realpath + '/real_im1_minus_im2_in%d.png' % j)

        imagpath = pyresult + '/ima'
        make_dir(imagpath)

        save_pic(become_0_255(coeff1[i][j].imag), imagpath + '/im1_%d_imag.png' % j)
        save_pic(become_0_255(coeff2[i][j].imag), imagpath + '/im2_%d_imag.png' % j)

        delta = become_0_255((coeff2[i][j].imag - coeff1[i][j].imag))
        save_pic(delta, imagpath + '/imag_im1_minus_im2_in%d.png' % j)

        arcpath = pyresult + '/arc'
        make_dir(arcpath)
        arc1 = become_0_255(np.arctan(coeff1[i][j].imag/coeff1[i][j].real))
        arc2 = become_0_255(np.arctan(coeff2[i][j].imag/coeff2[i][j].real))

        save_pic(arc1, arcpath + '/im1_%d_arc.png' % j)
        save_pic(arc2, arcpath + '/im2_%d_arc.png' % j)

        delta = become_0_255((arc1 - arc2))
        save_pic(delta, arcpath + '/arc_im1_minus_im2_in%d.png' % j)


        # for p in range(0,len(coeff1[i])):
        #     for k in range(0,len(coeff1[i][j])):
        #         if coeff1[i][j][p][k]!=coeff2[i][j][p][k]:
        #             print(i,j,p,k,coeff1[i][j][p][k],coeff2[i][j][p][k])








print('')