from perceptual.filterbank import Steerable
import cv2
from PIL import Image
import numpy as np
import os
import util.util as tool
import math


datappp = '1+2_scale_0.1_fre_2'


datapath = 'D:/database/action1-person1-white/frame'

datapath = 'data/'+datappp

frames_num = 500
basicframe = 0
start = 1
timeslot = 1
result_path = 'result/%s_start%d_nums%d_step%d' % (datappp,start, frames_num, timeslot)


# result_path = 'result/cine_video_start%d_nums%d_step%d' % (start, frames_num, timeslot)
def get_frame(num):
    pic = Image.open(datapath + '/%d.png' % num)
    pic = pic.convert('L')
    return np.array(pic)


pynums = 2
bands = 2
basicframepic = get_frame(basicframe)
image_w = basicframepic.shape[0]
image_h = basicframepic.shape[1]
csp = Steerable(2 + pynums, bands)
basiccsp = csp.buildSCFpyr(basicframepic)



def get_pymairds_AandP_localmotion():
    AandPandlocalmotion = []
    for i in range(start, start + timeslot * frames_num, timeslot):
        thiscsp = dict()
        thiscsp['A'] = []
        thiscsp['P'] = []
        thiscsp['L'] = []
        pic = get_frame(i)
        pic_csp = csp.buildSCFpyr(pic)
        for pyrmaid in range(1, 1 + pynums):
            nbandsA = []
            nbandsP = []
            nbandsL = []
            for nband in range(0, bands):
                a = ((pic_csp[pyrmaid][nband].real) ** 2 + (pic_csp[pyrmaid][nband].imag) ** 2) ** 0.5

                phase = np.arctan2(pic_csp[pyrmaid][nband].imag ,pic_csp[pyrmaid][nband].real)
                basicphase = np.arctan2(basiccsp[pyrmaid][nband].imag,basiccsp[pyrmaid][nband].real)
                p = phase - basicphase

                nbandsA.append(a)
                nbandsP.append(p)
                lllllllll = a ** 2 * p
                # print(lllllllll)
                nbandsL.append(lllllllll)
            nbandsP = np.array(nbandsP)
            nbandsL = np.array(nbandsL)
            nbandsA = np.array(nbandsA)

            thiscsp['A'].append(nbandsA)
            thiscsp['P'].append(nbandsP)
            thiscsp['L'].append(nbandsL)
        AandPandlocalmotion.append(thiscsp)
    return AandPandlocalmotion


def change_demension(APL):
    allbands_local = []
    a = []
    p = []
    for py in range(0, pynums):
        local = []
        la = []
        lp = []
        for i in range(len(APL)):
            local.append(APL[i]['L'][py])
            la.append(APL[i]['A'][py])
            lp.append(APL[i]['P'][py])
        allbands_local.append(np.array(local))
        a.append(np.array(la))
        p.append(np.array(lp))
    return allbands_local, a, p


def make_dir(result_path):
    if os.path.exists(result_path):
        pass
    else:
        os.mkdir(result_path)


def get_modepic(local_motion):
    a = []
    # for i in range(len(local_motion)):
    #     the = local_motion[i]
    #     the = np.transpose(the,(1,2,3,0))
    #     newthe = []
    #     for j in range(len(the)):
    #         newthej = []
    #         for k in range(len(the[j])):
    #             newthek= []
    #             for p in range(len(the[j][k])):
    #                 # print(the[j][k][p].shape)
    #                 # the[j][k][p] = the[j][k][p].astype(np.complex128)
    #                 affff =  np.fft.fft(the[j][k][p])
    #                 # print(the[j][k][p])
    #                 # print(affff)
    #                 newthek.append(affff)
    #             newthej.append(newthek)
    #         newthe.append(newthej)
    #     newthe = np.array(newthe)
    #
    #     # print(newthe.shape)
    #     newthe = np.transpose(newthe,(3,0,1,2))
    #     a.append(newthe)
    for i in range(len(local_motion)):
        a.append(np.fft.fft(local_motion[i],axis=0))

    return a

    # return np.fft.fft(local_motion, axis=0)


def visual(A, P, psource, path):
    A = normalize(A)
    # print(A)
    # print(np.max(A))
    # print(np.min(A))
    # print(P)
    R = []
    G = []
    B = []
    row = len(A)
    column = len(A[0])
    for i in range(row):
        r = []
        g = []
        b = []
        for j in range(column):
            phase = P[i][j]
            # imag = psource[i][j].imag
            # real = psource[i][j].real
            # if real < 0 and imag > 0:
            #     phase = phase + np.pi
            # elif imag < 0 and imag<0:
            #     phase = phase+np.pi
            # elif imag<0 and real>0:
            #     phase = phase+2*np.pi
            # elif imag==0 and real>0:
            #     phase = 0
            # elif imag==0 and real<0:
            #     phase = np.pi
            # elif real==0 and imag<0:
            #     phase = 1.5*np.pi
            # elif real ==0 and imag >0:
            #     phase = 0.5*np.pi
            # elif real==0 and imag==0:
            #     phase = 0
            if phase>=0:
                phase = 180 * (phase) / np.pi
            else:
                phase = 180 * (phase+2*np.pi) / np.pi
            # phase = 180*(phase+np.pi)/np.pi
            aaa = A[i][j]
            r1, g1, b1 = hsv2rgb(phase, 1, aaa)

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
    all = np.transpose(all, (1, 2, 0))

    save_pic(all, path)

    pass


def hsv2rgb(h, s, v):
    # print(h)
    # print(s)
    # print(v)
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
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def save_pic(array, file):
    array = array.astype(np.uint8)
    pic = Image.fromarray(array, "RGB")
    # pic = pic.convert('RGB')
    # print(array)
    pic.save(file)


def save_grad(array, file):
    array = normalize(array) * 255
    pic = Image.fromarray(array)
    pic = pic.convert('RGB')
    pic.save(file)


def cal_A(mat):
    return (mat.real ** 2 + mat.imag ** 2) ** 0.5


def cal_P(mat):
    return np.arctan2(mat.imag ,mat.real)


def normalize(array):
    return (array - np.min(array)+0.0000000000000000000000000000000000000000000000000000000000001) / (np.max(array) - np.min(array)+0.0000000000000000000000000000000000000000000000000000000000001)


def visual_mode(frequency):
    make_dir(result_path)
    # shape  (pynums,nums,bands,size,size)
    for i in range(pynums):
        pynumspath = result_path + '/pynums%d' % (i + 1)
        make_dir(pynumspath)
        for band in range(bands):
            bandpath = pynumspath + '/orientation%d/' % band
            make_dir(bandpath)
            for fre in range(frames_num):
                now_value = frequency[i][fre][band]
                # print(now_value)
                A = cal_A(now_value)
                P = cal_P(now_value)

                visual(A, P, frequency[i][fre][band], bandpath + 'fre%d.png' % fre)


def visual_phase(ppp):
    make_dir(result_path)
    phasepath = result_path + '/phase'
    make_dir(phasepath)
    # shape  (pynums,nums,bands,size,size)
    for i in range(pynums):
        pynumspath = phasepath + '/pynums%d' % (i + 1)
        make_dir(pynumspath)
        for band in range(bands):
            bandpath = pynumspath + '/orientation%d/' % band
            make_dir(bandpath)
            for fre in range(frames_num):
                now_value = ppp[i][fre][band]

                save_grad(now_value, bandpath + 'fre%d.png' % fre)

    pass


def visual_weighted_phase(wp):
    make_dir(result_path)
    phasepath = result_path + '/weighted_phase'
    make_dir(phasepath)
    # shape  (pynums,nums,bands,size,size)
    for i in range(pynums):
        pynumspath = phasepath + '/pynums%d' % (i + 1)
        make_dir(pynumspath)
        for band in range(bands):
            bandpath = pynumspath + '/orientation%d/' % band
            make_dir(bandpath)
            for fre in range(frames_num):
                now_value = wp[i][fre][band]

                save_grad(now_value, bandpath + 'fre%d.png' % fre)

    pass


if __name__ == '__main__':
    make_dir(result_path)

    apl = get_pymairds_AandP_localmotion()
    localmotion, a, p = change_demension(apl)
    print('finished csp')


    # for i in localmotion:
    #     for j in i:
    #         for
    #         print(j)
    frequency = get_modepic(localmotion)
    print('finished fft')
    visual_mode(frequency)
    visual_weighted_phase(localmotion)
    visual_phase(p)

    # total framenum frequecy
    print('')
