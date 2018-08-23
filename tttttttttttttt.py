from PIL import Image
import numpy as np

def get_mask(size,w,d):

    rowcenter = int(size[0] / 2)
    columncenter = int(size[0] / 2)


    mask1 = np.ones(size, np.uint8)

    mask1[rowcenter-d:rowcenter+d,columncenter-d:columncenter+d] = 0

    mask2 = np.zeros(size, np.uint8)

    mask2[rowcenter - d-w:rowcenter + d+w, columncenter - d-w:columncenter + d+w] = 1

    mask = mask1 * mask2

    return mask


pic1 = Image.open('result/frame0and3/im1_gray.png')


# pic1.show()
pic = np.array(pic1)
# print(pic.shape)
# print(pic)
# pic = np.array([[1,1],[1,7],[1,6],[1,1]])
# mask = get_mask(pic.shape, 4, 3)
# pic = np.arange(0,256*256).reshape(256,256)
phase_f = np.fft.fft2(pic)
print(phase_f)
# phase_fs = np.fft.fftshift(phase_f)
# phase_fs = phase_fs * mask
# new_phase = np.fft.ifftshift(phase_fs)
new_phase = np.fft.ifft2(phase_f)
print(new_phase)


# # print(pic)
pic = pic.astype(np.uint8)
pic = Image.fromarray(pic,'L')
#
pic.show()
#
# print(np.array(pic1)-np.array(pic))