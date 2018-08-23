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

