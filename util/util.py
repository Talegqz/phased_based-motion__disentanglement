from __future__ import print_function
import torch
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from PIL import Image
import math
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as functional
import os
from torch.autograd import Variable


def load_heatmap(hm_path):
    hm_array = np.load(hm_path)
    torch_heatmap = torch.transpose(torch.transpose(torch.from_numpy(hm_array), 1, 2), 0, 1)
    returned_mat = torch_heatmap[0:18, :, :]
    return returned_mat

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def hmp2pose_by_numpy(hmp_numpy):
    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = hmp_numpy[:, :, part]
        map = gaussian_filter(map_ori, sigma=5)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > 0.01))

        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        if len(peaks) > 0:
            max = 0
            for index, peak in enumerate(peaks):
                score = map_ori[peak[1], peak[0]]
                current_max_score = map_ori[peaks[max][1], peaks[max][0]]
                if score > current_max_score:
                    max = index
            peaks_with_score = [(peaks[max][0], peaks[max][1], map_ori[peaks[max][1], peaks[max][0]], peak_counter)]
            all_peaks.append(peaks_with_score)
            peak_counter += len(peaks_with_score)
        else:
            all_peaks.append([])
    return all_peaks


def hmp2pose(hmp_tensor):
    hmp_numpy = hmp_tensor[0].cpu().float().numpy()
    hmp_numpy = np.transpose(hmp_numpy, (1, 2, 0))
    return hmp2pose_by_numpy(hmp_numpy)


def hmp2im(heatmap_tensor):

    all_peaks = hmp2pose(heatmap_tensor)
    return pose2im_all(all_peaks)


def pose2im_all(all_peaks):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    image = pose2im(all_peaks, limbSeq, colors)
    return image


def pose2im_limb(all_peaks):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    image = pose2im(all_peaks, limbSeq, colors, _circle=False)
    return image


def pose2im_limb_filter(all_peaks, error, threshold):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for error_index, error_value in enumerate(error):
        if error_value > threshold:
            colors[error_index] = [0, 0, 0]
    image = pose2im(all_peaks, limbSeq, colors, _circle=False)
    return image


def pose2im(all_peaks, limbSeq, colors, _circle=True, _limb=True, imtype=np.uint8):
    canvas = np.zeros(shape=(256, 256, 3))
    canvas.fill(255)

    if _circle:
        for i in range(18):
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    if _limb:
        stickwidth = 4

        for i in range(len(limbSeq)):
            limb = limbSeq[i]
            cur_canvas = canvas.copy()
            point1_index = limb[0] - 1
            point2_index = limb[1] - 1

            if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
                point1 = all_peaks[point1_index][0][0:2]
                point2 = all_peaks[point2_index][0][0:2]
                X = [point1[1], point2[1]]
                Y = [point1[0], point2[0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                # cv2.line()
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas.astype(imtype)


def pose2limb(pose):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14]]
    limbs = []
    for seq_index, limb in enumerate(limbSeq):
        point1_index = limb[0] - 1
        point2_index = limb[1] - 1
        if len(pose[point1_index]) > 0 and len(pose[point2_index]) > 0:
            offset_x = pose[point2_index][0][0] - pose[point1_index][0][0]
            offset_y = pose[point2_index][0][1] - pose[point1_index][0][1]
            limbs.append([offset_x, offset_y])
        else:
            limbs.append([])
    return limbs


def distance_limb(limbs1, limbs2):
    assert len(limbs1) == len(limbs2)
    error_all = 0
    error_list = []
    count = 0
    for lamb_index in range(len(limbs1)):
        limb1 = limbs1[lamb_index]
        limb2 = limbs2[lamb_index]
        if len(limb1)>1 and len(limb2)>1:
            distance = (limb1[0] - limb2[0])**2 + (limb1[1] - limb2[1]) ** 2
            error_all += distance
            count += 1
        else:
            distance = None
        error_list.append(float(distance))
    for i, error in enumerate(error_list):
        if error is not None:
            error = math.sqrt(error)
        else:
            error = None
        error_list[i] = error
    error_list.append(math.sqrt(error_all/count))
    return np.array(error_list)


def distance_point(all_peaks, index1, index2):
    try:
        x1 = all_peaks[index1][0][1]
        y1 = all_peaks[index1][0][0]
        x2 = all_peaks[index2][0][1]
        y2 = all_peaks[index2][0][0]
    except IndexError:
        return 0
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def crop_head(original_tensor, heatmap_tensor, length):
    ear_offset = 10

    tensor_numpy = heatmap_tensor[0].cpu().float().numpy()
    tensor_numpy = np.transpose(tensor_numpy, (1, 2, 0))
    all_peaks = hmp2pose_by_numpy(tensor_numpy)
    center = [0, 0]
    count = 0
    for i in [0, 14, 15, 16, 17]:
        if len(all_peaks[i]) > 0:
            center[0] += all_peaks[i][0][1]
            center[1] += all_peaks[i][0][0]
            count += 1
    center[0] /= count
    center[1] /= count
    center[0] += (length/6)

    if length == None:
        a = distance_point(all_peaks, 0, 16) + ear_offset
        b = distance_point(all_peaks, 0, 17) + ear_offset
        c = distance_point(all_peaks, 1, 0)
        length = max(int(a), int(b), int(c))
    crop_regeion = crop_patch(original_tensor, center, length)
    return crop_regeion, center


def crop_patch(I, patch_center, patch_radius):
    [px, py] = [patch_center[0], patch_center[1]]
    r = patch_radius
    up_boundary = int(px - r) if px - r > 0 else 0
    down_boundary = int(px + r + 1) if px + r + 1 < I.size(2) else I.size(2)
    left_boundary = int(py - r) if py - r > 0 else 0
    right_boundary = int(py + r + 1) if py + r + 1 < I.size(3) else I.size(3)
    return I[:, :, up_boundary-1:down_boundary, left_boundary-1:right_boundary]


def paste_patch(I, patch, patch_center, patch_radius):
    [px, py] = [patch_center[0], patch_center[1]]
    r = patch_radius
    up_boundary = int(px - r) if px - r > 0 else 0
    down_boundary = int(px + r + 1) if px + r + 1 < I.size(2) else I.size(2)
    left_boundary = int(py - r) if py - r > 0 else 0
    right_boundary = int(py + r + 1) if py + r + 1 < I.size(3) else I.size(3)
    I[:, :, up_boundary+1:down_boundary+2, left_boundary-1:right_boundary] = patch[:, :, :, :]
    return I


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def get_height(poses):
    height = 0
    top = 1000
    bottom = 0
    for pose in poses:
        _top = 1000
        _bottom = 0
        for joint_index in [0, 14, 15, 16, 17]:
            if len(pose[joint_index]) > 0:
                if pose[joint_index][0][1] < _top:
                    _top = pose[joint_index][0][1]
        for joint_index in [10, 13]:
            if len(pose[joint_index]) > 0:
                if pose[joint_index][0][1] > _bottom:
                    _bottom = pose[joint_index][0][1]
                    if _bottom > bottom:
                        bottom = _bottom
        _height = _bottom - _top + 40
        if _height > height:
            height = _height
            top = _top
    return min(height, 255), max(0, top-20), min(bottom+20, 255)


def get_center_from_all(poses):
    center_x = 0
    center_y = 0
    count = 0
    for pose in poses:
        pose_center_x, pose_center_y = get_center(pose)
        center_x += pose_center_x
        center_y += pose_center_y
        count += 1
    center_x /= count
    center_y /= count
    return center_x, center_y


def get_bounding_box(pose):
    _top = 1000
    _bottom = 0
    for i in range(18):
        if len(pose[i]) > 0:
            value = pose[i][0][1]
            if value < _top:
                _top = value
            if value > _bottom:
                _bottom = value
    return max(_top-20, 0), min(_bottom+20, 255)


def get_center(pose):
    center_x = 0
    center_y = 0

    count = 0
    for i in range(18):
        if len(pose[i]) > 0:
            center_x += pose[i][0][1]
            count += 1
    if not count == 0:
        center_x /= count
    else:
        center_x = 0

    count = 0
    for i in [8, 11 , 2, 5]:
        if len(pose[i]) > 0:
            center_y += pose[i][0][0]
            count += 1
    if not count == 0:
        center_y /= count
    else:
        center_y = 0

    return center_x, center_y


def find_cloest_joint(joint_x, joint_y, joint_index, target_poses):
    min_index = 0
    min_distance = 10000
    for pose_index, pose in enumerate(target_poses):
        if len(pose[joint_index]) > 0:
            target_x = pose[joint_index][0][0]
            target_y = pose[joint_index][0][1]
            distance = (joint_x-target_x)**2 + (joint_y-target_y)**2
            if distance < min_distance:
                min_index = pose_index
                min_distance = distance
    return min_index


def find_cloest_limb(libm_index, errors):
    min_value = 1000
    min_index = 0
    for index, error in enumerate(errors):
        if error[libm_index] < min_value:
            min_index = index
            min_value = error[libm_index]
    return min_index


def offset_heatmap_channel(source_x, source_y, target_x, target_y, target_channel):
    offset_x = target_x - source_x
    offset_y = target_y - source_y
    target_channel_padding = np.pad(target_channel, ((abs(offset_y), abs(offset_y)), (abs(offset_x), abs(offset_x))), 'constant', constant_values=target_channel[0, 0])
    target_channel_crop = target_channel_padding[
                            abs(offset_y) + offset_y: abs(offset_y) + offset_y + target_channel.shape[1],
                            abs(offset_x) + offset_x: abs(offset_x) + offset_x + target_channel.shape[0]
                          ]
    return target_channel_crop


def replace_heatmaps(source_heatmaps, target_heatmaps):
    source_heatmaps_clone = np.copy(source_heatmaps)
    refer_map = np.zeros(shape=(len(source_heatmaps), 18))
    print('Generating the pose from dataset...')
    source_poses = [hmp2pose_by_numpy(heatmap) for heatmap in source_heatmaps]
    target_poses = [hmp2pose_by_numpy(heatmap) for heatmap in target_heatmaps]

    print('Converted the heatmaps to poses!')
    for pose_index, pose in enumerate(source_poses):
        for joint_index, joint in enumerate(pose):
            if len(joint) > 0:
                source_x = joint[0][0]
                source_y = joint[0][1]
                cloest_index = find_cloest_joint(source_x, source_y, joint_index, target_poses)
                target_x = target_poses[cloest_index][joint_index][0][0]
                target_y = target_poses[cloest_index][joint_index][0][1]
                target_channel = target_heatmaps[cloest_index][:, :, joint_index]
                source_heatmaps_clone[pose_index][:, :, joint_index] = offset_heatmap_channel(source_x, source_y, target_x, target_y, target_channel)
                refer_map[pose_index, joint_index] = cloest_index
                print('Replaced pose %d ...' % pose_index)
            else:
                refer_map[pose_index, joint_index] = -1
    return source_heatmaps_clone, refer_map


def translate_image(image, time, offset_x, offset_y):
    offset_x *= time
    offset_y *= time
    image_resize = cv2.resize(image, dsize=(int(image.shape[1] * time), int(image.shape[0] * time)))
    background = np.zeros(shape=(500, 500, 3))
    background.fill(255)
    left = (500 - image_resize.shape[1])/2
    top = (500 - image_resize.shape[0])/2
    left = int(left + offset_x)
    top = int(top + offset_y)
    background[left:(left+image_resize.shape[1]), top:(top+image_resize.shape[0]), :] = image_resize
    background = background[122:378, 122:378, :]
    return background


def offset_image(image, heatmap, padding=20):

    pose = hmp2pose_by_numpy(heatmap)
    source_top, source_bottom = get_bounding_box(pose)
    source_center_x, source_center_y = get_center(pose)
    source_center_x = source_center_x - source_top
    image_crop = image[source_top: source_bottom, :, :]
    # image_crop = cv2.circle(image_crop, center=(int(source_center_y), int(source_center_x)), radius=2, color=(122,122,0))
    # cv2.imwrite('test.png', image_crop)

    if source_bottom > 256 - padding:
        time = (256 - padding - source_top) / image_crop.shape[0]
        image_crop = cv2.resize(image_crop, dsize=(int(image_crop.shape[1] * time), (256 - padding - source_top)))
        source_center_x = source_center_x * time
        source_center_y = source_center_y * time
        # image_crop = cv2.circle(image_crop, center=(int(source_center_x), int(source_center_y)), radius=2, color=(122, 122, 0))
        # cv2.imwrite('test_resize.png', image_crop)

    if source_top < padding:
        time = (source_top + image_crop.shape[0] - padding) / (image_crop.shape[0])
        image_crop = cv2.resize(image_crop,
                                dsize=(int(image_crop.shape[1] * time), (source_top + image_crop.shape[0] - padding)))
        source_center_x = int(source_center_x * time)
        source_center_y = int(source_center_y * time)
        # cv2.imwrite('test_resize1.png', image_crop)
    # image_crop = cv2.circle(image_crop, center=(int(source_center_x), int(source_center_y)), radius=2, color=(122, 122, 0))

    background = np.zeros(shape=(500, 500, 3))
    target_left = int(500 / 2 - (source_center_y))
    target_top = int(500 / 2 - 20 - (source_center_x))
    background[target_top: target_top + image_crop.shape[0], target_left:target_left + image_crop.shape[1], :] = image_crop
    background = background[122:378, 122:378, :]
    return background


def channel2image(channel):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(channel)
    _max = np.max(channel)
    _min = np.min(channel)
    for x in range(channel.shape[0]):
        for y in range(channel.shape[1]):
            rgba_img[x][y][3] = 0.2 + 0.799*(channel[x][y] - _min) / (_max - _min)
    return rgba_img


def heatmap2array(heatmaps, size=256):
    arrays = np.zeros(shape=(size, size))
    arrays.fill(heatmaps.max())
    for channel in [18]:
        x_top = 0
        y_top = 0
        for x in range(heatmaps.shape[0]):
            for y in range(heatmaps.shape[1]):
                _x = int(x_top+x)
                _y = int(y_top+y)
                arrays[_x][_y] = heatmaps[x][y][channel]
    return arrays


def array2image(arrays):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(1-arrays)
    _max = np.max(arrays)
    _min = np.min(arrays)
    # for x in range(arrays.shape[0]):
    #     for y in range(arrays.shape[1]):
    #         rgba_img[x][y][3] = 0.1 + 0.899*(arrays[x][y] - _min) / (_max - _min)
    return rgba_img


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def error_graph(path, start=0, end=1500, multi=1):
    item_num = multi+2
    files = os.listdir(path)
    count = int(len(files)/item_num)
    x = list(range(count))[start:end]
    y = []
    for i in range(count)[start:end]:
        real_path = '%s/%d_real_B.png' % (path, i)
        fake_path = '%s/%d_fake_B.png' % (path, i)
        real = cv2.imread(real_path)
        fake = cv2.imread(fake_path)
        error = np.power(np.sum(np.square((real-fake))) / (256*256*3), 0.5)
        print('%d: %s' % (i, error))
        y.append(error)
    plt.figure()
    plt.plot(x, y, 'b')
    plt.savefig('test_all_D.png')


def make_dataset(dir_list, phase):
    images = []
    for dataroot in dir_list:
        _images = []
        assert os.path.isdir(dataroot), '%s is not a valid directory' % dataroot
        for root, _, fnames in sorted(os.walk(dataroot)):
            for fname in fnames:
                if phase in fname:
                    path = os.path.join(root, fname)
                    _images.append(path)
        _images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        images +=_images
    return images


def make_test_dataset(dir_list, phase):
    images = []
    for dataroot in dir_list:
        _images = []
        assert os.path.isdir(dataroot), '%s is not a valid directory' % dataroot
        for root, _, fnames in sorted(os.walk(dataroot)):
            for fname in fnames:
                if phase in fname:
                    path = os.path.join(root, fname)
                    _images.append(path)
        _images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))
        images +=_images
    return images


def transfer_feature_mean_and_stdv(F, new_mean, new_stdv):
    F_mean = mean_channels(F)
    F_stdv = stdv_channels(F)
    F_normalized = (F - F_mean) / F_stdv
    return new_stdv * F_normalized + new_mean


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum.expand_as(F) / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3)) + 10**(-10)
    return F_variance.expand_as(F).pow(0.5)


def convert_m1to1_to_vgg(self, A):
    A_0to1 = 0.5 * (A + 1)
    mean = Variable(self.Tensor([[[[0.485]], [[0.456]], [[0.406]]]]).expand_as(A_0to1))
    stdv = Variable(self.Tensor([[[[0.229]], [[0.224]], [[0.225]]]]).expand_as(A_0to1))
    return (A_0to1 - mean) / stdv


def norm_per_channels_m1to1(A, isVar=True):
    if isVar == True:
        A_relu = functional.relu(A)
    else:
        A_relu = functional.relu(Variable(A)).data

    A_res = A_relu / A_relu.max(3, keepdim=True)[0].max(2, keepdim=True)[0].expand_as(A_relu)
    return 2 * A_res - 1

# error_graph('/mnt/results/experiment10_pix2pix(withD_withoutD)/single_paired_D_loss_with_5_batchsize=16_lambda=100/test_latest/images')
