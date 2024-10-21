import torch
import numpy as np

import sys
import os
import cv2
import time

sys.path.append("/scratch/user/s4702415/GeneSegNet/GeneSegNet")
from dynamics import gen_pose_target


def gen_cells(d: int, n: int, r: int, colour: tuple, centre, variance):
    """
    Parameters:
        d: image size
        n: number of cells
        r: radius of cells (px)

    Returns:
        Cell channel
    """
    image = np.zeros(shape=(d, d, 3), dtype=np.float32)

    # centre = [(np.random.randint(low=0, high=d), np.random.randint(
    #     low=0, high=d)) for _ in range(n)]

    for i in range(n):
        image = cv2.circle(image, centre[i], r, colour, -1)

    image = image[:, :, 0]

    s = []
    for row, i in enumerate(image):
        for col, j in enumerate(i):
            if j:
                s.append((row, col))

    for j in range(d):
        for i in range(d):
            if image[j][i]:
                image[j][i] = 1 + -abs(np.random.normal(loc=0.0, scale=variance))
            else:
                image[j][i] = abs(np.random.normal(loc=0.0, scale=variance))

    return torch.tensor(image), centre, s

# def get_ground_truth(cells):

#     s = []
#     for row, i in enumerate(cells):
#         for col, j in enumerate(i):
#             if j:
#                 s.append((row, col))

#     return s

def gen_heatmap(centre, d, variance=None):

    centre_array = np.array([[x, y] for x, y in centre])
    gm = gen_pose_target(centre_array, 'cpu', h=d, w=d)

    # if variance:
    #     for j in range(d):
    #         for i in range(d):
    #             if gm[j][i]:
    #                 gm[j][i] = np.random.normal(loc=gm[j][i], scale=variance)
    #             else:
    #                 gm[j][i] = np.random.normal(loc=0.0, scale=variance)

    return gm

def display_confusion(gt, mask, d):
    print(f"gt = {gt}")
    pt = [(i, j) for i in range(d) for j in range(d) if mask[i][j]]
    print(f"pt = {pt}")
    tp = [value for value in pt if value in gt]
    print(f"tp = {tp}")
    fp = [value for value in pt if value not in tp]
    print(f"fp = {fp}")
    fn = [value for value in gt if value not in pt]
    print(f"fn = {fn}")
    tn = [(i, j) for i in range(d) for j in range(d) if (i, j) not in tp and (i, j) not in fp and (i, j) not in fn]
    print(f"tn = {tn}")

    print(f"tplen = {len(tp)}")
    print(f"fplen = {len(fp)}")
    print(f"fnlen = {len(fn)}")
    print(f"tnlen = {len(tn)}")

    return tp, fp, fn, tn

def calculate_mean_intensity(tp, fp, fn, tn, image):

    object_int = 0
    bg_int = 0

    for i, j in tp:
        object_int += image[i][j]
    for i, j in fp:
        object_int += image[i][j]
    for i, j in fn:
        bg_int += image[i][j]
    for i, j in tn:
        bg_int += image[i][j]
        # object_mean = (sum(np.random.normal(loc=1, scale=variance, size=tp)) + sum(np.random.normal(loc=0, scale=variance, size=fp)))/(tp + fp)
        # background_mean = (sum(np.random.normal(loc=1, scale=variance, size=fn)) + sum(np.random.normal(loc=0, scale=variance, size=tn)))/(fn + tn)
    object_mean = object_int / (len(tp) + len(fp))
    bg_mean = bg_int / (len(fn) + len(tn))

    return object_mean, bg_mean


# def 

#     image = torch.stack([image1, image2])
#     # try without noise?
#     image = image + torch.tensor(np.random.uniform(low=0.0, high=0.0001, size=(2, d, d)), dtype=torch.float)

#     images.append(image)
#     S.append(s)

# dataset = torch.stack(images)

# path_56 = "/scratch/user/s4702415/Honours/models/test_genesegnet/genesegnet_n56/GeneSegNet_hippocampus_residual_on_style_on_concatenation_off.929131_epoch_499.onnx"
# model_56 = onnx.load(path_56)

# models = {'56': model_56}
# ort_sess = ort.InferenceSession(path_56)
# # ort_sess = ort.InferenceSession(path_56)

# # images need to be scaled
# for img in range(n_images):
#     si_unet = SI4ONNX(model_56, thr=0.9)

#     # image = convert_image(dataset[img].unsqueeze(0).numpy(), [0, 1])

#     # image = torch.tensor(image)
#     # image = image.permute(2, 0, 1)
#     # image = image.unsqueeze(0)
#     # image = image.numpy()

#     image = dataset[img].unsqueeze(0).numpy()

#     # try:
#     outputs = ort_sess.run(None, {'input': dataset[img].unsqueeze(0).numpy()})
#     # outputs = ort_sess.run(None, {'input': image})
#     # out = outputs[0]
#     # prob = (out[0, 2, :, :] - out[0, 2, :, :].min())/(out[0, 2, :, :].max() - out[0, 2, :, :].min())
#     # mask = prob > 0.8

#     # not sure whether to use this compute masks function or not



#     img_path = "/scratch/user/s4702415/Honours/models/test_genesegnet/power_images_intensity"

#     # plt.imshow(image[0, 0, :, :])
#     # plt.savefig(os.path.join(img_path, "real_image_noise.png"))

#     # plt.imshow(image[0, 1, :, :])
#     # plt.savefig(os.path.join(img_path, "real_gm_noise.png"))

#     mask_path = os.path.join(img_path, f"mask.png")
#     mask, p = compute_masks(outputs[0][0, :2, :, :], outputs[0][0, 2, :, :], confidence_threshold=0.5)


#     # p0_path = os.path.join(img_path, "p0.png")
#     # p1_path = os.path.join(img_path, "p1.png")
#     # mask_path = os.path.join(img_path, f"mask.png")

#     # plt.imshow(p[0, :, :])
#     # plt.savefig(p0_path)
#     # plt.imshow(p[1, :, :])
#     # plt.savefig(p1_path)
#     plt.imshow(mask)
#     plt.savefig(mask_path)

#     # for c in np.arange(0, 1, 0.1):
#     #     for f in np.arange(0, 1, 0.1):
#     #         mask_path = os.path.join(img_path, f"mask_conf_{c}_flow_{f}.png")
#     #         mask, p = compute_masks(outputs[0][0, :2, :, :], outputs[0][0, 2, :, :], confidence_threshold=c, flow_threshold=f)
#     #         plt.imshow(mask)
#     #         plt.savefig(mask_path)


#     # mask, p = compute_masks(outputs[0][0, :2, :, :], outputs[0][0, 2, :, :], confidence_threshold=0.9, flow_threshold=0.5)
#     # postp = postprocess(mask, 1)




#     # plt.imshow(mask)
#     # plt.savefig("/scratch/user/s4702415/Honours/models/test_genesegnet/power_images/mask_synthetic_0.8.png")

#     # outputs = ort_sess.run(None, {'input': testimage.unsqueeze(0).numpy()})
#     # out = outputs[0]
#     # prob = (out[0, 2, :, :] - out[0, 2, :, :].min())/(out[0, 2, :, :].max() - out[0, 2, :, :].min())
#     # # prob = out[0, 2, :, :]
#     # mask = prob > 0.75
#     # plt.imshow(mask)
#     # plt.savefig("/scratch/user/s4702415/Honours/models/test_genesegnet/power_images/mask_real_0.75.png")

#     # decide how many are true and how many are false
#     # gt = S[img]
#     # print(f"gt = {gt}")
#     # pt = [(i, j) for i in range(d) for j in range(d) if mask[i][j]]
#     # print(f"pt = {pt}")
#     # tp = [value for value in pt if value in gt]
#     # print(f"tp = {tp}")
#     # fp = [value for value in pt if value not in tp]
#     # print(f"fp = {fp}")
#     # fn = [value for value in gt if value not in pt]
#     # print(f"fn = {fn}")
#     # tn = [(i, j) for i in range(d) for j in range(d) if (i, j) not in tp and (i, j) not in fp and (i, j) not in fn]
#     # print(f"tn = {tn}")

#     # print(f"tplen = {len(tp)}")
#     # print(f"fplen = {len(fp)}")
#     # print(f"fnlen = {len(fn)}")
#     # print(f"tnlen = {len(tn)}")

#     # start = time.time()
#     p_value = si_unet.inference(dataset[img].unsqueeze(0), var=1.0, termination_criterion='decision', over_conditioning=True)
#     print(f"p_value = {p_value}")
#     # print(f"Time = {time.time() - start}")