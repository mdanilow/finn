
import numpy as np
import time
import cv2


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    def nms(boxes, scores, overlap_threshold=0.5, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        index_array = scores.argsort()[::-1]
        keep = []
        while index_array.size > 0:
            keep.append(index_array[0])
            x1_ = np.maximum(x1[index_array[0]], x1[index_array[1:]])
            y1_ = np.maximum(y1[index_array[0]], y1[index_array[1:]])
            x2_ = np.minimum(x2[index_array[0]], x2[index_array[1:]])
            y2_ = np.minimum(y2[index_array[0]], y2[index_array[1:]])

            w = np.maximum(0.0, x2_ - x1_ + 1)
            h = np.maximum(0.0, y2_ - y1_ + 1)
            inter = w * h

            if min_mode:
                overlap = inter / np.minimum(areas[index_array[0]], areas[index_array[1:]])
            else:
                overlap = inter / (areas[index_array[0]] + areas[index_array[1:]] - inter)

            inds = np.where(overlap <= overlap_threshold)[0]
            index_array = index_array[inds + 1]
        return keep

    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y


    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # print(prediction[..., 4])

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # print('dupa')
        # print(x.shape)

        # If none remain process next image
        if not x.shape[0]:
            continue
        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Take conf of the best class
        j = x[:, 5:].argmax(axis=1, keepdims=True)
        conf = np.take_along_axis(x[:, 5:], j, axis=1)
        x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        # print(len(i), i)
        if len(i) > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def make_grid(sizes):

    grid = []
    for (nx, ny) in sizes:
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        grid.append(np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)))
    return grid


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def postprocess(obuf_normal, muls, adds, img1_shape, img0_shape, classes=None):
    if not isinstance(obuf_normal, list):
        obuf_normal = [obuf_normal]

    preds = []
    for i, x in enumerate(obuf_normal):
        x = x.transpose(0, 3, 1, 2)
        x *= muls[i]
        x += adds[i]
        bs, _, ny, nx = x.shape
        x = x.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
        x = 1/(1 + np.exp(-x))
        x[..., 0:2] = (x[..., 0:2] * 2. - 0.5 + grid[i]) * strides[i]
        x[..., 2:4] = (x[..., 2:4] * 2) ** 2 * anchor_grid[i]
        preds.append(x.reshape(bs, -1, no))

    preds = np.concatenate(preds, 1)
    preds = non_max_suppression(preds, conf_thres, iou_thres, classes=classes)[0]
    preds[:, :4] = scale_coords(img1_shape, preds[:, :4], img0_shape).round()

    return preds


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    


nl = 5
# muls = [np.load("Mul_{}.npy".format(i)) for i in range(nl)]
# adds = [np.load("Add_{}.npy".format(i)) for i in range(nl)]
na = 3
no = 85
sizes = [(20, 20), (10, 10), (5, 5), (3, 3), (2, 2)]
grid = make_grid(sizes)
strides = [16, 32, 64, 128, 256]
anchors = [
    [4.87787,   6.35631, 12.91158,   9.03208, 7.15285,  18.29766],
    [15.59967,  22.33371, 35.73561,  19.68820, 21.02688,  47.27071],
    [40.85436,  39.83278, 38.59727,  78.41331, 81.68491,  46.31688],
    [69.72626,  85.14220, 70.14906, 160.70825, 124.91834, 114.23557],
    [206.45392,  86.47435, 170.41608, 228.14876, 262.37347, 175.54692]
]
anchor_grid = np.array(anchors).reshape(nl, 1, -1, 1, 1, 2)
conf_thres = 0.2
iou_thres = 0.45
