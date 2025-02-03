import numpy as np
import cv2
import random

NUM_CLASSES = 80
DFL_REGRESSION_SPACE = 16
STRIDE = [8, 16, 32]
NUM_OUTPUTS = NUM_CLASSES + DFL_REGRESSION_SPACE * 4


class DetectorDriver():

    def __init__(self, accel, io_shape_dict, batch_size, stride, num_classes=80, dfl_regression_space=16):
        self.accel = accel
        self.io_shape_dict = io_shape_dict
        self.batch_size = batch_size
        self.stride = stride
        self.num_classes = num_classes
        self.dfl_regression_space = dfl_regression_space

        self.muls = [np.load("Mul_{}_param0".format(i)) for i in range(io_shape_dict['num_outputs'])]
        self.adds = [np.load("Add_{}_param0".format(i)) for i in range(io_shape_dict['num_outputs'])]
        self.outputs = [[np.zeros(1) for _ in range(io_shape_dict['num_outputs'])] for b in range(batch_size)]
        self.preproc_img_size = io_shape_dict['ishape_normal'][0][2]
        
        self.make_anchors()


    def make_anchors(self, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        output_shapes = self.io_shape_dict["oshape_normal"]
        for i, stride in enumerate(self.stride):
            _, h, w, _ = output_shapes[i]
            sx = np.arange(start=grid_cell_offset, stop=w, step=1)
            sy = np.arange(start=grid_cell_offset, stop=h, step=1)
            sx, sy = np.meshgrid(sx, sy)
            anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
            stride_tensor.append([stride] * (h*w))

        self.anchor_points = np.expand_dims(np.concatenate(anchor_points).transpose(1, 0), 0)
        self.strides_tensor = np.concatenate(stride_tensor)


    def yolov8_postproc(self, outs, batch_size, anchor_points, strides):

        dfl_integration_weights = np.arange(self.dfl_regression_space).reshape(1, -1, 1, 1)
        x_cat = np.concatenate([out.reshape(batch_size, NUM_OUTPUTS, -1) for out in outs], 2)
        boxes_classes = np.split(x_cat, [DFL_REGRESSION_SPACE * 4], axis=1)
        boxes, classes = boxes_classes
        classes = 1 / (1 + np.exp(-classes))

        # DFL expected value
        boxes = boxes.reshape(batch_size, 4, self.dfl_regression_space, -1).transpose(0, 2, 1, 3)
        exp_boxes = np.exp(boxes)
        boxes = exp_boxes / np.sum(exp_boxes, axis=1)
        boxes *= dfl_integration_weights
        boxes = np.sum(boxes, 1) 

        # decode bboxes
        left_top_right_bottom = np.split(boxes, 2, axis=1)
        lt = left_top_right_bottom[0]
        rb = left_top_right_bottom[1]
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        # center_xy = (x1y1 + x2y2) / 2
        # wh = x2y2 - x1y1
        # boxes = np.concatenate((center_xy, wh), 1) * strides
        boxes = np.concatenate((x1y1, x2y2), 1) * strides
        pred = np.concatenate((boxes, classes), 1)

        # nms
        pred = self.V8_non_max_suppression(pred)
        
        return pred


    def read_accel_and_postprocess(self):

        for o in range(self.io_shape_dict['num_outputs']):
            # np.save(outputfile[o], obuf)
            # self.accel.copy_output_data_from_device(self.accel.obuf_packed[o], ind=o)
            obuf_folded = self.accel.unpack_output(self.accel.obuf_packed[o], ind=o)
            obuf_normal = self.accel.unfold_output(obuf_folded, ind=o)
            out = obuf_normal.transpose(0, 3, 1, 2)
            out *= self.muls[o]
            out += self.adds[o]
            for in_batch_idx, single_output in enumerate(out):
                self.outputs[in_batch_idx][o] = single_output
        
        batch_detections = []
        for outs_idx, outs in enumerate(self.outputs):
            batch_detections.append(self.yolov8_postproc(outs, 1, self.anchor_points, self.strides_tensor)[0])

        return batch_detections


    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize and pad image while meeting stride-multiple constraints
        stride = max(self.stride)
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        img = img[:, :, ::-1]  # BGR to RGB, to 3x416x416
        
        return img


    def preproc_and_write_accel(self, batch):

        preprocessed_batch = []
        for i in range(self.batch_size):
            img = self.letterbox(batch[i], 320)
            preprocessed_batch.append(np.expand_dims(img, axis=0))
        ibuf_normal = np.concatenate(preprocessed_batch, axis=0)

        ibuf_folded = self.accel.fold_input(ibuf_normal)
        ibuf_packed = self.accel.pack_input(ibuf_folded)
        self.accel.copy_input_data_to_device(ibuf_packed)


    def V8_non_max_suppression(
        self,
        prediction,
        conf_thres=0.2,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=True,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        in_place=True,
        rotated=False,
    ):

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

        # import torchvision  # scope for faster 'import ultralytics'

        bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
        nc = nc or (prediction.shape[1] - 4)  # number of classes
        nm = prediction.shape[1] - nc - 4  # number of masks
        mi = 4 + nc  # mask start index
        # xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
        xc = np.max(prediction[:, 4:], axis=1) > conf_thres

        prediction = prediction.transpose(0, 2, 1)
        output = [np.zeros((0, 6))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, conf, cls)
            box_cls = np.split(x, [4], axis=1)
            box = box_cls[0]
            cls = box_cls[1]

            # if multi_label:
            #     i, j = torch.where(cls > conf_thres)
            #     x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            # else:  # best class only
            j = cls.argmax(1, keepdims=True)
            conf = np.take_along_axis(x[:, 4:], j, axis=1)
            x = np.concatenate((box, conf, j), 1)

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == classes).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            if n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            scores = x[:, 4]  # scores
            if rotated:
                boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
                i = nms_rotated(boxes, scores, iou_thres)
            else:
                boxes = x[:, :4] + c  # boxes (offset by class)
                i = nms(boxes, scores, iou_thres)
            i = i[:max_det]  # limit detections

            output[xi] = x[i]

        return output


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
    

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # print(img1_shape, img0_shape)
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