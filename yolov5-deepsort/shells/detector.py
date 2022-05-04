import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device

from shells.deepsortor import Deepsortor

OBJ_LIST = ['person', 'car', 'bus', 'truck']

class Shell(object):
    def __init__(self, deepsort_config_path, yolo_weight_path):
        self.deepsortor = Deepsortor(configFile=deepsort_config_path)
        self.detector = Detector(yolo_weight_path, imgSize=640, threshould=0.3, stride=1)
        self.build_config()

    def build_config(self):
        self.frameCounter = 0

    def update(self, im):
        retDict = {
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }

        self.frameCounter += 1

        _, bboxes = self.detector.detect(im)
        bbox_xywh = []
        confs = []
        bboxes2draw = []
        if len(bboxes):
            # Adapt detections to deep sort input format
            for x1, y1, x2, y2, _, conf in bboxes:
                obj = [
                    int((x1 + x2) / 2), int((y1 + y2) / 2),
                    x2 - x1, y2 - y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)
            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            im, obj_bboxes = self.deepsortor.update(xywhs, confss, im, bboxes2draw)


            retDict['frame'] = im
            retDict['obj_bboxes'] = obj_bboxes

        return retDict

class Detector(object):
    def __init__(self, weight_path, imgSize=640, threshould=0.3, stride=1):
        super(Detector, self).__init__()
        self.init_model(weight_path)
        self.img_size = imgSize
        self.threshold = threshould
        self.stride = stride

    def init_model(self, weight_path):
        self.weights = weight_path
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detect(self, im):
        im0, img = self.preprocess(im)
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in OBJ_LIST:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
        return im, pred_boxes

