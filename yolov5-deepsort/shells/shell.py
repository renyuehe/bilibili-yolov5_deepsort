import torch

from shells.deepsortor import Deepsortor
from shells.detector import Detector

class Shell(object):
    def __init__(self, deepsort_config_path, yolo_weight_path):
        self.deepsortor = Deepsortor(configFile=deepsort_config_path)
        self.detector = Detector(yolo_weight_path, imgSize=640, threshould=0.3, stride=1)
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