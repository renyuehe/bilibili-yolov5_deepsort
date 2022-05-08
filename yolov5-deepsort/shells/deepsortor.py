from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

class Deepsortor:
    def __init__(self, configFile):
        cfg = get_config()
        cfg.merge_from_file(configFile)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    def update(self, xywhs, confss, image):
        bboxes2draw = []
        # Pass detections to deepsort
        outputs = self.deepsort.update(xywhs, confss, image)

        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            bboxes2draw.append(
                (x1, y1, x2, y2, '', track_id)
            )

        return image, bboxes2draw