 deep_sort.yaml：这个yaml文件主要是保存一些参数。

（1）REID_CKPT：里面有特征提取权重的目录路径；

（2）MAX_DIST：最大余弦距离，用于级联匹配，如果大于该阈值，则忽略。

（3）MIN_CONFIDENCE：检测结果置信度阈值

（4）NMS_MAX_OVERLAP：非极大抑制阈值，设置为1代表不进行抑制

（5）MAX_IOU_DISTANCE：最大IOU阈值

（6）MAX_AGE：最大寿命，也就是经过MAX_AGE帧没有追踪到该物体，就将该轨迹变为删除态。

（7）N_INIT：最高击中次数，如果击中该次数，就由不确定态转为确定态。

（8）NN_BUDGET：最大保存特征帧数，如果超过该帧数，将进行滚动保存。