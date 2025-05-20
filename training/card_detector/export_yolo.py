from ultralytics import YOLO

YOLO("card_detector.pt").export(format="coreml", int8=True, nms=True)
