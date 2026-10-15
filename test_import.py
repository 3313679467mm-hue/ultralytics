import inspect

from ultralytics import YOLO

print(f"YOLO 类位置: {YOLO.__module__}")
print(f"YOLO 文件路径: {inspect.getfile(YOLO)}")
