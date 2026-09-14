import os

from ultralytics import YOLO

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    weights_dir = os.path.join(script_dir, "..", "..", "weights")
    model_path = os.path.join(weights_dir, "best_modified.pt")
    data_path = os.path.join(script_dir, "..", "..", "datasets", "phone_detection", "data.yaml")

    model = YOLO(model_path)
    results = model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        device=0,
        batch=4,
        name="phone_detection5",
        hsv_h=0.025,
        hsv_s=0.8,
        hsv_v=0.5,
        fliplr=0.7,
        copy_paste=0.5,
        patience=100,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.3,
        dfl=1.5,
        dropout=0.0,
        resume=False,
    )
    print("训练完成!")
