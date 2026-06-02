import cv2

from ultralytics import YOLO


class PhoneDetector:
    def __init__(self, model_path="runs/detect/phone_detection4/weights/best.pt"):
        """初始化手机检测器 1 参数: model_path: 训练好的模型路径."""
        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.classes = self.model.names
        print(f"模型加载完成, 类别: {self.classes}")

    def detect_image(self, image_path, conf_threshold=0.25, iou_threshold=0.45, save_path=None):
        """检测单张图片中的手机.

        参数:
            image_path: 输入图片路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            save_path: 保存结果的路径
        """
        print(f"正在检测图片: {image_path}")

        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            show=False,
            save=True,
            save_txt=False,
            save_conf=True,
        )

        result = results[0]

        print("检测完成!")
        print(f"找到 {len(result.boxes)} 个目标")

        if save_path:
            result.save(filename=save_path)
            print(f"结果已保存至: {save_path}")

        return result

    def detect_video(self, video_path, output_path=None, conf_threshold=0.25):
        """检测视频中的手机.

        参数:
            video_path: 输入视频路径
            output_path: 输出视频路径
            conf_threshold: 置信度阈值
        """
        print(f"正在检测视频: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        if output_path:
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"处理帧: {frame_count}")

            results = self.model.predict(source=frame, conf=conf_threshold, show=False, verbose=False)

            result = results[0]

            annotated_frame = result.plot()

            if out:
                out.write(annotated_frame)

        cap.release()
        if out:
            out.release()

        print(f"视频处理完成! 总帧数: {frame_count}")
        if output_path:
            print(f"输出视频保存至: {output_path}")

    def detect_webcam(self, conf_threshold=0.25):
        """使用摄像头实时检测手机.

        参数:
            conf_threshold: 置信度阈值
        """
        print("正在启动摄像头...")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("无法打开摄像头")

        print("按 q 键退出")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(source=frame, conf=conf_threshold, show=False, verbose=False)

            result = results[0]

            annotated_frame = result.plot()

            cv2.imshow("Phone Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_detections(self, image_path, conf_threshold=0.25):
        """获取检测结果（不绘制框）.

        参数:
            image_path: 输入图片路径
            conf_threshold: 置信度阈值

        返回:
            检测结果列表，每个元素包含: class, confidence, bounding_box
        """
        results = self.model.predict(source=image_path, conf=conf_threshold, show=False)

        result = results[0]
        detections = []

        for box in result.boxes:
            detection = {
                "class": self.classes[int(box.cls)],
                "confidence": float(box.conf),
                "bounding_box": box.xyxy[0].tolist(),
            }
            detections.append(detection)

        return detections


def main():
    detector = PhoneDetector()

    while True:
        print("\n=== 手机检测系统 ===")
        print("1. 检测单张图片")
        print("2. 检测视频")
        print("3. 摄像头实时检测")
        print("4. 退出")

        choice = input("请选择功能 (1-4): ").strip()

        if choice == "1":
            image_path = input("请输入图片路径: ").strip()
            if image_path:
                output_path = input("请输入输出路径 (可选, 直接回车使用默认路径): ").strip() or None
                conf = float(input("请输入置信度阈值 (默认0.25): ").strip() or "0.25")
                detector.detect_image(image_path, save_path=output_path, conf_threshold=conf)

        elif choice == "2":
            video_path = input("请输入视频路径: ").strip()
            if video_path:
                output_path = input("请输入输出路径 (可选, 直接回车使用默认路径): ").strip() or None
                conf = float(input("请输入置信度阈值 (默认0.25): ").strip() or "0.25")
                detector.detect_video(video_path, output_path=output_path, conf_threshold=conf)

        elif choice == "3":
            conf = float(input("请输入置信度阈值 (默认0.25): ").strip() or "0.25")
            detector.detect_webcam(conf_threshold=conf)

        elif choice == "4":
            print("退出程序")
            break

        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()
