from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import torch
import cv2
import numpy as np
import os
import base64
import tempfile
from pathlib import Path

# 导入 Ultralytics YOLO
from ultralytics import YOLO

# 初始化 FastAPI 应用
app = FastAPI(title="YOLOv8 手机检测 API", description="基于 YOLOv8 的手机检测服务")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    """根路径 - 返回前端界面"""
    return FileResponse("ai_studio_code.html")

# 模型配置
MODEL_PATH = "weights/best_modified.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 加载模型
print(f"正在加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("模型加载完成!")

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bounding_box: List[float]

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "YOLOv8 手机检测 API",
        "version": "1.0.0",
        "endpoints": {
            "/detect/image": "POST - 图片检测",
            "/detect/video": "POST - 视频检测",
            "/detect/frame": "POST - 帧检测"
        }
    }

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """单图检测"""
    try:
        # 读取上传的图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解码图片")
        
        # 进行检测
        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            show=False,
            save=False
        )
        
        # 绘制检测结果
        result = results[0]
        annotated_frame = result.plot()
        
        # 编码为 JPEG
        _, encoded_img = cv2.imencode('.jpg', annotated_frame)
        
        # 将图像转换为 base64
        base64_img = base64.b64encode(encoded_img).decode('utf-8')
        
        return {"image": f"data:image/jpeg;base64,{base64_img}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")
    finally:
        # 清理临时文件
        if 'temp_path' in locals():
            os.unlink(temp_path)

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """视频检测"""
    temp_input_path = None
    temp_output_path = None
    
    try:
        # 读取上传的视频
        contents = await file.read()
        
        # 保存临时视频文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            temp_input_path = temp_input.name
            temp_input.write(contents)
        
        # 创建输出临时文件
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        # 读取视频
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {temp_input_path}")
            raise HTTPException(status_code=400, detail="无法打开视频文件")
        
        # 获取视频属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频信息: {frame_width}x{frame_height}, FPS: {fps}")
        
        if fps == 0:
            fps = 30.0
        
        # 创建视频写入器 - 使用 H.264 编码
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            # 如果 H.264 不可用，尝试 mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                cap.release()
                raise HTTPException(status_code=500, detail="无法创建视频写入器")
        
        # 处理每一帧
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 进行检测
            results = model.predict(
                source=frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                show=False,
                save=False
            )
            
            # 绘制检测结果
            result = results[0]
            annotated_frame = result.plot()
            
            # 写入帧
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        # 检查输出文件是否存在且不为空
        if not os.path.exists(temp_output_path):
            print(f"错误: 输出文件不存在 {temp_output_path}")
            raise HTTPException(status_code=500, detail="视频处理失败，输出文件不存在")
        
        if os.path.getsize(temp_output_path) == 0:
            print(f"错误: 输出文件为空 {temp_output_path}")
            raise HTTPException(status_code=500, detail="视频处理失败，输出文件为空")
        
        print(f"视频处理完成，文件大小: {os.path.getsize(temp_output_path)} 字节")
        
        # 读取处理后的视频并转换为 base64
        with open(temp_output_path, 'rb') as video_file:
            video_data = video_file.read()
            base64_video = base64.b64encode(video_data).decode('utf-8')
        
        return {"video": f"data:video/mp4;base64,{base64_video}"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"视频处理异常: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"视频处理失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
            except:
                pass
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.unlink(temp_output_path)
            except:
                pass

@app.post("/detect/frame")
async def detect_frame(file: UploadFile = File(...)):
    """帧检测（用于实时监控）"""
    try:
        # 读取上传的帧
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解码帧")
        
        # 进行检测
        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            show=False,
            save=False
        )
        
        # 绘制检测结果
        result = results[0]
        annotated_frame = result.plot()
        
        # 编码为 JPEG
        _, encoded_img = cv2.imencode('.jpg', annotated_frame)
        
        # 将图像转换为 base64
        base64_img = base64.b64encode(encoded_img).decode('utf-8')
        
        # 返回处理后的帧
        return {"image": f"data:image/jpeg;base64,{base64_img}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"帧检测失败: {str(e)}")

if __name__ == "__main__":
    print("启动服务器...")
    print("访问 http://127.0.0.1:8000/docs 查看 API 文档")
    print("访问 http://127.0.0.1:8000 查看前端界面")
    
    # 启动服务器
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )
