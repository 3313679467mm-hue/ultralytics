"""
告警管理服务 - FastAPI后端
接收手机端告警，存储到SQLite，提供网页可视化

端口: 8001
"""

import base64
import os
import sqlite3
import time
from datetime import datetime
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="告警管理服务")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库和文件存储路径
DB_PATH = "alerts.db"
IMAGE_DIR = "alert_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================
# 数据模型
# ============================================================

class AlertRequest(BaseModel):
    timestamp: str
    image_base64: str
    detections: list
    consecutive_frames: int
    alert_type: str


class AlertResponse(BaseModel):
    id: int
    timestamp: str
    image_path: str
    detections: str
    consecutive_frames: int
    alert_type: str


# ============================================================
# 数据库操作
# ============================================================

def init_db():
    """初始化SQLite数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            detections TEXT,
            consecutive_frames INTEGER,
            alert_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_alert(data: AlertRequest) -> int:
    """保存告警记录"""
    # 保存图像
    image_data = base64.b64decode(data.image_base64)
    filename = f"alert_{int(time.time())}_{len(os.listdir(IMAGE_DIR))}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(image_data)

    # 保存到数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO alerts (timestamp, image_path, detections, consecutive_frames, alert_type) VALUES (?, ?, ?, ?, ?)",
        (data.timestamp, filepath, str(data.detections), data.consecutive_frames, data.alert_type),
    )
    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return alert_id


def get_alerts(limit: int = 20, offset: int = 0, date_filter: Optional[str] = None) -> tuple:
    """获取告警列表"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if date_filter:
        cursor.execute(
            "SELECT * FROM alerts WHERE timestamp LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (f"{date_filter}%", limit, offset),
        )
        alerts = [dict(row) for row in cursor.fetchall()]
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?", (f"{date_filter}%",))
    else:
        cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
        alerts = [dict(row) for row in cursor.fetchall()]
        cursor.execute("SELECT COUNT(*) FROM alerts")

    total = cursor.fetchone()[0]
    conn.close()
    return alerts, total


def get_alert_by_id(alert_id: int) -> Optional[dict]:
    """获取单条告警"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_alert(alert_id: int) -> bool:
    """删除告警记录及图片"""
    alert = get_alert_by_id(alert_id)
    if not alert:
        return False

    # 删除图片文件
    if os.path.exists(alert["image_path"]):
        os.remove(alert["image_path"])

    # 删除数据库记录
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()
    return True


def get_stats() -> dict:
    """获取统计信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("SELECT COUNT(*) FROM alerts")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM alerts WHERE timestamp LIKE ?", (f"{today}%",))
    today_count = cursor.fetchone()[0]

    # 近7天趋势
    cursor.execute("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM alerts
        WHERE timestamp >= date('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
    """)
    trend = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]

    conn.close()
    return {"total": total, "today": today_count, "trend": trend}


# ============================================================
# API 路由
# ============================================================

@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
async def root():
    """返回告警可视化页面"""
    return FileResponse("alerts_viewer.html")


@app.post("/api/alert")
async def receive_alert(data: AlertRequest):
    """接收手机端告警"""
    alert_id = save_alert(data)
    return {"status": "success", "alert_id": alert_id}


@app.get("/api/alerts")
async def list_alerts(limit: int = 20, offset: int = 0, date: Optional[str] = None):
    """获取告警列表"""
    alerts, total = get_alerts(limit, offset, date)
    return {"alerts": alerts, "total": total}


@app.get("/api/alerts/{alert_id}")
async def get_alert(alert_id: int):
    """获取单条告警"""
    alert = get_alert_by_id(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="告警记录不存在")
    return alert


@app.get("/api/alerts/image/{alert_id}")
async def get_alert_image(alert_id: int):
    """获取告警图片"""
    alert = get_alert_by_id(alert_id)
    if not alert or not os.path.exists(alert["image_path"]):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(alert["image_path"])


@app.delete("/api/alerts/{alert_id}")
async def remove_alert(alert_id: int):
    """删除告警"""
    if delete_alert(alert_id):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="告警记录不存在")


@app.get("/api/alerts/stats")
async def stats():
    """获取统计信息"""
    return get_stats()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
