from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.Object_Detection import ObjectDetection
import yaml  
import cv2
import numpy as np
import base64
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('config/obb_det.yaml', 'r',encoding="utf-8") as f:
    config = yaml.safe_load(f)

model = None

class ModelConfig(BaseModel):
    platform: str
    precision: str

@app.post("/initialize_model/")
async def initialize_model(model_config: ModelConfig):
    global model, config
    
    config["platform"] = model_config.platform
    config["precision"] = model_config.precision
    config["model_path"] = r"C:\Users\87758\vue3_fastAPI\backend\deploy_model\obb_model\best_ckpt.pt"
    model = ObjectDetection(config)
    return {"message": "模型初始化成功"}
   

@app.post("/upload_image/")
async def upload_image(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        _, buffer = cv2.imencode('.png', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return {"image": f"data:image/png;base64,{img_str}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_image/")
async def process_image(
    image: str = Body(...),
    conf_thres: float = Body(...),
    iou_thres: float = Body(...)
):
    global model, config
    
    if model is None:
        raise HTTPException(status_code=400, detail="模型未初始化")
    
    try:
        # 解码base64图像
        image = base64.b64decode(image.split(',')[1])
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 更新推理参数
        inference_config = config.copy()
        inference_config["conf_thres"] = conf_thres
        inference_config["iou_thres"] = iou_thres
        
        # 运行推理
        result_img = model.run(img, conf_thres, iou_thres)
        
        # 将结果图像编码为base64字符串
        _, buffer = cv2.imencode('.png', result_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "processed_image": f"data:image/png;base64,{img_str}",
            "inference_result": {
                "platform": config["platform"],
                "precision": config["precision"],
                "conf_thres": conf_thres,
                "iou_thres": iou_thres,
                "message": "图像处理完成"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)