#!/usr/bin/env python3
"""
API phân loại giới tính từ tiếng nói tiếng Việt - Phiên bản 2
Vietnamese Gender Classification API - Version 2
"""

import os
import pickle
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from gender_classification import VietnameseGenderClassifier

# Khởi tạo FastAPI app
app = FastAPI(
    title="Vietnamese Gender Classification API",
    description="API phân loại giới tính từ tiếng nói tiếng Việt - Hỗ trợ cả đặc trưng cơ bản và nâng cao",
    version="2.0.0"
)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến global để lưu mô hình
models = {
    'basic': {},  # Mô hình cơ bản (MFCC + F0)
    'advanced': {}  # Mô hình nâng cao (thêm RMS, spectral, shimmer)
}
classifier = None

class PredictionResponse(BaseModel):
    """Response model cho dự đoán"""
    success: bool
    prediction: Dict[str, Any]
    error: str = None

def load_models():
    """Tải các mô hình đã lưu"""
    global models, classifier

    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            raise FileNotFoundError("Thư mục models không tồn tại")

        # Tải mô hình cơ bản
        models['basic'] = {
            'svm': pickle.load(open(os.path.join(models_dir, "ver1/svm_model.pkl"), 'rb')),
            'rf': pickle.load(open(os.path.join(models_dir, "ver1/rf_model.pkl"), 'rb')),
            'scaler': pickle.load(open(os.path.join(models_dir, "ver1/scaler.pkl"), 'rb')),
            'info': pickle.load(open(os.path.join(models_dir, "ver1/model_info.pkl"), 'rb'))
        }

        # Tải mô hình nâng cao
        models["advanced"] = {
            "svm": pickle.load(
                open(os.path.join(models_dir, "ver2/svm_model.pkl"), "rb")
            ),
            "rf": pickle.load(
                open(os.path.join(models_dir, "ver2/rf_model.pkl"), "rb")
            ),
            "scaler": pickle.load(
                open(os.path.join(models_dir, "ver2/scaler.pkl"), "rb")
            ),
            "info": pickle.load(
                open(os.path.join(models_dir, "ver2/model_info.pkl"), "rb")
            ),
        }

        # Khởi tạo classifier
        classifier = VietnameseGenderClassifier()

        print("✅ Đã tải thành công các mô hình!")
        return True

    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return False

def predict_basic(audio_path: str) -> Dict[str, Any]:
    """Dự đoán với mô hình cơ bản (MFCC + F0)"""
    try:
        # Trích xuất đặc trưng
        features = classifier.extract_features(audio_path)
        if features is None:
            return {"success": False, "error": "Không thể trích xuất đặc trưng"}
        
        # Chỉ lấy MFCC và F0
        features = features[:14]
        features_scaled = models['basic']['scaler'].transform(features.reshape(1, -1))
        
        # SVM prediction
        svm_pred = models['basic']['svm'].predict(features_scaled)[0]
        svm_decision = models['basic']['svm'].decision_function(features_scaled)[0]
        svm_male_prob = 1 / (1 + np.exp(svm_decision))
        svm_female_prob = 1 - svm_male_prob
        
        # Random Forest prediction  
        rf_pred = models['basic']['rf'].predict(features_scaled)[0]
        rf_prob = models['basic']['rf'].predict_proba(features_scaled)[0]
        
        results = {
            'SVM': {
                'prediction': 'Nữ' if svm_pred == 1 else 'Nam',
                'confidence': float(max(svm_male_prob, svm_female_prob)),
                'probabilities': {'Nam': float(svm_male_prob), 'Nữ': float(svm_female_prob)}
            },
            'RandomForest': {
                'prediction': 'Nữ' if rf_pred == 1 else 'Nam',
                'confidence': float(max(rf_prob)),
                'probabilities': {'Nam': float(rf_prob[0]), 'Nữ': float(rf_prob[1])}
            }
        }
        
        # Kết quả tổng hợp
        avg_male_prob = (svm_male_prob + rf_prob[0]) / 2
        avg_female_prob = (svm_female_prob + rf_prob[1]) / 2
        
        return {
            "success": True,
            "predictions": results,
            "final_prediction": 'Nữ' if avg_female_prob > avg_male_prob else 'Nam',
            "final_confidence": float(max(avg_male_prob, avg_female_prob)),
            "features": {
                "mfcc": features[:13].tolist(),
                "f0": float(features[13]),
                "feature_vector": features.tolist()
            },
            "model_info": models['basic']['info']
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def predict_advanced(audio_path: str) -> Dict[str, Any]:
    """Dự đoán với mô hình nâng cao (thêm RMS, spectral, shimmer)"""
    try:
        # Trích xuất đặc trưng
        features = classifier.extract_features(audio_path)
        if features is None:
            return {"success": False, "error": "Không thể trích xuất đặc trưng"}
            
        features_scaled = models['advanced']['scaler'].transform(features.reshape(1, -1))
        
        # SVM prediction
        svm_pred = models['advanced']['svm'].predict(features_scaled)[0]
        svm_decision = models['advanced']['svm'].decision_function(features_scaled)[0]
        svm_male_prob = 1 / (1 + np.exp(svm_decision))
        svm_female_prob = 1 - svm_male_prob
        
        # Random Forest prediction
        rf_pred = models['advanced']['rf'].predict(features_scaled)[0]
        rf_prob = models['advanced']['rf'].predict_proba(features_scaled)[0]
        
        results = {
            'SVM': {
                'prediction': 'Nữ' if svm_pred == 1 else 'Nam',
                'confidence': float(max(svm_male_prob, svm_female_prob)),
                'probabilities': {'Nam': float(svm_male_prob), 'Nữ': float(svm_female_prob)}
            },
            'RandomForest': {
                'prediction': 'Nữ' if rf_pred == 1 else 'Nam',
                'confidence': float(max(rf_prob)),
                'probabilities': {'Nam': float(rf_prob[0]), 'Nữ': float(rf_prob[1])}
            }
        }
        
        # Kết quả tổng hợp
        avg_male_prob = (svm_male_prob + rf_prob[0]) / 2
        avg_female_prob = (svm_female_prob + rf_prob[1]) / 2
        
        return {
            "success": True,
            "predictions": results,
            "final_prediction": 'Nữ' if avg_female_prob > avg_male_prob else 'Nam',
            "final_confidence": float(max(avg_male_prob, avg_female_prob)),
            "features": {
                "mfcc": features[:13].tolist(),
                "f0": float(features[13]),
                "rms_energy": {
                    "mean": float(features[14]),
                    "std": float(features[15])
                },
                "spectral_rolloff": float(features[16]),
                "shimmer": float(features[17]),
                "feature_vector": features.tolist()
            },
            "model_info": models['advanced']['info']
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Khởi tạo khi API start"""
    print("🚀 Khởi động API phân loại giới tính v2.0...")
    if not load_models():
        print("❌ Không thể tải mô hình. API sẽ không hoạt động.")
    else:
        print("✅ API v2.0 sẵn sàng nhận requests!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vietnamese Gender Classification API",
        "version": "2.0.0",
        "endpoints": {
            "/predict/basic": "Phân loại với đặc trưng cơ bản (MFCC + F0)",
            "/predict/advanced": "Phân loại với đặc trưng nâng cao (thêm RMS, spectral, shimmer)",
            "/health": "Kiểm tra trạng thái API",
            "/model-info": "Thông tin về các mô hình"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "basic_models_loaded": len(models['basic']) > 0,
        "advanced_models_loaded": len(models['advanced']) > 0
    }

@app.post("/predict/basic")
async def predict_gender_basic(file: UploadFile = File(...)):
    """Dự đoán giới tính với đặc trưng cơ bản"""
    try:
        # Lưu file tạm
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Dự đoán
        result = predict_basic(temp_path)
        
        # Xóa file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if result["success"]:
            return PredictionResponse(success=True, prediction=result)
        else:
            return PredictionResponse(success=False, error=result["error"])
            
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/advanced")
async def predict_gender_advanced(file: UploadFile = File(...)):
    """Dự đoán giới tính với đặc trưng nâng cao"""
    try:
        # Lưu file tạm
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Dự đoán
        result = predict_advanced(temp_path)
        
        # Xóa file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if result["success"]:
            return PredictionResponse(success=True, prediction=result)
        else:
            return PredictionResponse(success=False, error=result["error"])
            
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
