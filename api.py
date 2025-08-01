#!/usr/bin/env python3
"""
API phân loại giới tính từ tiếng nói tiếng Việt
Vietnamese Gender Classification API
"""

import os
import pickle
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from gender_classification import VietnameseGenderClassifier

# Khởi tạo FastAPI app
app = FastAPI(
    title="Vietnamese Gender Classification API",
    description="API phân loại giới tính từ tiếng nói tiếng Việt sử dụng MFCC + F0",
    version="1.0.0"
)

# Biến global để lưu mô hình
models = {}
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
        
        # Kiểm tra xem có thư mục models không
        if not os.path.exists(models_dir):
            raise FileNotFoundError("Thư mục models không tồn tại. Vui lòng chạy save_model.py trước.")
        
        # Tải SVM model
        with open(os.path.join(models_dir, "svm_model.pkl"), 'rb') as f:
            models['svm'] = pickle.load(f)
        
        # Tải Random Forest model
        with open(os.path.join(models_dir, "random_forest_model.pkl"), 'rb') as f:
            models['rf'] = pickle.load(f)
        
        # Tải scaler
        with open(os.path.join(models_dir, "scaler.pkl"), 'rb') as f:
            models['scaler'] = pickle.load(f)
        
        # Tải thông tin mô hình
        with open(os.path.join(models_dir, "model_info.pkl"), 'rb') as f:
            models['info'] = pickle.load(f)
        
        # Khởi tạo classifier để trích xuất đặc trưng
        classifier = VietnameseGenderClassifier()
        
        print("✅ Đã tải thành công các mô hình!")
        print(f"📊 Thông tin mô hình:")
        print(f"   - SVM Accuracy: {models['info']['svm_accuracy']:.4f}")
        print(f"   - Random Forest Accuracy: {models['info']['rf_accuracy']:.4f}")
        print(f"   - Tổng số mẫu: {models['info']['total_samples']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}")
        return False

def predict_gender_from_audio(audio_path: str) -> Dict[str, Any]:
    """Dự đoán giới tính từ file âm thanh"""
    try:
        # Trích xuất đặc trưng
        features = classifier.extract_features(audio_path)
        
        if features is None:
            return {
                "success": False,
                "error": "Không thể trích xuất đặc trưng từ file âm thanh"
            }
        
        # Chuẩn hóa đặc trưng
        features_scaled = models['scaler'].transform(features.reshape(1, -1))
        
        # Dự đoán với cả hai mô hình
        results = {}
        
        # SVM prediction
        svm_pred = models['svm'].predict(features_scaled)[0]
        svm_confidence = abs(models['svm'].decision_function(features_scaled)[0])
        svm_confidence = min(svm_confidence / 2.0, 1.0)
        
        results['SVM'] = {
            'prediction': 'Nữ' if svm_pred == 1 else 'Nam',
            'confidence': float(svm_confidence),
            'gender_code': int(svm_pred)
        }
        
        # Random Forest prediction
        rf_pred = models['rf'].predict(features_scaled)[0]
        rf_prob = models['rf'].predict_proba(features_scaled)[0]
        rf_confidence = max(rf_prob)
        
        results['RandomForest'] = {
            'prediction': 'Nữ' if rf_pred == 1 else 'Nam',
            'confidence': float(rf_confidence),
            'gender_code': int(rf_pred)
        }
        
        # Kết quả tổng hợp
        final_prediction = 'Nữ' if svm_pred == 1 else 'Nam'
        final_confidence = max(svm_confidence, rf_confidence)
        
        return {
            "success": True,
            "predictions": results,
            "final_prediction": final_prediction,
            "final_confidence": float(final_confidence),
            "features": {
                "mfcc": features[:13].tolist(),
                "f0": float(features[13]),
                "feature_vector": features.tolist()
            },
            "model_info": {
                "svm_accuracy": models['info']['svm_accuracy'],
                "rf_accuracy": models['info']['rf_accuracy'],
                "total_samples": models['info']['total_samples']
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Lỗi khi dự đoán: {str(e)}"
        }

@app.on_event("startup")
async def startup_event():
    """Khởi tạo khi API start"""
    print("🚀 Khởi động API phân loại giới tính...")
    if not load_models():
        print("❌ Không thể tải mô hình. API sẽ không hoạt động.")
    else:
        print("✅ API sẵn sàng nhận requests!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vietnamese Gender Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/": "API info",
            "/predict": "Upload audio file for gender classification",
            "/health": "Health check",
            "/model-info": "Model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "available_models": list(models.keys()) if models else []
    }

@app.get("/model-info")
async def model_info():
    """Thông tin mô hình"""
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "model_info": models.get('info', {}),
        "available_models": ["SVM", "RandomForest"],
        "features": models.get('info', {}).get('feature_names', [])
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_gender(file: UploadFile = File(...)):
    """
    Dự đoán giới tính từ file âm thanh
    
    - **file**: File âm thanh (WAV, MP3, etc.)
    """
    # Kiểm tra mô hình đã được tải chưa
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Kiểm tra file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Kiểm tra định dạng file
    allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )
    
    try:
        # Lưu file tạm thời
        temp_file_path = f"temp_{file.filename}"
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Dự đoán
        result = predict_gender_from_audio(temp_file_path)
        
        # Xóa file tạm
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        if result["success"]:
            return PredictionResponse(
                success=True,
                prediction=result
            )
        else:
            return PredictionResponse(
                success=False,
                error=result["error"]
            )
            
    except Exception as e:
        # Xóa file tạm nếu có lỗi
        temp_file_path = f"temp_{file.filename}"
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/predict-url")
async def predict_gender_from_url(audio_url: str):
    """
    Dự đoán giới tính từ URL file âm thanh
    
    - **audio_url**: URL của file âm thanh
    """
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Tải file từ URL (cần thêm thư viện requests)
        import requests
        
        response = requests.get(audio_url)
        response.raise_for_status()
        
        # Lưu file tạm
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(response.content)
        
        # Dự đoán
        result = predict_gender_from_audio(temp_file_path)
        
        # Xóa file tạm
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

if __name__ == "__main__":
    # Chạy API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 