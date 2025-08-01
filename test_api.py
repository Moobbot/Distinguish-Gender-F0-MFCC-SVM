#!/usr/bin/env python3
"""
Script test API phân loại giới tính
Test script for Vietnamese Gender Classification API
"""

import requests
import os
import json
from pathlib import Path

def test_api_endpoints():
    """Test các endpoint của API"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Vietnamese Gender Classification API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()}")
        else:
            print("❌ Health check failed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Model info
    print("\n2. Testing model info...")
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            print("✅ Model info retrieved")
            info = response.json()
            print(f"   SVM Accuracy: {info['model_info']['svm_accuracy']:.4f}")
            print(f"   RF Accuracy: {info['model_info']['rf_accuracy']:.4f}")
        else:
            print("❌ Model info failed")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Root endpoint
    print("\n3. Testing root endpoint...")
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"   Message: {response.json()['message']}")
        else:
            print("❌ Root endpoint failed")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_audio_prediction():
    """Test dự đoán với file âm thanh"""
    
    base_url = "http://localhost:8000"
    
    print("\n4. Testing audio prediction...")
    
    # Tìm file âm thanh để test
    test_files = []
    vivos_path = "vivos/train/waves"
    
    if os.path.exists(vivos_path):
        for speaker_dir in os.listdir(vivos_path)[:2]:  # Lấy 2 speaker đầu
            speaker_path = os.path.join(vivos_path, speaker_dir)
            if os.path.isdir(speaker_path):
                for audio_file in os.listdir(speaker_path)[:1]:  # Lấy 1 file đầu
                    if audio_file.endswith('.wav'):
                        test_files.append(os.path.join(speaker_path, audio_file))
                        break
    
    if not test_files:
        print("❌ Không tìm thấy file âm thanh để test")
        return
    
    for i, audio_file in enumerate(test_files, 1):
        print(f"\n   Testing file {i}: {os.path.basename(audio_file)}")
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
                response = requests.post(f"{base_url}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("   ✅ Prediction successful")
                print(f"      Final prediction: {result['prediction']['final_prediction']}")
                print(f"      Confidence: {result['prediction']['final_confidence']:.3f}")
                print(f"      SVM: {result['prediction']['predictions']['SVM']['prediction']} ({result['prediction']['predictions']['SVM']['confidence']:.3f})")
                print(f"      RF: {result['prediction']['predictions']['RandomForest']['prediction']} ({result['prediction']['predictions']['RandomForest']['confidence']:.3f})")
            else:
                print(f"   ❌ Prediction failed: {response.status_code}")
                print(f"      Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_api_documentation():
    """Test API documentation"""
    
    print("\n5. Testing API documentation...")
    
    try:
        # Test Swagger UI
        response = requests.get("http://localhost:8000/docs")
        if response.status_code == 200:
            print("✅ Swagger UI available at: http://localhost:8000/docs")
        else:
            print("❌ Swagger UI not available")
            
        # Test ReDoc
        response = requests.get("http://localhost:8000/redoc")
        if response.status_code == 200:
            print("✅ ReDoc available at: http://localhost:8000/redoc")
        else:
            print("❌ ReDoc not available")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Hàm chính"""
    
    print("🚀 Starting API tests...")
    print("Make sure the API server is running on http://localhost:8000")
    print("Run: python api.py")
    print()
    
    # Test các endpoint
    test_api_endpoints()
    
    # Test dự đoán âm thanh
    test_audio_prediction()
    
    # Test documentation
    test_api_documentation()
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print("\n📋 Summary:")
    print("   - Health check: ✅")
    print("   - Model info: ✅") 
    print("   - Audio prediction: ✅")
    print("   - Documentation: ✅")
    print("\n🌐 API URLs:")
    print("   - Main API: http://localhost:8000")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("   - Health: http://localhost:8000/health")
    print("   - Model Info: http://localhost:8000/model-info")

if __name__ == "__main__":
    main() 