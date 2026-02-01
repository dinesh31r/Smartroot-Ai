import time
import os
import sys
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.cnn_inference import ModelRegistry, predict_stress, predict_stress_ensemble
from backend.root_image_analyzer import analyze_root_image

def generate_synthetic_plant_image(size=128):
    """Generates a green image to pass is_probable_plant check."""
    # Create a green image (HSV: ~60 (Green), Sat: High, Val: High)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = [30, 200, 50] # BGR for Greenish
    return img

def generate_synthetic_root_image(size=512):
    """Generates a simple root-like image (white lines on black background)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Draw some "roots"
    cv2.line(img, (size//2, 0), (size//2, size), (255, 255, 255), 2)
    cv2.line(img, (size//2, size//2), (0, size), (255, 255, 255), 2)
    cv2.line(img, (size//2, size//2), (size, size), (255, 255, 255), 2)
    return img

def benchmark_models():
    print("="*50)
    print("🚀 STARTING MODEL PERFORMANCE BENCHMARK")
    print("="*50)

    # Prepare temp images
    plant_img_path = "temp_plant.jpg"
    root_img_path = "temp_root.jpg"
    cv2.imwrite(plant_img_path, generate_synthetic_plant_image())
    cv2.imwrite(root_img_path, generate_synthetic_root_image())

    results = []

    models = ["default", "resnet50", "efficientnet"]
    
    # 1. Model Loading Time (Cold Start)
    print("\n📦 Benchmarking Model Loading Time...")
    for model_name in models:
        # Clear cache/registry if possible or just measure call time (registry handles caching)
        # Note: ModelRegistry caches, so we are measuring first access vs subsequent if we wanted
        # But here we want 'Loading' time, so we'll assume it's the first time strictly for this script run
        pass 
        # Actually ModelRegistry doesn't have a clear method, but we can measure the first prediction call overhead
        # or we can modify registry. Let's just measure first inference which attempts to load.
        
    
    # Let's measure Inference Time
    print("\n⚡ Benchmarking Inference Latency (Batch Size=1)...")
    
    for model_name in models:
        print(f"  👉 Testing {model_name}...")
        
        # Cold start / Loading
        start_time = time.time()
        ModelRegistry.load_model(model_name)
        load_time = (time.time() - start_time) * 1000
        
        # Warmup
        predict_stress(plant_img_path, model_name=model_name)
        
        # Measure Inference
        latencies = []
        for _ in range(10):
            start = time.time()
            predict_stress(plant_img_path, model_name=model_name)
            latencies.append((time.time() - start) * 1000)
            
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        results.append({
            "Task": f"Inference ({model_name})",
            "Load Time (ms)": round(load_time, 2),
            "Avg Latency (ms)": round(avg_latency, 2),
            "P99 Latency (ms)": round(p99_latency, 2)
        })

    # Benchmark Ensemble
    print(f"  👉 Testing Ensemble...")
    start_time = time.time()
    predict_stress_ensemble(plant_img_path, model_names=models)
    # Warmup done by individual calls above mostly, but ensemble logic has overhead
    latencies = []
    for _ in range(5):
        start = time.time()
        predict_stress_ensemble(plant_img_path, model_names=models)
        latencies.append((time.time() - start) * 1000)
    
    results.append({
        "Task": "Inference (Ensemble)",
        "Load Time (ms)": "N/A",
        "Avg Latency (ms)": round(np.mean(latencies), 2),
        "P99 Latency (ms)": round(np.percentile(latencies, 99), 2)
    })

    # 2. Root Analysis Benchmark
    print("\n🌱 Benchmarking Root Analysis...")
    
    # Fast Mode
    start = time.time()
    analyze_root_image(root_img_path, fast=True)
    load_time = (time.time() - start) * 1000 # Treat first run as load/init
    
    latencies = []
    for _ in range(5):
        start = time.time()
        analyze_root_image(root_img_path, fast=True)
        latencies.append((time.time() - start) * 1000)
        
    results.append({
        "Task": "Root Analysis (Fast)",
        "Load Time (ms)": round(load_time, 2),
        "Avg Latency (ms)": round(np.mean(latencies), 2),
        "P99 Latency (ms)": round(np.percentile(latencies, 99), 2)
    })

    # Detailed Mode
    latencies = []
    for _ in range(5):
        start = time.time()
        analyze_root_image(root_img_path, fast=False)
        latencies.append((time.time() - start) * 1000)
        
    results.append({
        "Task": "Root Analysis (Detailed)",
        "Load Time (ms)": "N/A", # Already loaded by fast
        "Avg Latency (ms)": round(np.mean(latencies), 2),
        "P99 Latency (ms)": round(np.percentile(latencies, 99), 2)
    })

    # Cleanup
    if os.path.exists(plant_img_path): os.remove(plant_img_path)
    if os.path.exists(root_img_path): os.remove(root_img_path)

    # Display Results
    print("\n" + "="*50)
    print("📊 BENCHMARK RESULTS")
    print("="*50)
    print(f"{'Task':<25} | {'Load (ms)':<10} | {'Avg (ms)':<10} | {'P99 (ms)':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['Task']:<25} | {str(r['Load Time (ms)']):<10} | {str(r['Avg Latency (ms)']):<10} | {str(r['P99 Latency (ms)']):<10}")
    print("-" * 65)
    print("✅ Done!")

if __name__ == "__main__":
    benchmark_models()
