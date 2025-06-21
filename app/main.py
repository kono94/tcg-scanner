from app.inferencer import classify_image
from time import time
from app.util import PROJECT_ROOT

if __name__ == "__main__":
    start = time()
    predicted_class_name, confidence = classify_image(PROJECT_ROOT / 'resources' / 'test_images' / 'monkey.png')
    inference_time_ms = (time() - start) * 1000  # Convert seconds to milliseconds
    print(f"Inference time: {inference_time_ms:.2f} ms")
    print(f"Predicted card ID: {predicted_class_name}, Confidence: {confidence:.4f}")