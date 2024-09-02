from flask import Flask, request, jsonify
import boto3
import os
from PIL import Image

app = Flask(__name__)

# Load the YOLOv5 model
from ultralytics import YOLO

model = YOLO('yolov5s.pt')  # Load the lightweight YOLOv5s model
 # Using a lightweight model

# AWS S3 setup
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
BUCKET_NAME = os.getenv('netflix.jeenge')
s3_client = boto3.client('s3')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image name from the query parameter
        img_name = request.args.get('imgName')
        if not img_name:
            return jsonify({"error": "imgName query parameter is required"}), 400

        # Define local file path to save the downloaded image
        local_img_path = f"/tmp/{img_name}"

        # TODO: Download the image from S3
        try:
            s3_client.download_file(BUCKET_NAME, img_name, local_img_path)
        except Exception as e:
            return jsonify({"error": f"Failed to download image from S3: {str(e)}"}), 500

        # TODO: Load the image using PIL
        try:
            img = Image.open(local_img_path)
        except Exception as e:
            return jsonify({"error": f"Failed to open the image: {str(e)}"}), 400

        # TODO: Perform object detection
        results = model.predict(img)
        detected_objects = []
        for result in results:
            for box, label, conf in zip(result.boxes.xyxy, result.boxes.names, result.boxes.conf):
                detected_objects.append({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": [float(coord) for coord in box]
                })

        # Clean up local file
        if os.path.exists(local_img_path):
            os.remove(local_img_path)

        # Return the detected objects
        return jsonify({"detected_objects": detected_objects}), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
