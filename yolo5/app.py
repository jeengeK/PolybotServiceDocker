from flask import Flask, request, jsonify
import boto3
import os
import torch
import time
from PIL import Image
import io
import logging
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    logging.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLOv5 model: {e}")
    model = None  # Set model to None in case of loading error

# Get AWS credentials from environment variables
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION", "us-east-1")

# Get S3 bucket name from env variable
s3_bucket_name = os.environ.get('S3_BUCKET_NAME')

if not all([aws_access_key_id, aws_secret_access_key, s3_bucket_name]):
    logging.error("Missing AWS or S3 configuration. Check environment variables.")
    exit(1) # Exit if critical env variables are missing
else:
    logging.info("AWS and S3 configurations found.")



s3 = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        img_name = request.args.get('imgName')
        if not img_name:
            return jsonify({'error': 'imgName parameter is missing'}), 400

        # Download the image from the S3 bucket
        try:
            logging.info(f"Downloading image {img_name} from S3 bucket {s3_bucket_name}")
            response = s3.get_object(Bucket=s3_bucket_name, Key=img_name)
            image_data = response['Body'].read()
            logging.info(f"Image {img_name} downloaded successfully from S3.")
        except Exception as e:
            logging.error(f"Error downloading image from S3: {e}")
            return jsonify({'error': f"Error downloading image from S3: {e}"}), 500

        # Open and prepare the image
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            logging.info(f"Image {img_name} opened and converted to RGB.")
        except Exception as e:
            logging.error(f"Error opening image: {e}")
            return jsonify({'error': f"Error opening image: {e}"}), 500

        if model is None:
            return jsonify({'error': 'YOLOv5 model is not loaded'}), 500

        # Perform object detection
        try:
            results = model(image)
            predictions = results.pandas().xyxy[0].to_dict(orient="records")
            logging.info(f"Object detection completed successfully for image {img_name}.")
        except Exception as e:
            logging.error(f"Error performing object detection: {e}")
            return jsonify({'error': f"Error performing object detection: {e}"}), 500

        end_time = time.time()
        prediction_time = end_time - start_time
        logging.info(f"Total prediction time for {img_name}: {prediction_time:.4f} seconds.")

        return jsonify({'predictions': predictions, 'prediction_time': prediction_time})
    except Exception as e:
        logging.error(f"Error during prediction process: {e}")
        return jsonify({'error': f"Error during prediction process: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)