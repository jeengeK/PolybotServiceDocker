import os
import time
import bot
from flask import Flask, request, jsonify
import base64  # Import base64 module
from PIL import Image  # Import Pillow for image handling
from io import BytesIO  # Import BytesIO
import img_proc  # Keep your img_proc import


print("img_proc.py is being imported")


app = Flask(__name__)




@app.route('/process_image', methods=['POST'])
def process_image():
   if 'image' not in request.files:
       return jsonify({'error': 'No image provided'}), 400


   image_file = request.files['image']


   if image_file.filename == '':
       return jsonify({'error': 'No image selected'}), 400


   try:
       start_time = time.time()
       # Open image using Pillow
       image = Image.open(image_file)


       output_image = img_proc.process(image)  # Pass the image object
       end_time = time.time()
       elapsed_time = end_time - start_time


       # Convert image to base64 for JSON
       buffered = BytesIO()
       output_image.save(buffered, format="PNG")  # Save processed image to buffer
       img_str = base64.b64encode(buffered.getvalue()).decode()  # Base64 encode


       return jsonify({
           'message': 'Image processed successfully',
           'elapsed_time': elapsed_time,
           'output_image_base64': img_str  # Use encoded string
       }), 200


   except Exception as e:
       return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

       # TODO: Download the image from the S3 bucket
   try:
       logging.info(f"Downloading image {img_name} from S3 bucket {s3_bucket_name}")
       response = s3.get_object(Bucket=s3_bucket_name, Key=img_name)
       image_data = response['Body'].read()
       logging.info(f"Image {img_name} downloaded successfully from S3.")
   except Exception as e:
       logging.error(f"Error downloading image from S3: {e}")
       return jsonify({'error': f"Error downloading image from S3: {e}"}), 500

       # TODO: Open and prepare the image
   try:
       image = Image.open(io.BytesIO(image_data)).convert('RGB')
       logging.info(f"Image {img_name} opened and converted to RGB.")
   except Exception as e:
       logging.error(f"Error opening image: {e}")
       return jsonify({'error': f"Error opening image: {e}"}), 500

   if model is None:
       return jsonify({'error': 'YOLOv5 model is not loaded'}), 500

       # TODO: Perform object detection
   try:
       results = model(image)
       predictions = results.pandas().xyxy[0].to_dict(orient="records")
       logging.info(f"Object detection completed successfully for image {img_name}.")
   except Exception as e:
       logging.error(f"Error performing object detection: {e}")
       return jsonify({'error': f"Error performing object detection: {e}"}), 500


PORT = int(os.environ.get("PORT", "8080"))
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=PORT)


