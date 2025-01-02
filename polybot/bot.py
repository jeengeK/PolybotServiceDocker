# polybot/bot.py




import logging
import os
import requests
import json
import boto3
import time




from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes




# Set logging to be more verbose
logging.basicConfig(
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  level=logging.INFO,
)




# TODO: Setup telegram bot, s3 and yolo5 service configuration parameters via env variables or config file
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION_NAME = os.environ.get("S3_REGION_NAME")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
YOLO5_SERVICE_URL = os.environ.get("YOLO5_SERVICE_URL")


class ObjectDetectionBot:
   def __init__(self, token, telegram_chat_url, s3_bucket_name, s3_client):
       self.telegram_bot_client = telebot.TeleBot(token)
       self.s3_bucket_name = s3_bucket_name
       self.s3_client = s3_client


       self.s3_client = boto3.client('s3', region_name='eu-north-1')  # IAM Role handles authentication


       # Get the NGROK URL from environment variables
       ngrok_url = os.getenv('TELEGRAM_APP_URL')


       if not ngrok_url:
           raise ValueError("NGROK URL (TELEGRAM_APP_URL) is missing in the .env file")


       # Remove any existing webhooks and set a new webhook URL
       self.telegram_bot_client.remove_webhook()
       time.sleep(0.5)
       self.telegram_bot_client.set_webhook(url=f'{ngrok_url}/{token}/', timeout=60)


       logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')


   def send_text(self, chat_id, text):
       self.telegram_bot_client.send_message(chat_id, text)


   def send_text_with_quote(self, chat_id, text, quoted_msg_id):
       self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)


   def is_current_msg_photo(self, msg):
       return 'photo' in msg


   def download_user_photo(self, msg):
       """Downloads the photos that are sent to the Bot to the local file system."""
       if not self.is_current_msg_photo(msg):
           raise RuntimeError(f'Message content of type "photo" expected')


       file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
       data = self.telegram_bot_client.download_file(file_info.file_path)
       folder_name = 'photos'  # Store all photos in a directory named 'photos'


       if not os.path.exists(folder_name):
           os.makedirs(folder_name)


       file_path = os.path.join(folder_name, file_info.file_path.split('/')[-1])
       with open(file_path, 'wb') as photo:
           photo.write(data)


       return file_path


   def send_photo(self, chat_id, img_path):
       if not os.path.exists(img_path):
           raise RuntimeError("Image path doesn't exist")


       self.telegram_bot_client.send_photo(chat_id, InputFile(img_path))


   def upload_to_s3(self, file_path):
       """Uploads the downloaded image to S3."""
       s3_key = os.path.basename(file_path)  # Use the image filename as the S3 key
       self.s3_client.upload_file(file_path, self.s3_bucket_name, s3_key)
       s3_url = f'https://{self.s3_bucket_name}.s3.amazonaws.com/{s3_key}'
       return s3_url


   def get_yolo5_results(self, img_name):
       """Sends an HTTP request to the yolo5 service and returns the predictions."""
       try:
           # Log the image name being sent to YOLOv5
           logger.info(f'Sending imgName to YOLOv5 service: {img_name}')
           # Send the HTTP request to the YOLOv5 service
           logger.debug(f'Preparing to send POST request to http://yolov5:8081/predict with payload: {{"imgName": "{img_name}"}}')
           response = requests.post(
               "http://yolov5:8081/predict",
               json={"imgName": img_name}  # Send imgName as part of the request payload
           )
           # Log the HTTP response status code
           logger.debug(f'Response status code from YOLOv5 service: {response.status_code}')
           # Ensure the response is successful
           response.raise_for_status()
           # Log the successful response data
           logger.info(f'Received response from YOLOv5 service: {response.json()}')  # Log the JSON response
           return response.json()  # Return the JSON response from YOLOv5


       except requests.exceptions.RequestException as e:
           logger.error(f"Error contacting YOLOv5 microservice: {e}")
           logger.debug(f'Error details: {e.__class__} - {str(e)}')
           return None


   def handle_message(self, msg):
       """Main message handler for the bot."""
       logger.info(f'Incoming message: {msg}')
       chat_id = msg['chat']['id']


       if self.is_current_msg_photo(msg):
           try:
               # Step 1: Download the user's photo
               logger.info('Step 1: Downloading user photo')
               photo_path = self.download_user_photo(msg)
               logger.info(f'Photo downloaded to: {photo_path}')


               # Step 2: Upload the image to S3
               logger.info('Step 2: Uploading photo to S3')
               image_url = self.upload_to_s3(photo_path)
               logger.info(f'Photo uploaded to S3: {image_url}')
               self.send_text(chat_id, "Image uploaded to S3. Processing...")


               # Step 3: Get the image filename (imgName) to pass to YOLOv5
               img_name = os.path.basename(photo_path)
               logger.info(f'Image name for YOLOv5: {img_name}')


               # Step 4: Send the image to the YOLOv5 service
               logger.info('Step 4: Sending image to YOLOv5 service')
               yolo_results = self.get_yolo5_results(img_name)
               logger.info(f'YOLOv5 results: {yolo_results}')


               if yolo_results and 'labels' in yolo_results:
                   # Step 5: Send the prediction results back to the user
                   logger.info('Step 5: Sending prediction results to user')
                   detected_objects = [label['class'] for label in yolo_results['labels']]
                   if detected_objects:
                       results_text = f"Detected objects: {', '.join(detected_objects)}"
                   else:
                       results_text = "No objects detected."
                   self.send_text(chat_id, results_text)
               else:
                   logger.error('YOLOv5 service returned no results or incorrect response format')
                   self.send_text(chat_id, "There was an error processing the image.")
#
           except Exception as e:
               logger.error(f"Error processing image: {e}")
               self.send_text(chat_id, "There was an error processing the image.")
       else:
           logger.info('Message does not contain a photo')
           self.send_text(chat_id, "Please send a photo with a valid caption.")








# --- S3 Client Setup ---
s3_client = boto3.client(
  "s3",
  region_name=S3_REGION_NAME,
  aws_access_key_id=S3_ACCESS_KEY,
  aws_secret_access_key=S3_SECRET_KEY,
)








async def handle_image_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
  """Handles incoming image messages."""
  # 1. Download the image
  try:
      image_file = await context.bot.get_file(update.message.photo[-1].file_id)
      image_path = f"/tmp/{image_file.file_unique_id}.jpg"
      await image_file.download_to_drive(image_path)
      logging.info(f"Image downloaded to: {image_path}")
  except Exception as e:
      logging.error(f"Error downloading image: {e}")
      await update.message.reply_text(f"Error downloading image. {e}")
      return








  # 2. Upload the image to S3
  try:
    s3_file_name = f"{time.time()}-{image_file.file_unique_id}.jpg"
    s3_client.upload_file(image_path, S3_BUCKET_NAME, s3_file_name)
    s3_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION_NAME}.amazonaws.com/{s3_file_name}"
    logging.info(f"Image uploaded to S3: {s3_url}")




    os.remove(image_path)  # Delete the local image file
    logging.info(f"Image deleted from local filesystem: {image_path}")
  except Exception as e:
      logging.error(f"Error uploading to S3: {e}")
      await update.message.reply_text(f"Error uploading image to S3. {e}")
      os.remove(image_path)
      return




  # 3. Send an HTTP request to the YOLO5 service
  try:
      yolo5_response = requests.post(YOLO5_SERVICE_URL+"/predict", params={"imgName": s3_file_name})




      yolo5_response.raise_for_status()  # Raise an exception for bad status codes
      yolo5_data = yolo5_response.json()




      logging.info(f"YOLO5 Service Response: {yolo5_data}")




      # 4. Parse the response
      formatted_text = "Detected objects:\n"
      if "objects" in yolo5_data and len(yolo5_data["objects"]) > 0:
          for obj in yolo5_data["objects"]:
               formatted_text += f"{obj['label']}: {obj['count']}\n"
      else:
          formatted_text = "No objects were detected in the image."




      await update.message.reply_text(formatted_text)








  except Exception as e:
      logging.error(f"Error communicating with YOLO5: {e}")
      await update.message.reply_text(f"Error communicating with YOLO5. {e}")
      return








def main():
  logging.info("Starting polybot service")
  # Setup application object for telegram bot
  application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
  # Setup the message handler
  application.add_handler(MessageHandler(filters.PHOTO, handle_image_message))
  # Start the bot listening to telegram messages
  application.run_polling()








if __name__ == '__main__':
  main()



