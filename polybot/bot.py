import logging
import os
import requests
import boto3
import time
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# Set logging to be more verbose
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Configuration from environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION_NAME = os.environ.get("S3_REGION_NAME")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
YOLO5_SERVICE_URL = os.environ.get("YOLO5_SERVICE_URL")


# --- S3 Client Setup ---
s3_client = boto3.client(
    "s3",
    region_name=S3_REGION_NAME,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)


async def handle_image_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles incoming image messages, uploads to S3, and processes with YOLOv5."""
    logging.info(f"Incoming message: {update.message}")
    chat_id = update.message.chat_id

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
        # Do not remove the image until it has been used by the YOLOv5 service
        #os.remove(image_path)  # Delete the local image file
        #logging.info(f"Image deleted from local filesystem: {image_path}")
        await update.message.reply_text("Image uploaded to S3. Processing...")  # Sending confirmation

    except Exception as e:
        logging.error(f"Error uploading to S3: {e}")
        await update.message.reply_text(f"Error uploading image to S3. {e}")
        try:
            os.remove(image_path)  # Remove file, if an exception occurred
        except FileNotFoundError:
            pass
        return

    # 3. Send an HTTP request to the YOLO5 service
    try:
        # Log the image name being sent to YOLOv5
        logging.info(f"Sending image to YOLOv5 service using multipart/form-data")

        with open(image_path, 'rb') as image_file_content:
            files = {'image': (s3_file_name, image_file_content)}
            logging.debug(f'Preparing to send POST request to {YOLO5_SERVICE_URL}/predict with the image in the body')

            yolo5_response = requests.post(
                f"{YOLO5_SERVICE_URL}/predict",
                files=files
            )

        # Log the HTTP response status code
        logging.debug(f"Response status code from YOLOv5 service: {yolo5_response.status_code}")
        yolo5_response.raise_for_status()  # Raise an exception for bad status codes
        yolo5_data = yolo5_response.json()
        logging.info(f"YOLO5 Service Response: {yolo5_data}")

        # Delete the local image file after processing
        os.remove(image_path)
        logging.info(f"Image deleted from local filesystem: {image_path}")


        # 4. Parse the response
        formatted_text = "Detected objects:\n"
        if "labels" in yolo5_data and len(yolo5_data["labels"]) > 0:
            for label_data in yolo5_data["labels"]:
                formatted_text += f"{label_data['class']}\n"
        else:
            formatted_text = "No objects were detected in the image."
        await update.message.reply_text(formatted_text)

    except requests.exceptions.RequestException as e:
        logging.error(f"Error communicating with YOLO5 service: {e}")
        logging.debug(f'Error details: {e.__class__} - {str(e)}')
        await update.message.reply_text(f"Error communicating with YOLO5 service. {e}")

    except Exception as e:
        logging.error(f"Error processing image with YOLOv5: {e}")
        await update.message.reply_text(f"Error processing image with YOLOv5. {e}")


def main():
    logging.info("Starting polybot service")
    # Setup application object for telegram bot
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    # Setup the message handler for photos
    application.add_handler(MessageHandler(filters.PHOTO, handle_image_message))
    # Start the bot listening to telegram messages
    application.run_polling()


if __name__ == "__main__":
    main()