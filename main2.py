import pytesseract
from PIL import Image
import cv2
import os
import numpy as np

print(f"Tesseract version: {pytesseract.get_tesseract_version()}\n")

# Define the path to the 'images' folder that contains document images
image_folder = './images/'  # Change this path to wherever your images are stored


if not os.path.exists(image_folder):
    print(f"Folder '{image_folder}' not found. Please add images to this folder.")
else:
    # Process each image in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_path.lower().endswith(('png', 'jpg', 'jpeg', 'tiff')):  # Process only image files
            print(f"\033[1mProcessing {image_name}...\033[0m")  # Bold text for the image name

            # Open the image using Pillow
            try:
                # Open the image using OpenCV for preprocessing
                img = cv2.imread(image_path)

                # Convert image to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Apply thresholding (Binarization) to get black text on white background
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                # Optionally, you can apply noise removal or blurring here (if needed)
                # blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

                # If needed, you can also apply dilation to help improve text detection
                # dilated = cv2.dilate(thresh, None, iterations=1)

                # Convert back to PIL Image format for pytesseract
                pil_image = Image.fromarray(thresh)

                # Run Tesseract OCR on the preprocessed image
                text = pytesseract.image_to_string(pil_image)

                # Pretty print extracted text
                print("\033[1mText extracted from image:\033[0m")  # Bold title for text
                print("-" * 50)  # Separator for readability
                print(text.strip())  # Remove any extra spaces or newlines
                print("-" * 50)  # Separator for readability

            except Exception as e:
                print(f"Error processing {image_name}: {e}")
