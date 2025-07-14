import pytesseract
from PIL import Image
import os

print(f"Tesseract version: {pytesseract.get_tesseract_version()}")

image_folder = './images/'

if not os.path.exists(image_folder):
    print(f"Folder '{image_folder}' not found. Please add images to this folder.")
else:
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_path.lower().endswith(('png', 'jpg', 'jpeg', 'tiff')):
            print(f"\033[1mProcessing {image_name}...\033[0m")  # Bold text for the image name

            try:
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img)

                print("\033[1mText extracted from image:\033[0m")
                print("-" * 50)
                print(text.strip())
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

