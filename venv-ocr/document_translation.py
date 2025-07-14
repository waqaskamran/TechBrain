def extractTextFromImage(destination):
    app.logger.info("extractTextFromImage destination ..%s", destination)
    try:
        # Use higher resolution for better OCR results
        image = cv2.imread(destination, cv2.IMREAD_GRAYSCALE)

        # Image preprocessing for better OCR
        # Apply adaptive thresholding to improve text detection
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Configure tesseract to preserve layout
        # --psm 1: Automatic page segmentation with OSD (Orientation and Script Detection)
        # --psm 3: Fully automatic page segmentation, but no OSD
        # --oem 3: Default, based on LSTM neural networks
        config = '--oem 3 --psm 1'

        text = pytesseract.image_to_string(image, config=config)

        # Clean the text while preserving structure
        cleaned_text = clean_text_preserve_structure(text)

        return translate_text(text, cleaned_text)

    except Exception as e:
        app.logger.error("error while extracting text...%s ", e)
        raise e

def clean_text_preserve_structure(text):
    """
    Clean text while preserving document structure, including headings and paragraphs.
    """
    # Replace multiple newlines with a single newline to standardize paragraph breaks
    # This preserves paragraph structure but removes excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Convert common Unicode quotes to ASCII
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')

    # Remove empty lines at beginning and end
    text = text.strip()

    # Identify potential headings (all caps lines or short lines followed by newlines)
    lines = text.split('\n')

    processed_lines = []
    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Skip empty lines
        if not stripped_line:
            processed_lines.append('')
            continue

        # Check if this line might be a heading
        is_heading = False

        # Check if it's all uppercase (common for headings)
        if stripped_line.isupper():
            is_heading = True

        # Check if it's short (less than 50 chars) and followed by an empty line
        if len(stripped_line) < 50 and i < len(lines) - 1 and not lines[i+1].strip():
            is_heading = True

        # Preserve heading format
        if is_heading:
            processed_lines.append(f"<h>{stripped_line}</h>")
        else:
            processed_lines.append(stripped_line)

    return '\n'.join(processed_lines)

def translate_text(original_text, cleaned_text, target_language="ar"):
    try:
        libretranslate_url = app.config[constants.CONFIG_LIBRE_TRANSLATION_ENDPOINT] + "/translate"
        headers = {
            "Content-Type": "application/json"
        }

        # Preserve heading tags during translation by adding a special marker
        # This way we can easily identify them after translation
        payload = {
            "q": cleaned_text,
            "source": "auto",
            "target": target_language,
            "format": "text"
        }

        app.logger.info(
            "Sending request to LibreTranslate: URL=%s, Headers=%s, Payload=%s",
            libretranslate_url,
            headers,
            payload
        )
        response = requests.post(libretranslate_url, json=payload, headers=headers)
        response.raise_for_status()

        translated_text = response.json().get("translatedText", "")
        app.logger.info("Translation successful")

        # Format the result with structured data
        structured_original = format_text_with_headings(original_text)
        structured_cleaned = format_text_with_headings(cleaned_text)
        structured_translated = format_text_with_headings(translated_text)

        return {
            "original_text": original_text,
            "OCR_Result": cleaned_text,
            "translated_text": translated_text,
            "structured_original": structured_original,
            "structured_cleaned": structured_cleaned,
            "structured_translated": structured_translated
        }
    except requests.RequestException as e:
        app.logger.error("Translation failed: %s", str(e))
        return {
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "translated_text": None,
            "error": f"Translation failed: {str(e)}"
        }

def format_text_with_headings(text):
    """
    Format text with proper HTML-like structure for displaying with headings.
    """
    # Convert heading tags to proper HTML-like structure
    formatted = re.sub(r'<h>(.*?)</h>', r'<h3>\1</h3>', text)

    # Convert paragraphs (lines separated by double newlines)
    paragraphs = formatted.split('\n\n')
    formatted_paragraphs = []

    for p in paragraphs:
        if p.strip():
            if not (p.startswith('<h3>') and p.endswith('</h3>')):
                formatted_paragraphs.append(f"<p>{p}</p>")
            else:
                formatted_paragraphs.append(p)

    return '\n'.join(formatted_paragraphs)

# Don't forget to add this import at the top of your file
import re