from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize PaddleOCR with optimized parameters
ocr_model = PaddleOCR(
    text_detection_model_dir='models/en_PP-OCRv3_det_infer',
    text_recognition_model_dir='models/en_PP-OCRv3_rec_infer',
    use_textline_orientation=True,
    text_rec_score_thresh=0.2
)

# Nigerian phone number regex (matches 10-11 digit numbers starting with 0[7-9] or +234[7-9])
nigerian_phone_pattern = re.compile(r'^(?:\+234[7-9]\d{8,9}|0[7-9]\d{8,9})$')

@app.post("/ocr/")
async def extract_nigerian_numbers(file: UploadFile = File(...)):
    try:
        # Read and decode image
        image_bytes = await file.read()
        logger.debug(f"Image bytes length: {len(image_bytes)}")
        if not image_bytes:
            logger.error("No image data received.")
            return JSONResponse(status_code=400, content={"error": "No image data received."})

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            logger.error("Empty or unreadable image.")
            return JSONResponse(status_code=400, content={"error": "Empty or unreadable image."})

        # Validate image dimensions
        height, width = img.shape[:2]
        if height < 100 or width < 100:
            logger.warning(f"Image too small: {width}x{height}")
        logger.debug(f"Input image shape: {img.shape}, dtype: {img.dtype}")

        # Preprocess image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)  # Reduce noise
        # Skip binarization for now, test contrast only
        contrast_img = cv2.convertScaleAbs(blurred_img, alpha=1.2, beta=20)  # Adjust contrast
        contrast_img = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel BGR

        # Check if preprocessed image is empty
        if contrast_img.size == 0:
            logger.error("Preprocessed image is empty.")
            return JSONResponse(status_code=400, content={"error": "Preprocessed image is empty."})

        logger.debug(f"Preprocessed image shape: {contrast_img.shape}, dtype: {contrast_img.dtype}")

        # Perform OCR on preprocessed image
        logger.debug("Running OCR on preprocessed image...")
        ocr_results = ocr_model.ocr(contrast_img)
        logger.debug(f"OCR raw result type: {type(ocr_results)}, length: {len(ocr_results)}")
        logger.debug(f"OCR raw result content: {ocr_results}")

        # Handle empty or invalid OCR results
        if not ocr_results or not isinstance(ocr_results, list):
            logger.warning("No text detected in OCR results.")
            return {"extracted_phone_numbers": []}

        # Extract Nigerian phone numbers
        extracted_numbers = []
        for page in ocr_results:
            if isinstance(page, dict) and 'rec_texts' in page and 'rec_scores' in page:
                for text, score in zip(page['rec_texts'], page['rec_scores']):
                    # Clean text: remove spaces, dashes, parentheses, dots, slashes, commas, colons, quotes, brackets
                    cleaned_text = re.sub(r'[\s\-\(\)\./,:\'"\[\]]+', '', text)
                    logger.debug(f"Cleaned text: {text} -> {cleaned_text}, confidence: {score}")

                    # Validate cleaned text as a 10-11 digit Nigerian phone number
                    if re.match(nigerian_phone_pattern, cleaned_text) and 10 <= len(cleaned_text) <= 11:
                        extracted_numbers.append(cleaned_text)
                        logger.debug(f"Accepted: {cleaned_text}")
                    else:
                        logger.debug(f"Rejected: {cleaned_text} (not a valid 10-11 digit phone number)")
            else:
                logger.warning("Unexpected OCR result format")

        # Truncate or warn if not 30 numbers
        if len(extracted_numbers) < 30:
            logger.warning(f"Only {len(extracted_numbers)} phone numbers extracted, expected 30")
        elif len(extracted_numbers) > 30:
            logger.warning(f"Truncated {len(extracted_numbers)} phone numbers to 30")
            extracted_numbers = extracted_numbers[:30]

        logger.debug(f"Extracted Nigerian phone numbers: {extracted_numbers}")

        return {"extracted_phone_numbers": extracted_numbers}

    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)