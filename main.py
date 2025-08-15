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
        contrast_img = cv2.convertScaleAbs(blurred_img, alpha=1.2, beta=20)  # Adjust contrast
        contrast_img = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel BGR

        if contrast_img.size == 0:
            logger.error("Preprocessed image is empty.")
            return JSONResponse(status_code=400, content={"error": "Preprocessed image is empty."})

        logger.debug(f"Preprocessed image shape: {contrast_img.shape}, dtype: {contrast_img.dtype}")

        # Perform OCR on preprocessed image
        logger.debug("Running OCR on preprocessed image...")
        ocr_results = ocr_model.ocr(contrast_img)
        print("RAW OCR RESULTS >>>", ocr_results)
        logger.debug(f"OCR raw result type: {type(ocr_results)}, length: {len(ocr_results)}")
        logger.debug(f"OCR raw result content: {ocr_results}")

        if not ocr_results or not isinstance(ocr_results, list):
            logger.warning("No text detected in OCR results.")
            return {"extracted_phone_numbers": []}

        # Extract Nigerian phone numbers
        extracted_numbers = []
        for page in ocr_results:
            for entry in page:
                if len(entry) != 2:
                    continue
                _, (text, score) = entry

                # Clean text
                cleaned_text = re.sub(r'[\s\-\(\)\./,:\'"\[\]]+', '', text)
                logger.debug(f"Cleaned text: {text} -> {cleaned_text}, confidence: {score}")

                # Extract phone number candidates
                candidates = re.findall(r'(?:\+234|0)\d{9,10}', cleaned_text)
                for number in candidates:
                    if re.match(nigerian_phone_pattern, number) and 10 <= len(number[-11:]) <= 11:
                        extracted_numbers.append(number)
                        logger.debug(f"Accepted: {number}")
                    else:
                        logger.debug(f"Rejected: {number} (not a valid Nigerian phone number)")

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
