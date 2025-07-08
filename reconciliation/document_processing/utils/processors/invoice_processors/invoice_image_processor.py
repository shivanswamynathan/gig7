import os
import json
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import pdf2image
import google.generativeai as genai
from typing import Dict, List, Any, Optional, Tuple
import re
from pathlib import Path
from dotenv import load_dotenv
import logging
import base64
import io
import tempfile
from datetime import datetime
import time  # Ensure time is imported for retry logic

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class InvoiceImageProcessor:
    """
    Enhanced Invoice Image processor using advanced OCR techniques and Gemini AI
    Now uses dynamic retry logic for Gemini API rate limits (429 errors).
    Fixed delay is no longer required or recommended.
    """

    def __init__(self, delay_seconds: float = 0):
        """Initialize the Enhanced Invoice Processor. delay_seconds is ignored (kept for backward compatibility)."""
        logger.info("Initializing Enhanced Invoice Image Processor...")
        
        # Configure Tesseract
        tesseract_cmd = os.getenv('TESSERACT_CMD', 'tesseract')
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            logger.info(f"Tesseract configured: {tesseract_cmd}")
        else:
            logger.warning("Tesseract path not found in environment. Using system default.")
        
        # Test Tesseract installation
        self._test_tesseract()
        
        # Configure Gemini API
        gemini_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini AI configured successfully")
        else:
            logger.warning("Gemini API key not found. LLM features disabled.")
            self.model = None
        
        # Check Poppler path for PDF processing
        poppler_path = os.getenv('POPPLER_PATH')
        if poppler_path and os.path.exists(poppler_path):
            logger.info(f"Poppler configured: {poppler_path}")
        else:
            logger.info("Poppler path not found, will try system PATH")
        
        # Define the target schema (maintaining compatibility with existing structure)
        self.invoice_schema = {
            "vendor_details": {
                "vendor_name": "",
                "vendor_gst": "",
                "vendor_pan": ""
            },
            "invoice_info": {
                "invoice_number": "",
                "invoice_date": ""
            },
            "line_items": [
                {
                    "sr_no": "",
                    "item_description": "",
                    "hsn_sac_code": "",
                    "quantity": "",
                    "unit": "",
                    "rate_per_unit": "",
                    "gross_amount": "",
                    "discount_amount": "",
                    "taxable_amount": "",
                    "gst_rate_percent": "",
                    "cgst_rate": "",
                    "cgst_amount": "",
                    "sgst_rate": "",
                    "sgst_amount": "",
                    "igst_rate": "",
                    "igst_amount": "",
                    "total_gst_on_item": "",
                    "final_amount_including_gst": ""
                }
            ],
            "tax_summary_by_hsn": [
                {
                    "hsn_sac_code": "",
                    "taxable_amount": "",
                    "cgst_rate": "",
                    "cgst_amount": "",
                    "sgst_rate": "",
                    "sgst_amount": "",
                    "igst_rate": "",
                    "igst_amount": "",
                    "total_tax_amount": ""
                }
            ],
            "invoice_totals": {
                "total_items": "",
                "total_quantity": "",
                "gross_total_before_discount": "",
                "total_discount": "",
                "total_taxable_amount": "",
                "total_cgst": "",
                "total_sgst": "",
                "total_igst": "",
                "total_gst": "",
                "final_invoice_amount": ""
            }
        }
        
        self.delay_seconds = 0  # delay_seconds is now ignored; dynamic retry is used instead
        logger.info("Enhanced Invoice Image Processor initialized successfully!")

    def _test_tesseract(self):
        """Test if Tesseract is properly configured and accessible"""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR version: {version}")
        except Exception as e:
            logger.error(f"Tesseract configuration error: {e}")
            logger.error("Please ensure Tesseract is installed and TESSERACT_CMD environment variable is set correctly.")
            raise Exception(f"Tesseract OCR not properly configured: {str(e)}")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply advanced image preprocessing techniques to improve OCR accuracy."""
        try:
            logger.info("Preprocessing image for better OCR accuracy...")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Noise reduction
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Morphological operations to clean up the image
            kernel = np.ones((1,1), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(binary)
            
            # Additional PIL enhancements
            processed_image = processed_image.filter(ImageFilter.MedianFilter())
            enhancer = ImageEnhance.Sharpness(processed_image)
            processed_image = enhancer.enhance(2.0)
            
            logger.info("Image preprocessing completed")
            return processed_image
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return image

    def extract_plain_text(self, image: Image.Image) -> str:
        """Extract plain text from image using advanced Tesseract OCR."""
        try:
            logger.info("Extracting plain text using enhanced OCR...")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            configs = [
                '--psm 6',  
                '--psm 4',  
                '--psm 3',  
                '--psm 11', 
                '--psm 12'  
            ]
            
            best_text = ""
            max_confidence = 0
            
            for i, config in enumerate(configs):
                try:
                    logger.info(f"  Trying plain text config {i+1}/{len(configs)}: {config}")
                    
                    # Get text with confidence
                    data = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        logger.info(f"    Average confidence: {avg_confidence:.2f}%")
                        
                        if avg_confidence > max_confidence:
                            max_confidence = avg_confidence
                            text = pytesseract.image_to_string(processed_image, config=config)
                            if text.strip():
                                best_text = text
                                logger.info(f"Best plain text result (confidence: {avg_confidence:.2f}%)")
                                
                except Exception as e:
                    logger.warning(f"Plain text OCR config failed: {e}")
                    continue
            
            # Fallback to basic OCR if no good result
            if not best_text.strip():
                logger.info("Falling back to basic plain text OCR...")
                best_text = pytesseract.image_to_string(processed_image)
            
            logger.info(f"Plain text extraction completed. Extracted {len(best_text)} characters")
            return best_text.strip()
            
        except Exception as e:
            logger.error(f"Error in plain text extraction: {e}")
            return ""

    def extract_key_value_pairs(self, image: Image.Image) -> Dict[str, str]:
        """Extract key-value pairs from image using bounding box analysis."""
        try:
            logger.info("Extracting key-value pairs using bounding box analysis...")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Get detailed OCR data with bounding boxes
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            # Extract words with their positions and confidence
            words_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Only consider words with good confidence
                    word_info = {
                        'text': data['text'][i].strip(),
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': data['conf'][i]
                    }
                    if word_info['text']:  # Only add non-empty words
                        words_data.append(word_info)
            
            logger.info(f"  Found {len(words_data)} words with good confidence")
            
            # Group words into lines based on vertical position
            lines = self._group_words_into_lines(words_data)
            logger.info(f"  Grouped into {len(lines)} lines")
            
            # Extract key-value pairs from lines
            key_value_pairs = self._extract_kv_pairs_from_lines(lines)
            
            logger.info(f"Extracted {len(key_value_pairs)} key-value pairs")
            return key_value_pairs
            
        except Exception as e:
            logger.error(f"Error in key-value pair extraction: {e}")
            return {}

    def _group_words_into_lines(self, words_data: List[Dict]) -> List[List[Dict]]:
        """Group words into lines based on their vertical position."""
        if not words_data:
            return []
        sorted_words = sorted(words_data, key=lambda x: x['top'])
        
        lines = []
        current_line = [sorted_words[0]]
        line_tolerance = 10 
        
        for word in sorted_words[1:]:
            # Check if word is on the same line (similar top position)
            if abs(word['top'] - current_line[-1]['top']) <= line_tolerance:
                current_line.append(word)
            else:
                # Sort current line by left position (x-coordinate)
                current_line.sort(key=lambda x: x['left'])
                lines.append(current_line)
                current_line = [word]
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda x: x['left'])
            lines.append(current_line)
        
        return lines

    def _extract_kv_pairs_from_lines(self, lines: List[List[Dict]]) -> Dict[str, str]:
        """Extract key-value pairs from grouped lines."""
        key_value_pairs = {}
        
        # Common key patterns for invoices
        key_patterns = [
            r'(?:invoice|bill)\s*(?:no|number|#)',
            r'(?:date|dated)',
            r'(?:gst|gstin)\s*(?:no|number)?',
            r'(?:pan|pan\s*no)',
            r'(?:vendor|supplier|company)\s*(?:name)?',
            r'(?:total|grand\s*total|final\s*amount)',
            r'(?:quantity|qty)',
            r'(?:rate|price|amount)',
            r'(?:hsn|sac)',
            r'(?:cgst|sgst|igst)',
            r'(?:taxable|tax)\s*(?:amount|value)',
            r'(?:description|particulars|item)'
        ]
        
        for line in lines:
            if len(line) < 2:  # Need at least 2 words for key-value
                continue
            
            # Reconstruct line text
            line_text = ' '.join([word['text'] for word in line])
            
            # Look for colon-separated key-value pairs
            if ':' in line_text:
                parts = line_text.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    
                    # Check if key matches common patterns
                    for pattern in key_patterns:
                        if re.search(pattern, key, re.IGNORECASE):
                            key_value_pairs[key] = value
                            break
                    else:
                        # Add even if doesn't match patterns (might be useful)
                        if len(key) > 2 and len(value) > 0:
                            key_value_pairs[key] = value
            
            # Look for label-value patterns (words close to each other)
            elif len(line) >= 2:
                for i in range(len(line) - 1):
                    key_word = line[i]['text'].lower()
                    value_word = line[i + 1]['text']
                    
                    # Check if key matches patterns
                    for pattern in key_patterns:
                        if re.search(pattern, key_word, re.IGNORECASE):
                            # Check if words are close enough horizontally
                            distance = line[i + 1]['left'] - (line[i]['left'] + line[i]['width'])
                            if distance < 100:  # Within 100 pixels
                                key_value_pairs[key_word] = value_word
                            break
        
        return key_value_pairs

    def extract_table_data(self, image: Image.Image) -> List[Dict[str, str]]:
        """Extract table/structured data from image."""
        try:
            logger.info("Extracting table data...")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
            
            # Extract words with positions
            words_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30 and data['text'][i].strip():
                    words_data.append({
                        'text': data['text'][i].strip(),
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'conf': data['conf'][i]
                    })
            
            # Group into table rows and columns
            table_data = self._extract_table_structure(words_data)
            
            logger.info(f"Extracted {len(table_data)} table rows")
            return table_data
            
        except Exception as e:
            logger.error(f"Error in table extraction: {e}")
            return []

    def _extract_table_structure(self, words_data: List[Dict]) -> List[Dict[str, str]]:
        """Extract table structure from word positions."""
        if not words_data:
            return []
        
        # Group words into rows
        lines = self._group_words_into_lines(words_data)
        
        # Find potential table headers and data rows
        table_rows = []
        header_row = None
        
        for line_idx, line in enumerate(lines):
            if len(line) >= 3:  # Potential table row (at least 3 columns)
                line_text = [word['text'] for word in line]
                
                # Check if this looks like a header row
                if any(keyword in ' '.join(line_text).lower() for keyword in 
                       ['description', 'qty', 'rate', 'amount', 'hsn', 'total', 'particulars']):
                    header_row = line_text
                    continue
                
                # If we have a header, map data to it
                if header_row and len(line_text) <= len(header_row):
                    row_data = {}
                    for i, value in enumerate(line_text):
                        if i < len(header_row):
                            row_data[header_row[i].lower()] = value
                    table_rows.append(row_data)
                else:
                    # Generic row without header mapping
                    row_data = {f'col_{i}': value for i, value in enumerate(line_text)}
                    table_rows.append(row_data)
        
        return table_rows

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for Gemini."""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""

    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images using Poppler path from .env if available."""
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            
            poppler_path = os.getenv('POPPLER_PATH')
            if poppler_path and os.path.exists(poppler_path):
                logger.info(f"  Using Poppler from: {poppler_path}")
                images = pdf2image.convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
            else:
                logger.info("  Using system Poppler installation")
                images = pdf2image.convert_from_path(pdf_path, dpi=300)
            
            logger.info(f"Converted PDF to {len(images)} images")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            if "poppler" in str(e).lower():
                logger.error("Hint: Make sure Poppler is installed and POPPLER_PATH is correct in .env file")
            return []

    def _invoke_gemini_with_retry(self, content) -> Any:
        """
        Invoke Gemini LLM with automatic retry on 429 errors using retry_delay from error message.
        """
        max_retries = 5
        attempt = 0
        while True:
            try:
                return self.model.generate_content(content)
            except Exception as e:
                err_msg = str(e)
                if '429' in err_msg and 'retry_delay' in err_msg:
                    import re
                    match = re.search(r'retry_delay[":\s]*([0-9.]+)', err_msg)
                    if match:
                        retry_delay = float(match.group(1))
                        logger.warning(f"Gemini API rate limit hit (429). Retrying after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        attempt += 1
                        if attempt >= max_retries:
                            logger.error(f"Max retries reached for Gemini API 429 errors.")
                            raise
                        continue
                raise  # re-raise other errors

    def extract_structured_data_with_gemini(self, plain_text: str, key_value_pairs: Dict[str, str], 
                                         table_data: List[Dict], image: Image.Image) -> Dict[str, Any]:
        """Use Gemini LLM with multiple data sources for enhanced extraction. Now uses dynamic retry for rate limiting."""
        if not self.model:
            logger.warning("LLM model not available. Falling back to rule-based extraction.")
            return self.extract_with_rules(plain_text)
        
        try:
            logger.info("Using Gemini AI with enhanced multi-modal analysis...")
            
            # Convert image to base64 for Gemini
            image_base64 = self.image_to_base64(image)
            
            prompt = f"""
            You are an expert at extracting structured data from Indian GST invoices. 
            I'm providing you with multiple data sources from the same invoice:
            
            1. PLAIN TEXT (from OCR):
            {plain_text}
            
            2. KEY-VALUE PAIRS (extracted using bounding box analysis):
            {json.dumps(key_value_pairs, indent=2)}
            
            3. TABLE DATA (structured table information):
            {json.dumps(table_data, indent=2)}
            
            4. ORIGINAL IMAGE: [Image provided separately]
            
            Please analyze ALL these data sources together and extract the most accurate information.
            Cross-reference between the different sources to resolve any conflicts or ambiguities.
            
            EXTRACTION RULES:
            1. Use information from ALL sources - plain text, key-value pairs, table data, and visual analysis
            2. When there are conflicts between sources, prioritize the most reliable/consistent information
            3. If a field is not found in any source, use an empty string ""
            4. For numerical values, extract only the number (remove currency symbols like ₹, $, etc.)
            5. For dates, use DD/MM/YYYY or DD-MM-YYYY format
            6. For GST numbers, extract the complete 15-digit alphanumeric code
            7. For PAN numbers, extract the 10-character alphanumeric code
            8. For each line item, extract ALL GST details including rates and amounts
            9. Unit of measurement should be units like KG, PCS, NOS, LTR, etc.
            10. HSN codes should be extracted for each item - extract the FULL HSN code as shown
            11. Extract the Tax Summary section data into tax_summary_by_hsn array
            12. Be precise and accurate - cross-validate between all data sources
            13. Return ONLY the JSON object, no additional text

            VENDOR DETAILS:
            - vendor_name: Extract company/business name from header
            - vendor_gst: Extract 15-digit GSTIN (format: ##AAAAA####A#A)
            - vendor_pan: Extract 10-character PAN (if not shown, derive from GST positions 3-12)
            
            INVOICE INFO:
            - invoice_number: Extract invoice/bill number
            - invoice_date: Extract invoice date in DD/MM/YYYY format
            
            LINE ITEMS:
            For each product/service line (use table data when available):
            - sr_no: Serial number from invoice
            - item_description: Full product/service description
            - hsn_sac_code: HSN or SAC code for the item
            - quantity: Numeric quantity
            - unit: Unit of measurement (PCS, KG, LTR, etc.)
            - rate_per_unit: Rate per unit before tax
            - gross_amount: Quantity × Rate (before discount)
            - discount_amount: Discount amount (if any)
            - taxable_amount: Amount on which GST is calculated
            - gst_rate_percent: Total GST rate (CGST + SGST + IGST)
            - cgst_rate: CGST rate percentage
            - cgst_amount: CGST amount in rupees
            - sgst_rate: SGST rate percentage  
            - sgst_amount: SGST amount in rupees
            - igst_rate: IGST rate percentage
            - igst_amount: IGST amount in rupees
            - total_gst_on_item: Total GST for this item
            - final_amount_including_gst: Final amount including all taxes
            
            TAX SUMMARY BY HSN:
            Extract from the "Tax Summary" or similar section:
            - hsn_sac_code: HSN/SAC code
            - taxable_amount: Taxable amount for this HSN
            - cgst_rate & cgst_amount: CGST rate and amount
            - sgst_rate & sgst_amount: SGST rate and amount  
            - igst_rate & igst_amount: IGST rate and amount
            - total_tax_amount: Total tax for this HSN
            
            INVOICE TOTALS:
            - total_items: Count of different line items
            - total_quantity: Sum of all quantities
            - gross_total_before_discount: Total before any discounts
            - total_discount: Total discount amount
            - total_taxable_amount: Total taxable amount
            - total_cgst: Total CGST amount
            - total_sgst: Total SGST amount
            - total_igst: Total IGST amount
            - total_gst: Total GST amount
            - final_invoice_amount: Final invoice amount
            
            Required JSON Schema:
            {json.dumps(self.invoice_schema, indent=2)}
            
            Analyze all the provided data sources and return the extracted information in the exact JSON format.
            """
            
            content = [
                prompt,
                {
                    "mime_type": "image/png",
                    "data": image_base64
                }
            ]
            
            response = self._invoke_gemini_with_retry(content)
            
            # Clean and parse the response
            response_text = response.text.strip()
            
            # Remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                extracted_data = json.loads(response_text)
                logger.info("Successfully extracted data using enhanced Gemini AI analysis")
                return extracted_data
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in LLM response: {e}")
                logger.error(f"Raw response (first 500 chars): {response_text[:500]}...")
                logger.info("Falling back to rule-based extraction")
                return self.extract_with_rules(plain_text)
                
        except Exception as e:
            logger.error(f"Error in enhanced LLM extraction: {e}")
            logger.info("Falling back to rule-based extraction")
            return self.extract_with_rules(plain_text)

    def extract_with_rules(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based extraction method."""
        logger.info("Using rule-based extraction as fallback...")
        
        result = json.loads(json.dumps(self.invoice_schema))  # Deep copy
        
        # Extract vendor details
        logger.info("  Extracting vendor details...")
        gst_match = re.search(r'GST[^:]*:?\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9][Z][0-9])', text, re.IGNORECASE)
        if gst_match:
            result["vendor_details"]["vendor_gst"] = gst_match.group(1)
            logger.info(f"    Found GST: {gst_match.group(1)}")
        
        pan_match = re.search(r'PAN[^:]*:?\s*([A-Z]{5}[0-9]{4}[A-Z])', text, re.IGNORECASE)
        if pan_match:
            result["vendor_details"]["vendor_pan"] = pan_match.group(1)
            logger.info(f"    Found PAN: {pan_match.group(1)}")
        
        # Extract invoice info
        logger.info("  Extracting invoice information...")
        invoice_num_patterns = [
            r'Invoice\s*(?:No\.?|Number)[:\s]*([A-Z0-9\-/]+)',
            r'Bill\s*(?:No\.?|Number)[:\s]*([A-Z0-9\-/]+)',
            r'(?:Inv|Bill)\s*#?\s*([A-Z0-9\-/]+)'
        ]
        
        for pattern in invoice_num_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["invoice_info"]["invoice_number"] = match.group(1)
                logger.info(f"    Found Invoice Number: {match.group(1)}")
                break
        
        date_match = re.search(r'(?:Date|Dated)[:\s]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', text, re.IGNORECASE)
        if date_match:
            result["invoice_info"]["invoice_date"] = date_match.group(1)
            logger.info(f"    Found Date: {date_match.group(1)}")
        
        # Extract totals
        logger.info("  Extracting totals...")
        total_patterns = [
            r'Total[:\s]*(?:Rs\.?\s*)?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'Grand\s*Total[:\s]*(?:Rs\.?\s*)?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'Final\s*Amount[:\s]*(?:Rs\.?\s*)?(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount = match.group(1).replace(',', '')
                result["invoice_totals"]["final_invoice_amount"] = amount
                logger.info(f"    Found Total Amount: ₹{amount}")
                break
        
        logger.info("Rule-based extraction completed")
        return result

    def validate_and_clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the extracted data."""
        try:
            logger.info("Validating and cleaning extracted data...")
            
            # Clean monetary values
            def clean_amount(value):
                if isinstance(value, str):
                    cleaned = re.sub(r'[^\d.]', '', value)
                    return cleaned if cleaned else ""
                return str(value) if value else ""
            
            # Clean vendor details
            if "vendor_details" in data:
                vendor = data["vendor_details"]
                if "vendor_gst" in vendor and vendor["vendor_gst"]:
                    # Validate GST format
                    gst = vendor["vendor_gst"].upper().replace(" ", "")
                    if re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9][Z][0-9]$', gst):
                        vendor["vendor_gst"] = gst
                        logger.info(f"Validated GST: {gst}")
                    else:
                        logger.warning(f"Invalid GST format: {gst}")
            
            # Clean invoice totals
            if "invoice_totals" in data:
                totals = data["invoice_totals"]
                for key, value in totals.items():
                    if "amount" in key.lower() or "total" in key.lower():
                        original = totals[key]
                        totals[key] = clean_amount(value)
                        if original != totals[key] and totals[key]:
                            logger.info(f"Cleaned {key}: {original} → {totals[key]}")
            
            # Clean line items
            if "line_items" in data:
                for i, item in enumerate(data["line_items"]):
                    for key, value in item.items():
                        if "amount" in key.lower() or "rate" in key.lower():
                            original = item[key]
                            item[key] = clean_amount(value)
                            if original != item[key] and item[key]:
                                logger.info(f" Cleaned item {i+1} {key}: {original} → {item[key]}")
            
            # Clean tax summary
            if "tax_summary_by_hsn" in data:
                for i, tax_item in enumerate(data["tax_summary_by_hsn"]):
                    for key, value in tax_item.items():
                        if "amount" in key.lower() or "rate" in key.lower():
                            original = tax_item[key]
                            tax_item[key] = clean_amount(value)
                            if original != tax_item[key] and tax_item[key]:
                                logger.info(f"Cleaned tax summary {i+1} {key}: {original} → {tax_item[key]}")
            
            logger.info("Data validation and cleaning completed")
            return data
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            return data

    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process Django uploaded image file and extract invoice data.
        Enhanced version with multi-modal analysis - maintains Django compatibility.
        delay_seconds is ignored; dynamic retry is used for rate limiting.
        """
        temp_path = None
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
                if hasattr(uploaded_file, 'chunks'):
                    for chunk in uploaded_file.chunks():
                        temp_file.write(chunk)
                else:
                    temp_file.write(uploaded_file.read())

            logger.info(f"Processing uploaded image file: {getattr(uploaded_file, 'name', 'unknown')}")
            
            # Load image for enhanced processing
            image = Image.open(temp_path)
            
            # Enhanced extraction with multiple methods
            logger.info("Starting enhanced multi-modal extraction...")
            
            # Extract plain text using advanced OCR
            plain_text = self.extract_plain_text(image)
            
            # Extract key-value pairs using bounding box analysis
            key_value_pairs = self.extract_key_value_pairs(image)
            
            # Extract table data
            table_data = self.extract_table_data(image)
            
            if not plain_text.strip():
                raise ValueError("No text could be extracted from the image")

            logger.info(f"Extracted {len(plain_text)} characters of text")
            logger.info(f"Found {len(key_value_pairs)} key-value pairs")
            logger.info(f"Found {len(table_data)} table rows")

            # Enhanced extraction using all data sources
            structured_data = self.extract_structured_data_with_gemini(
                plain_text, 
                key_value_pairs, 
                table_data, 
                image
            )
            
            # Validate and clean the data
            cleaned_data = self.validate_and_clean_data(structured_data)
            
            # Add processing metadata (maintaining compatibility)
            cleaned_data["_metadata"] = {
                "filename": getattr(uploaded_file, 'name', os.path.basename(temp_path)),
                "file_size": getattr(uploaded_file, 'size', os.path.getsize(temp_path)),
                "processed_at": datetime.now().isoformat(),
                "processing_status": "success",
                "processing_method": "Enhanced OCR + Multi-Modal Gemini AI",
                "text_length": len(plain_text),
                "extraction_methods": {
                    "plain_text_chars": len(plain_text),
                    "key_value_pairs": len(key_value_pairs),
                    "table_rows": len(table_data),
                    "ai_analysis": "multi_modal_gemini"
                }
            }
            
            logger.info(f"Successfully processed image invoice with enhanced analysis: {getattr(uploaded_file, 'name', 'unknown')}")
            return cleaned_data

        except Exception as e:
            logger.error(f"Error processing {getattr(uploaded_file, 'name', 'unknown file')}: {e}")
            raise Exception(f"Failed to process invoice image: {str(e)}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def process_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        Process invoice image from file path with enhanced analysis
        Supports both image files and PDFs
        delay_seconds is ignored; dynamic retry is used for rate limiting.
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Processing file with enhanced analysis: {file_path}")
            
            if file_path_obj.suffix.lower() == '.pdf':
                # Process PDF with enhanced analysis
                images = self.convert_pdf_to_images(str(file_path))
                if not images:
                    raise ValueError("Could not convert PDF to images")
                
                # Process all pages with enhanced extraction
                all_plain_text = ""
                all_key_value_pairs = {}
                all_table_data = []
                main_image = images[0]  # Use first page as main image for LLM
                
                for i, image in enumerate(images):
                    logger.info(f"Processing page {i+1}/{len(images)} with enhanced analysis")
                    
                    # Extract plain text
                    page_plain_text = self.extract_plain_text(image)
                    all_plain_text += f"\n--- Page {i+1} ---\n{page_plain_text}"
                    
                    # Extract key-value pairs
                    page_kv_pairs = self.extract_key_value_pairs(image)
                    for key, value in page_kv_pairs.items():
                        all_key_value_pairs[f"page_{i+1}_{key}"] = value
                    
                    # Extract table data
                    page_table_data = self.extract_table_data(image)
                    all_table_data.extend(page_table_data)
                
                plain_text = all_plain_text
                key_value_pairs = all_key_value_pairs
                table_data = all_table_data
                processing_image = main_image
                
            else:
                # Process image file with enhanced analysis
                logger.info("Processing image file with enhanced analysis")
                processing_image = Image.open(file_path)
                
                # Enhanced extraction
                plain_text = self.extract_plain_text(processing_image)
                key_value_pairs = self.extract_key_value_pairs(processing_image)
                table_data = self.extract_table_data(processing_image)

            if not plain_text.strip():
                raise ValueError("No text could be extracted from the file")

            logger.info(f"Total extracted text length: {len(plain_text)} characters")
            logger.info(f"Total key-value pairs: {len(key_value_pairs)}")
            logger.info(f"Total table rows: {len(table_data)}")

            # Enhanced extraction using all data sources
            structured_data = self.extract_structured_data_with_gemini(
                plain_text, 
                key_value_pairs, 
                table_data, 
                processing_image
            )
            
            # Validate and clean the data
            cleaned_data = self.validate_and_clean_data(structured_data)
            
            # Add processing metadata
            cleaned_data["_metadata"] = {
                "file_path": str(file_path),
                "processed_at": datetime.now().isoformat(),
                "processing_status": "success",
                "processing_method": "Enhanced OCR + Multi-Modal Gemini AI",
                "text_length": len(plain_text),
                "extraction_methods": {
                    "plain_text_chars": len(plain_text),
                    "key_value_pairs": len(key_value_pairs),
                    "table_rows": len(table_data),
                    "ai_analysis": "multi_modal_gemini"
                }
            }
            
            logger.info(f"Successfully processed file with enhanced analysis: {file_path}")
            return cleaned_data

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise Exception(f"Failed to process invoice image: {str(e)}")

    # Legacy methods for backward compatibility with existing code
    def ocr_image(self, image: Image.Image) -> str:
        """Legacy method - redirects to enhanced plain text extraction"""
        logger.info("Legacy ocr_image method called - using enhanced extraction")
        return self.extract_plain_text(image)

    def ocr_file(self, file_path: str) -> str:
        """Legacy method - extract text from file"""
        try:
            img = Image.open(file_path).convert("RGB")
            return self.extract_plain_text(img)
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            raise Exception(f"OCR failed for file: {str(e)}")

    def extract_structured_data_with_gemini_legacy(self, ocr_text: str, max_retries: int = 2) -> Dict[str, Any]:
        """Legacy method for backward compatibility - redirects to enhanced method"""
        logger.info("Legacy extraction method called - using enhanced extraction")
        
        try:
            # Create a blank image as placeholder
            blank_image = Image.new('RGB', (100, 100), color='white')
            return self.extract_structured_data_with_gemini(
                ocr_text, 
                {}, 
                [], 
                blank_image
            )
        except Exception as e:
            logger.error(f"Legacy extraction failed: {e}")
            return self.extract_with_rules(ocr_text)