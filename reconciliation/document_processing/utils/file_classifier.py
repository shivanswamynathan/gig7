import requests
import tempfile
import os
import logging
from PIL import Image
import fitz  # PyMuPDF
from typing import Tuple, Dict, Any
from urllib.parse import quote, unquote, urlparse, urlunparse

logger = logging.getLogger(__name__)

class SmartFileClassifier:
    """
    Smart file classifier that determines processing method needed
    Updated to include better image file support and URL encoding
    """
    
    @staticmethod
    def clean_url(url: str) -> str:
        """Clean URL by encoding problematic characters"""
        try:
            # Handle common problematic characters
            cleaned_url = url.replace('&&', '%26%26')
            cleaned_url = cleaned_url.replace(' ', '%20')
            cleaned_url = cleaned_url.replace('#', '%23')
            
            return cleaned_url
        except Exception as e:
            logger.warning(f"Error cleaning URL: {e}")
            return url
    
    @staticmethod
    def download_and_analyze(url: str) -> Dict[str, Any]:
        """
        Download file and perform complete analysis with proper URL encoding
        
        Returns:
            {
                'success': bool,
                'temp_file_path': str,
                'original_extension': str,
                'detected_format': str,
                'file_type': str,  # pdf_text, pdf_image, image, unknown
                'processing_method': str,
                'file_size': int,
                'error': str or None
            }
        """
        result = {
            'success': False,
            'temp_file_path': None,
            'original_extension': None,
            'detected_format': None,
            'file_type': 'unknown',
            'processing_method': None,
            'file_size': 0,
            'error': None
        }
        
        try:
            # Step 1: Clean and encode the URL properly
            logger.info(f"Original URL: {url}")
            
            # First try simple cleaning
            cleaned_url = SmartFileClassifier.clean_url(url)
            logger.info(f"Cleaned URL: {cleaned_url}")
            
            # If simple cleaning isn't enough, try full URL encoding
            if cleaned_url == url and ('&&' in url or ' ' in url):
                try:
                    # Parse and properly encode the URL
                    parsed = urlparse(url)
                    encoded_path = quote(parsed.path, safe='/')
                    cleaned_url = urlunparse((
                        parsed.scheme,
                        parsed.netloc,
                        encoded_path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment
                    ))
                    logger.info(f"Full encoded URL: {cleaned_url}")
                except Exception as e:
                    logger.warning(f"Full URL encoding failed: {e}, using simple cleaning")
            
            # Step 2: Download file with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            logger.info(f"Downloading file from: {cleaned_url}")
            response = requests.get(cleaned_url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            
            result['file_size'] = len(response.content)
            logger.info(f"Downloaded file size: {result['file_size']} bytes")
            
            # Step 3: Detect file format from content and headers
            content_type = response.headers.get('content-type', '').lower()
            first_bytes = response.content[:20]
            
            # File signature detection
            if first_bytes.startswith(b'%PDF'):
                detected_format = 'PDF'
                extension = '.pdf'
            elif first_bytes.startswith(b'\xff\xd8\xff'):
                detected_format = 'JPEG'
                extension = '.jpg'
            elif first_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                detected_format = 'PNG'
                extension = '.png'
            elif first_bytes.startswith(b'BM'):  # Bitmap
                detected_format = 'BMP'
                extension = '.bmp'
            elif first_bytes.startswith(b'GIF8'):  # GIF
                detected_format = 'GIF'
                extension = '.gif'
            elif first_bytes.startswith(b'\xff\xd8') and b'JFIF' in first_bytes[:20]:  # Another JPEG variant
                detected_format = 'JPEG'
                extension = '.jpg'
            elif 'pdf' in content_type:
                detected_format = 'PDF'
                extension = '.pdf'
            elif any(img_type in content_type for img_type in ['jpeg', 'jpg']):
                detected_format = 'JPEG'
                extension = '.jpg'
            elif 'png' in content_type:
                detected_format = 'PNG'
                extension = '.png'
            elif 'bmp' in content_type:
                detected_format = 'BMP'
                extension = '.bmp'
            elif 'gif' in content_type:
                detected_format = 'GIF'
                extension = '.gif'
            elif any(img_type in content_type for img_type in ['tiff', 'tif']):
                detected_format = 'TIFF'
                extension = '.tiff'
            elif 'webp' in content_type:
                detected_format = 'WEBP'
                extension = '.webp'
            else:
                # Fallback to URL extension
                url_lower = url.lower()
                if url_lower.endswith('.pdf'):
                    detected_format = 'PDF'
                    extension = '.pdf'
                elif url_lower.endswith(('.jpg', '.jpeg')):
                    detected_format = 'JPEG'
                    extension = '.jpg'
                elif url_lower.endswith('.png'):
                    detected_format = 'PNG'
                    extension = '.png'
                elif url_lower.endswith('.bmp'):
                    detected_format = 'BMP'
                    extension = '.bmp'
                elif url_lower.endswith('.gif'):
                    detected_format = 'GIF'
                    extension = '.gif'
                elif url_lower.endswith(('.tiff', '.tif')):
                    detected_format = 'TIFF'
                    extension = '.tiff'
                elif url_lower.endswith('.webp'):
                    detected_format = 'WEBP'
                    extension = '.webp'
                else:
                    raise ValueError(f"Unsupported file format: {url}")
            
            result['detected_format'] = detected_format
            result['original_extension'] = extension
            
            # Step 4: Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(response.content)
                result['temp_file_path'] = temp_file.name
            
            logger.info(f"Saved to temporary file: {result['temp_file_path']}")
            
            # Step 5: Analyze content to determine processing method
            if detected_format == 'PDF':
                # Check if PDF has extractable text
                can_extract_text = SmartFileClassifier._analyze_pdf_content(result['temp_file_path'])
                
                if can_extract_text:
                    result['file_type'] = 'pdf_text'
                    result['processing_method'] = 'Direct text extraction + LLM'
                else:
                    result['file_type'] = 'pdf_image'
                    result['processing_method'] = 'PDF → Images → OCR → LLM'
                    
            elif detected_format in ['JPEG', 'PNG', 'BMP', 'GIF', 'TIFF', 'WEBP']:
                # Verify image is valid and processable
                if SmartFileClassifier._verify_image(result['temp_file_path']):
                    result['file_type'] = 'image'
                    result['processing_method'] = 'OCR → LLM'
                else:
                    raise ValueError("Invalid or unprocessable image file")
            
            result['success'] = True
            logger.info(f"File classified as: {result['file_type']} ({result['detected_format']})")
            
        except Exception as e:
            error_msg = f"Error analyzing file {url}: {str(e)}"
            logger.error(error_msg)
            result['error'] = error_msg
            
            # Clean up temp file if created
            if result['temp_file_path'] and os.path.exists(result['temp_file_path']):
                try:
                    os.unlink(result['temp_file_path'])
                except:
                    pass
                result['temp_file_path'] = None
        
        return result
    
    @staticmethod
    def _analyze_pdf_content(pdf_path: str) -> bool:
        """
        Analyze PDF to determine if it has extractable text
        
        Returns:
            True if text can be extracted, False if OCR is needed
        """
        try:
            doc = fitz.open(pdf_path)
            total_text_length = 0
            pages_to_check = min(3, len(doc))  # Check first 3 pages
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text().strip()
                total_text_length += len(text)
            
            doc.close()
            
            # Decision threshold: if average > 100 chars per page, consider it text-based
            avg_text_per_page = total_text_length / pages_to_check if pages_to_check > 0 else 0
            has_extractable_text = avg_text_per_page > 100
            
            logger.info(f"PDF analysis: {total_text_length} chars in {pages_to_check} pages, "
                       f"avg {avg_text_per_page:.1f} per page, text-extractable: {has_extractable_text}")
            
            return has_extractable_text
            
        except Exception as e:
            logger.error(f"Error analyzing PDF content: {str(e)}")
            return False  # Assume OCR needed if analysis fails
    
    @staticmethod
    def _verify_image(image_path: str) -> bool:
        """
        Verify that image file is valid and openable
        Also check if it's suitable for OCR processing
        """
        try:
            with Image.open(image_path) as img:
                # Verify the image can be opened
                img.verify()
            
            # Re-open for additional checks (verify() closes the image)
            with Image.open(image_path) as img:
                # Check image dimensions (minimum size for OCR)
                width, height = img.size
                if width < 100 or height < 100:
                    logger.warning(f"Image too small for OCR: {width}x{height}")
                    return False
                
                # Check if image is too large (memory concerns)
                if width * height > 50_000_000:  # 50 megapixels
                    logger.warning(f"Image very large: {width}x{height}, may need resizing")
                
                # Convert to RGB if necessary (for better OCR processing)
                if img.mode not in ['RGB', 'L']:
                    logger.info(f"Image mode {img.mode} will be converted for OCR")
                
                logger.info(f"Image verified: {width}x{height}, mode: {img.mode}")
                return True
                
        except Exception as e:
            logger.error(f"Image verification failed: {str(e)}")
            return False
    
    @staticmethod
    def cleanup_temp_file(temp_path: str):
        """Clean up temporary file"""
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_path}: {str(e)}")