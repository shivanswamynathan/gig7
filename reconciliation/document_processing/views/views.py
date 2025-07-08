from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import logging
from document_processing.utils.processors.invoice_processors.invoice_pdf_processor import InvoicePDFProcessor
from document_processing.utils.processors.invoice_processors.invoice_image_processor import InvoiceImageProcessor

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class ProcessInvoiceAPI(View):
    """
    API endpoint to process invoice PDF and image files
    Updated to support both PDF and image processing
    """
    
    def post(self, request):
        """
        Process uploaded PDF or image invoice file
        
        Expected: multipart/form-data with 'invoice_file' field
        Returns: JSON response with extracted invoice data
        """
        try:
            # Check if file is provided
            if 'invoice_file' not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': 'No invoice file provided. Please upload a file with key "invoice_file".',
                    'status': 'error'
                }, status=400)
            
            invoice_file = request.FILES['invoice_file']
            filename_lower = invoice_file.name.lower()
            
            # Determine file type and validate
            supported_pdf_extensions = ['.pdf']
            supported_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
            
            is_pdf = any(filename_lower.endswith(ext) for ext in supported_pdf_extensions)
            is_image = any(filename_lower.endswith(ext) for ext in supported_image_extensions)
            
            if not (is_pdf or is_image):
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid file type. Please upload a PDF file or image file. Supported formats: {", ".join(supported_pdf_extensions + supported_image_extensions)}',
                    'status': 'error'
                }, status=400)
            
            # Validate file size
            max_size = 10 * 1024 * 1024  # 10MB
            if invoice_file.size > max_size:
                return JsonResponse({
                    'success': False,
                    'error': f'File too large. Maximum size allowed is {max_size // (1024*1024)}MB.',
                    'status': 'error'
                }, status=400)
            
            # Process based on file type
            logger.info(f"Processing invoice file: {invoice_file.name}")
            
            if is_pdf:
                # Process PDF file
                processor = InvoicePDFProcessor()
                extracted_data = processor.process_uploaded_file(invoice_file)
                processing_method = "PDF Text Extraction + LLM"
                
            elif is_image:
                # Process image file
                processor = InvoiceImageProcessor()
                extracted_data = processor.process_uploaded_file(invoice_file)
                processing_method = "OCR + LLM"
            
            # Add processing method to metadata
            if '_metadata' in extracted_data:
                extracted_data['_metadata']['processing_method'] = processing_method
                extracted_data['_metadata']['file_type_detected'] = 'PDF' if is_pdf else 'Image'
            
            # Return successful response
            return JsonResponse({
                'success': True,
                'message': f'Invoice processed successfully using {processing_method}',
                'status': 'completed',
                'data': extracted_data
            }, status=200)
            
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            return JsonResponse({
                'success': False,
                'error': str(ve),
                'status': 'validation_error'
            }, status=400)
            
        except Exception as e:
            logger.error(f"Error processing invoice: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Failed to process invoice: {str(e)}',
                'status': 'processing_error'
            }, status=500)