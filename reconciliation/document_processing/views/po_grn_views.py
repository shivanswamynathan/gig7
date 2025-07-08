from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import logging
import tempfile
import os
from document_processing.utils.processors.data_ingestion.po_grn_extractor import PoGrnDataProcessor
from document_processing.models import UploadHistory

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class ProcessPoGrnAPI(View):
    """
    API endpoint to process PO-GRN Excel/CSV files
    """
    
    def post(self, request):
        """
        Process uploaded Excel/CSV file containing PO-GRN data
        
        Expected: multipart/form-data with 'data_file' field
        Returns: JSON response with processing results
        """
        try:
            # Check if file is provided
            if 'data_file' not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': 'No data file provided. Please upload a file with key "data_file".',
                    'status': 'error'
                }, status=400)
            
            data_file = request.FILES['data_file']
            
            # Validate file type
            allowed_extensions = ['.xlsx', '.xls', '.csv']
            file_extension = None
            for ext in allowed_extensions:
                if data_file.name.lower().endswith(ext):
                    file_extension = ext
                    break
            
            if not file_extension:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid file type. Please upload an Excel (.xlsx, .xls) or CSV (.csv) file.',
                    'status': 'error'
                }, status=400)
            
            # Validate file size (50MB limit)
            max_size = 50 * 1024 * 1024  # 50MB
            if data_file.size > max_size:
                return JsonResponse({
                    'success': False,
                    'error': f'File too large. Maximum size allowed is {max_size // (1024*1024)}MB.',
                    'status': 'error'
                }, status=400)
            
            # Save file temporarily
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    for chunk in data_file.chunks():
                        temp_file.write(chunk)
                    temp_path = temp_file.name
                
                # Initialize processor
                logger.info(f"Processing PO-GRN file: {data_file.name}")
                processor = PoGrnDataProcessor()
                
                # Process file based on extension
                if file_extension in ['.xlsx', '.xls']:
                    results = processor.process_excel_file(temp_path, data_file.name)
                else:  # CSV
                    results = processor.process_csv_file(temp_path, data_file.name)
                
                # Update file size in upload history
                upload_history = UploadHistory.objects.get(batch_id=results['batch_id'])
                upload_history.file_size = data_file.size
                upload_history.save()
                
                # Return successful response
                return JsonResponse({
                    'success': True,
                    'message': f'File processed successfully. {results["successful_records"]} records imported.',
                    'status': 'completed',
                    'data': {
                        'batch_id': results['batch_id'],
                        'total_records': results['total_records'],
                        'successful_records': results['successful_records'],
                        'failed_records': results['failed_records'],
                        'success_rate': results['success_rate'],
                        'processing_status': results['processing_status'],
                        'errors': results['errors'][:5] if results['errors'] else []  # First 5 errors
                    }
                }, status=200)
                
            finally:
                # Clean up temporary file
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            return JsonResponse({
                'success': False,
                'error': str(ve),
                'status': 'validation_error'
            }, status=400)
            
        except Exception as e:
            logger.error(f"Error processing PO-GRN file: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Failed to process file: {str(e)}',
                'status': 'processing_error'
            }, status=500)