import asyncio
import logging
from typing import Dict, Any, List
from django.db import transaction
from asgiref.sync import sync_to_async
from django.db import models
from datetime import datetime
from decimal import Decimal, InvalidOperation

from document_processing.models import ItemWiseGrn, InvoiceData, InvoiceItemData
from document_processing.utils.file_classifier import SmartFileClassifier
from document_processing.utils.processors.invoice_processors.invoice_pdf_processor import InvoicePDFProcessor
from document_processing.utils.processors.invoice_processors.invoice_image_processor import InvoiceImageProcessor
from document_processing.utils.failure_reason_classifier import classify_failure_reason

logger = logging.getLogger(__name__)

class AttachmentProcessorFromGrn:
    """
    FIXED VERSION: Actually uses its own processors instead of creating new ones
    """
    def __init__(self, max_concurrent_requests: int = 10):
        # Initialize processors ONCE and reuse them
        self.file_classifier = SmartFileClassifier()
        self.text_pdf_processor = InvoicePDFProcessor()
        self.image_processor = InvoiceImageProcessor()
        self.max_concurrent_requests = max_concurrent_requests

    async def process_from_grn_table(self, process_limit: int = 10, force_reprocess: bool = False) -> Dict[str, Any]:
        try:
            # Step 1: Extract attachments from ItemWiseGrn table
            attachment_data = await sync_to_async(self._extract_attachments_from_grn_table)()
            if not attachment_data:
                return {
                    'success': False,
                    'error': 'No attachment URLs found in ItemWiseGrn table.',
                    'total_attachments_found': 0,
                    'processed_attachments': 0,
                    'successful_extractions': 0,
                    'failed_extractions': 0,
                    'results': []
                }
            
            # If process_limit is None or not set, process all attachments
            if process_limit is None:
                attachments_to_process = attachment_data
            else:
                attachments_to_process = attachment_data[:process_limit]
            
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            tasks = [
                self._process_attachment_direct(attachment_info, force_reprocess, semaphore)
                for attachment_info in attachments_to_process
            ]
            
            logger.info(f"Starting concurrent processing of {len(tasks)} attachments from GRN table (max {self.max_concurrent_requests} concurrent)")
            processing_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            final_results = []
            successful_extractions = 0
            failed_extractions = 0
            
            for i, result in enumerate(processing_results):
                if isinstance(result, Exception):
                    logger.error(f"Attachment {i} failed: {str(result)}")
                    final_results.append({
                        'url': attachments_to_process[i]['url'][:50] + '...',
                        'po_number': attachments_to_process[i]['po_number'],
                        'grn_number': attachments_to_process[i].get('grn_number', 'N/A'),
                        'success': False,
                        'error': str(result)
                    })
                    failed_extractions += 1
                else:
                    final_results.append(result)
                    if result['success']:
                        successful_extractions += 1
                    else:
                        failed_extractions += 1
            
            return {
                'success': True,
                'total_attachments_found': len(attachment_data),
                'processed_attachments': len(attachments_to_process),
                'successful_extractions': successful_extractions,
                'failed_extractions': failed_extractions,
                'success_rate': f"{successful_extractions}/{len(attachments_to_process)}",
                'results': final_results
            }
        except Exception as e:
            logger.error(f"Error processing from GRN table: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_attachments_found': 0,
                'processed_attachments': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'results': []
            }

    def _extract_attachments_from_grn_table(self) -> List[Dict[str, Any]]:
        """
        Extract attachment URLs and related info from ItemWiseGrn table.
        Only include records where extracted_data is False.
        """
        attachments = []
        grn_records = ItemWiseGrn.objects.filter(extracted_data=False)
        
        for grn in grn_records:
            po_number = getattr(grn, 'po_number', None) or getattr(grn, 'po_no', None)
            grn_number = getattr(grn, 'grn_number', None) or getattr(grn, 'grn_no', None)
            supplier = getattr(grn, 'supplier', None) or getattr(grn, 'vendor', 'Unknown')
            
            # Check all attachment fields
            for i in range(1, 6):
                url = getattr(grn, f'attachment_{i}', None)
                if url and str(url).strip().startswith(('http://', 'https://')):
                    attachments.append({
                        'url': str(url).strip(),
                        'po_number': str(po_number) if po_number else 'N/A',
                        'grn_number': str(grn_number) if grn_number else 'N/A',
                        'supplier': str(supplier) if supplier else 'Unknown',
                        'attachment_number': i,
                        'row_number': grn.pk
                    })
        
        # Deduplicate by URL
        unique_attachments = []
        seen_urls = set()
        for att in attachments:
            if att['url'] not in seen_urls:
                unique_attachments.append(att)
                seen_urls.add(att['url'])
        
        return unique_attachments

    async def _process_attachment_direct(self, attachment_info: Dict[str, Any], force_reprocess: bool = False, semaphore: asyncio.Semaphore = None) -> Dict[str, Any]:
        """
        FIXED: Actually uses self.file_classifier instead of creating new processor
        """
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)
            
        url = attachment_info['url']
        
        async with semaphore:
            try:
                logger.info(f"Processing attachment: {url[:50]}...")
                
                # Check if already processed
                if not force_reprocess:
                    existing = await sync_to_async(InvoiceData.objects.filter(attachment_url=url).first)()
                    if existing:
                        logger.info(f"Attachment already processed, skipping: {url}")
                        return {
                            'url': url[:50] + '...',
                            'po_number': attachment_info['po_number'],
                            'grn_number': attachment_info.get('grn_number', 'N/A'),
                            'supplier': attachment_info.get('supplier', 'N/A'),
                            'success': True,
                            'status': 'already_processed',
                            'invoice_id': existing.id,
                            'vendor_name': existing.vendor_name,
                            'invoice_number': existing.invoice_number
                        }
                
                # Step 1: Download and classify file using OUR classifier
                classification = await self._download_and_analyze_async(url)
                
                if not classification['success']:
                    await self._save_error_record_direct_async(attachment_info, classification['error'], 'unknown', None)
                    return {
                        'url': url[:50] + '...',
                        'po_number': attachment_info['po_number'],
                        'grn_number': attachment_info.get('grn_number', 'N/A'),
                        'supplier': attachment_info.get('supplier', 'N/A'),
                        'success': False,
                        'error': classification['error'],
                        'file_type': 'unknown'
                    }
                
                temp_file_path = classification['temp_file_path']
                
                try:
                    # Step 2: Process based on file type using OUR processors
                    file_type = classification['file_type']
                    
                    if file_type == 'pdf_text':
                        # Use OUR PDF processor
                        extracted_data = await self._process_pdf_text_async(temp_file_path)
                        
                    elif file_type == 'pdf_image':
                        # Use OUR image processor for PDF images
                        extracted_data = await self._process_pdf_image_async(temp_file_path)
                        
                    elif file_type == 'image':
                        # Use OUR image processor
                        extracted_data = await self._process_image_async(temp_file_path)
                        
                    else:
                        raise ValueError(f"Unsupported file type: {file_type}")
                    
                    # Step 3: Save to database
                    invoice_record = await self._save_extracted_data_direct_async(attachment_info, classification, extracted_data)

                    # Mark GRN records as processed
                    await self._mark_grn_records_as_processed(url, attachment_info.get('row_number'))

                    return {
                        'url': url[:50] + '...',
                        'po_number': attachment_info['po_number'],
                        'grn_number': attachment_info.get('grn_number', 'N/A'),
                        'supplier': attachment_info.get('supplier', 'N/A'),
                        'success': True,
                        'file_type': file_type,
                        'processing_method': classification['processing_method'],
                        'invoice_id': invoice_record.id,
                        'vendor_name': self._safe_get_vendor_name(extracted_data),
                        'invoice_number': self._safe_get_invoice_number(extracted_data),
                        'invoice_total': self._safe_get_invoice_total(extracted_data)
                    }
                    
                finally:
                    # Clean up temp file
                    if temp_file_path:
                        await self._cleanup_temp_file_async(temp_file_path)
            
            except Exception as e:
                # Save error record
                await self._save_error_record_direct_async(attachment_info, str(e), 'unknown', None)
                logger.error(f"Error processing attachment {url}: {str(e)}")
                return {
                    'url': url[:50] + '...',
                    'po_number': attachment_info['po_number'],
                    'success': False,
                    'error': str(e)
                }

    # ========== ASYNC WRAPPER METHODS FOR OUR PROCESSORS ==========
    
    async def _download_and_analyze_async(self, url: str) -> Dict[str, Any]:
        """Use OUR file classifier"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.file_classifier.download_and_analyze, url)
    
    async def _process_pdf_text_async(self, temp_file_path: str) -> Dict[str, Any]:
        """Use OUR PDF processor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.text_pdf_processor.process_file_path, temp_file_path)
    
    async def _process_pdf_image_async(self, temp_file_path: str) -> Dict[str, Any]:
        """Use OUR image processor for PDF images"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.image_processor.process_file_path, temp_file_path)
    
    async def _process_image_async(self, temp_file_path: str) -> Dict[str, Any]:
        """Use OUR image processor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.image_processor.process_file_path, temp_file_path)
    
    async def _cleanup_temp_file_async(self, temp_path: str):
        """Clean up temp file"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.file_classifier.cleanup_temp_file, temp_path)

    # ========== HELPER METHODS ==========
    
    def _safe_get_vendor_name(self, extracted_data: Dict[str, Any]) -> str:
        try:
            vendor_details = extracted_data.get('vendor_details', {})
            return vendor_details.get('vendor_name', '') if isinstance(vendor_details, dict) else ''
        except:
            return ''
    
    def _safe_get_invoice_number(self, extracted_data: Dict[str, Any]) -> str:
        try:
            invoice_info = extracted_data.get('invoice_info', {})
            return invoice_info.get('invoice_number', '') if isinstance(invoice_info, dict) else ''
        except:
            return ''
    
    def _safe_get_invoice_total(self, extracted_data: Dict[str, Any]) -> str:
        try:
            invoice_totals = extracted_data.get('invoice_totals', {})
            return str(invoice_totals.get('final_invoice_amount', '')) if isinstance(invoice_totals, dict) else ''
        except:
            return ''

    async def _mark_grn_records_as_processed(self, url: str, row_number: int = None):
        """Mark GRN records as processed"""
        try:
            # Mark specific row if row_number provided
            if row_number:
                grn_obj = await sync_to_async(ItemWiseGrn.objects.filter(pk=row_number).first)()
                if grn_obj and not grn_obj.extracted_data:
                    grn_obj.extracted_data = True
                    await sync_to_async(grn_obj.save)(update_fields=["extracted_data"])
                    logger.info(f"Marked ItemWiseGrn(pk={row_number}) as extracted.")
            
            # Mark all records with this URL
            updated_count = await sync_to_async(ItemWiseGrn.objects.filter(
                models.Q(attachment_1=url) |
                models.Q(attachment_2=url) |
                models.Q(attachment_3=url) |
                models.Q(attachment_4=url) |
                models.Q(attachment_5=url),
                extracted_data=False
            ).update(extracted_data=True))
            
            if updated_count > 0:
                logger.info(f"Marked extracted_data=True for {updated_count} ItemWiseGrn records with URL {url}.")
                
        except Exception as e:
            logger.warning(f"Could not mark GRN records as processed for URL {url}: {e}")

    async def _save_extracted_data_direct_async(self, attachment_info: Dict[str, Any], classification: Dict[str, Any], extracted_data: Dict[str, Any]) -> InvoiceData:
        """Save extracted data to database"""
        return await sync_to_async(self._save_extracted_data_direct)(attachment_info, classification, extracted_data)
    
    async def _save_error_record_direct_async(self, attachment_info: Dict[str, Any], error_message: str, file_type: str, original_extension: str):
        """Save error record to database"""
        await sync_to_async(self._save_error_record_direct)(attachment_info, error_message, file_type, original_extension)

    def _save_extracted_data_direct(self, attachment_info: Dict[str, Any], classification: Dict[str, Any], extracted_data: Dict[str, Any]) -> InvoiceData:
        """Save extracted data to database (sync version)"""
        # Implementation same as in SimplifiedAttachmentProcessor
        with transaction.atomic():
            vendor_details = extracted_data.get('vendor_details', {})
            vendor_name = vendor_details.get('vendor_name', '') if isinstance(vendor_details, dict) else ''
            vendor_gst = vendor_details.get('vendor_gst', '') if isinstance(vendor_details, dict) else ''
            vendor_pan = vendor_details.get('vendor_pan', '') if isinstance(vendor_details, dict) else ''

            invoice_info = extracted_data.get('invoice_info', {})
            invoice_number = invoice_info.get('invoice_number', '') if isinstance(invoice_info, dict) else ''
            invoice_date_str = invoice_info.get('invoice_date', '') if isinstance(invoice_info, dict) else ''

            invoice_data = InvoiceData(
                attachment_number=str(attachment_info['attachment_number']),
                attachment_url=attachment_info['url'],
                file_type=classification['file_type'],
                original_file_extension=classification['original_extension'],
                po_number=attachment_info['po_number'],
                grn_number=attachment_info.get('grn_number', None),
                vendor_name=vendor_name,
                vendor_pan=vendor_pan,
                vendor_gst=vendor_gst,
                invoice_number=invoice_number,
                processing_status='completed',
                extracted_at=datetime.now()
            )
            
            # Parse date
            if invoice_date_str:
                try:
                    if '/' in invoice_date_str:
                        invoice_data.invoice_date = datetime.strptime(invoice_date_str, '%d/%m/%Y').date()
                    elif '-' in invoice_date_str:
                        invoice_data.invoice_date = datetime.strptime(invoice_date_str, '%d-%m-%Y').date()
                except ValueError:
                    logger.warning(f"Could not parse date: {invoice_date_str}")
            
            # Parse financial fields
            invoice_totals = extracted_data.get('invoice_totals', {})
            if isinstance(invoice_totals, dict):
                financial_fields = {
                    'invoice_value_without_gst': invoice_totals.get('total_taxable_amount'),
                    'invoice_total_post_gst': invoice_totals.get('final_invoice_amount'),
                    'cgst_amount': invoice_totals.get('total_cgst'),
                    'sgst_amount': invoice_totals.get('total_sgst'),
                    'igst_amount': invoice_totals.get('total_igst'),
                    'total_gst_amount': invoice_totals.get('total_gst')
                }
                
                for field, value in financial_fields.items():
                    if value:
                        try:
                            clean_value = str(value).replace(',', '').replace('₹', '').strip()
                            if clean_value:
                                setattr(invoice_data, field, Decimal(clean_value))
                        except (InvalidOperation, ValueError):
                            logger.warning(f"Could not parse {field}: {value}")
            
            # Set failure reason
            total_gst_val = invoice_totals.get('total_gst') if isinstance(invoice_totals, dict) else None
            invoice_total_val = invoice_totals.get('final_invoice_amount') if isinstance(invoice_totals, dict) else None
            failure_reason = classify_failure_reason(
                classification['file_type'],
                invoice_number,
                vendor_gst,
                total_gst_val,
                invoice_total_val
            )
            invoice_data.failure_reason = failure_reason
            
            invoice_data.save()
            
            # Create line items
            line_items = extracted_data.get('line_items', [])
            if line_items and isinstance(line_items, list):
                self._create_invoice_items(invoice_data, line_items, attachment_info)

            logger.info(f"Saved invoice data for PO {attachment_info['po_number']}, attachment {attachment_info['attachment_number']}")
            return invoice_data

    def _create_invoice_items(self, invoice_data: InvoiceData, line_items: List[Dict[str, Any]], attachment_info: Dict[str, Any]):
        """Create invoice item records"""
        # Implementation same as in SimplifiedAttachmentProcessor
        try:
            items_to_create = []
            
            for idx, item in enumerate(line_items, 1):
                if not isinstance(item, dict):
                    continue
                
                item_record = InvoiceItemData(
                    invoice_data_id=invoice_data.id,
                    item_description=item.get('item_description', ''),
                    hsn_code=item.get('hsn_sac_code', ''),
                    unit_of_measurement=item.get('unit', ''),
                    item_sequence=idx,
                    po_number=attachment_info['po_number'],
                    invoice_number=invoice_data.invoice_number,
                    vendor_name=invoice_data.vendor_name
                )
                
                # Parse numeric fields
                numeric_fields = {
                    'quantity': 'quantity',
                    'unit_price': 'rate_per_unit',
                    'invoice_value_item_wise': 'final_amount_including_gst',
                    'cgst_rate': 'cgst_rate',
                    'cgst_amount': 'cgst_amount',
                    'sgst_rate': 'sgst_rate',
                    'sgst_amount': 'sgst_amount',
                    'igst_rate': 'igst_rate',
                    'igst_amount': 'igst_amount',
                    'total_tax_amount': 'total_gst_on_item',
                    'item_total_amount': 'final_amount_including_gst'
                }
                
                for field, source_field in numeric_fields.items():
                    value = item.get(source_field)
                    if value:
                        try:
                            clean_value = str(value).replace(',', '').replace('\u20b9', '').replace('%', '').strip()
                            if clean_value:
                                setattr(item_record, field, Decimal(clean_value))
                        except (InvalidOperation, ValueError):
                            logger.warning(f"Could not parse {field} for item {idx}: {value}")
                
                items_to_create.append(item_record)
            
            # Bulk create
            if items_to_create:
                InvoiceItemData.objects.bulk_create(items_to_create)
                logger.info(f"Created {len(items_to_create)} item records for invoice {invoice_data.invoice_number}")
        
        except Exception as e:
            logger.error(f"Error creating invoice items: {str(e)}")

    def _save_error_record_direct(self, attachment_info: Dict[str, Any], error_message: str, file_type: str, original_extension: str):
        """Save error record to database (sync version)"""
        try:
            with transaction.atomic():
                invoice_data = InvoiceData(
                    attachment_number=str(attachment_info['attachment_number']),
                    attachment_url=attachment_info['url'],
                    file_type=file_type or 'unknown',
                    original_file_extension=original_extension,
                    po_number=attachment_info['po_number'],
                    grn_number=attachment_info.get('grn_number', None),
                    processing_status='failed',
                    error_message=error_message,
                    extracted_at=datetime.now()
                )
                
                failure_reason = classify_failure_reason(
                    file_type,
                    attachment_info.get('invoice_number', ''),
                    attachment_info.get('vendor_gst', ''),
                    '',
                    ''
                )
                invoice_data.failure_reason = failure_reason
                
                invoice_data.save()
                logger.info(f"Saved error record for PO {attachment_info['po_number']}, attachment {attachment_info['attachment_number']}")
                
        except Exception as e:
            logger.error(f"Error saving error record: {str(e)}")