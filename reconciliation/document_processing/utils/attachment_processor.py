import asyncio
import aiohttp
import logging
import pandas as pd
from typing import Dict, Any, List
from django.db import transaction
from decimal import Decimal, InvalidOperation
from datetime import datetime
import tempfile
import os
from asgiref.sync import sync_to_async
from django.db import models

from document_processing.models import ItemWiseGrn, InvoiceData, InvoiceItemData
from document_processing.utils.file_classifier import SmartFileClassifier
from document_processing.utils.processors.invoice_processors.invoice_pdf_processor import InvoicePDFProcessor
from document_processing.utils.processors.invoice_processors.invoice_image_processor import InvoiceImageProcessor
from document_processing.utils.failure_reason_classifier import classify_failure_reason

logger = logging.getLogger(__name__)

class SimplifiedAttachmentProcessor:
    """
    Direct async-only SimplifiedAttachmentProcessor with concurrent processing
    Updated to include image processing capabilities
    """
    
    def __init__(self, max_concurrent_requests: int = 10):
        # Use the separate file classifier
        self.file_classifier = SmartFileClassifier()
        self.text_pdf_processor = InvoicePDFProcessor()
        self.image_processor = InvoiceImageProcessor()  # NEW: Image processor
        self.max_concurrent_requests = max_concurrent_requests
        
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.errors = []
    
    async def process_from_excel_file(self, file_path: str, file_extension: str, process_limit: int = 10, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process attachments directly from Excel file - DIRECT ASYNC
        """
        try:
            # Step 1: Extract attachments from file
            attachment_data = self._extract_attachments_from_file(file_path, file_extension)
            
            if not attachment_data:
                return {
                    'success': False,
                    'error': 'No attachment URLs found in the uploaded file.',
                    'total_attachments_found': 0,
                    'processed_attachments': 0,
                    'successful_extractions': 0,
                    'failed_extractions': 0,
                    'results': []
                }
            
            # Step 2: Process attachments CONCURRENTLY
            # If process_limit is None or not set, process all attachments
            if process_limit is None:
                attachments_to_process = attachment_data
            else:
                attachments_to_process = attachment_data[:process_limit]
            
            # Concurrent processing with semaphore
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
            # Create async tasks for each attachment
            tasks = [
                self._process_attachment_direct(attachment_info, force_reprocess, semaphore)
                for attachment_info in attachments_to_process
            ]
            
            logger.info(f"Starting concurrent processing of {len(tasks)} attachments (max {self.max_concurrent_requests} concurrent)")
            
            # Execute all tasks concurrently
            processing_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions and format results
            final_results = []
            successful_extractions = 0
            failed_extractions = 0
            
            for i, result in enumerate(processing_results):
                if isinstance(result, Exception):
                    logger.error(f"Attachment {i} failed: {str(result)}")
                    final_results.append({
                        'url': attachments_to_process[i]['url'][:50] + '...',
                        'po_number': attachments_to_process[i]['po_number'],
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
            logger.error(f"Error processing Excel file: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_attachments_found': 0,
                'processed_attachments': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'results': []
            }
    
    async def _process_attachment_direct(self, attachment_info: Dict[str, Any], force_reprocess: bool = False, semaphore: asyncio.Semaphore = None) -> Dict[str, Any]:
        """
        Process a single attachment directly from Excel data
        UPDATED to support image processing
        """
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)
            
        url = attachment_info['url']
        
        async with semaphore:  # Limit concurrent requests
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
                
                # Step 1: Download and classify file
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
                    # Step 2: Process based on file type
                    file_type = classification['file_type']
                    
                    if file_type == 'pdf_text':
                        # Async LLM processing
                        extracted_data = await self._process_pdf_text_async(temp_file_path)
                        
                    elif file_type == 'pdf_image':
                        # NEW: Process image-based PDF using OCR
                        extracted_data = await self._process_pdf_image_async(temp_file_path)
                        
                    elif file_type == 'image':
                        # NEW: Process image files using OCR
                        extracted_data = await self._process_image_async(temp_file_path)
                        
                    else:
                        raise ValueError(f"Unsupported file type: {file_type}")
                    
                    # Step 3: Save to database
                    invoice_record = await self._save_extracted_data_direct_async(attachment_info, classification, extracted_data)

                    # === Mark ItemWiseGrn.extracted_data = True if row_number is present ===
                    row_number = attachment_info.get('row_number')
                    if row_number:
                        try:
                            grn_obj = await sync_to_async(ItemWiseGrn.objects.filter(pk=row_number).first)()
                            if grn_obj and not grn_obj.extracted_data:
                                grn_obj.extracted_data = True
                                await sync_to_async(grn_obj.save)(update_fields=["extracted_data"])
                                logger.info(f"Marked ItemWiseGrn(pk={row_number}) as extracted.")
                        except Exception as e:
                            logger.warning(f"Could not mark ItemWiseGrn(pk={row_number}) as extracted: {e}")
                    # === End mark ===

                    # === Mark ItemWiseGrn.extracted_data = True for all records with this URL ===
                    url_to_mark = url
                    try:
                        updated_count = await sync_to_async(ItemWiseGrn.objects.filter(
                            models.Q(attachment_1=url_to_mark) |
                            models.Q(attachment_2=url_to_mark) |
                            models.Q(attachment_3=url_to_mark) |
                            models.Q(attachment_4=url_to_mark) |
                            models.Q(attachment_5=url_to_mark),
                            extracted_data=False
                        ).update(extracted_data=True))
                        logger.info(f"Marked extracted_data=True for {updated_count} ItemWiseGrn records with URL {url_to_mark}.")
                    except Exception as e:
                        logger.warning(f"Could not mark all ItemWiseGrn records with URL {url_to_mark} as extracted: {e}")
                    # === End mark ===

                    return {
                        'url': url[:50] + '...',
                        'po_number': attachment_info['po_number'],
                        'grn_number': attachment_info.get('grn_number', 'N/A'),
                        'supplier': attachment_info.get('supplier', 'N/A'),
                        'success': True,
                        'file_type': file_type,
                        'processing_method': classification['processing_method'],
                        'invoice_id': invoice_record.id,
                        'vendor_name': extracted_data.get('vendor_details', {}).get('vendor_name', '') if isinstance(extracted_data.get('vendor_details'), dict) else '',
                        'invoice_number': extracted_data.get('invoice_info', {}).get('invoice_number', '') if isinstance(extracted_data.get('invoice_info'), dict) else '',
                        'invoice_total': str(extracted_data.get('invoice_totals', {}).get('final_invoice_amount', '')) if isinstance(extracted_data.get('invoice_totals'), dict) else ''
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
    
    # NEW: Async methods for image processing
    async def _process_pdf_image_async(self, temp_file_path: str) -> Dict[str, Any]:
        """Async version of PDF image processing with OCR + LLM"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.image_processor.process_file_path, temp_file_path)
    
    async def _process_image_async(self, temp_file_path: str) -> Dict[str, Any]:
        """Async version of image processing with OCR + LLM"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.image_processor.process_file_path, temp_file_path)
    
    # Existing async helper methods
    async def _download_and_analyze_async(self, url: str) -> Dict[str, Any]:
        """Async version of file download and analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.file_classifier.download_and_analyze, url)
    
    async def _process_pdf_text_async(self, temp_file_path: str) -> Dict[str, Any]:
        """Async version of PDF text processing with LLM"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.text_pdf_processor.process_file_path, temp_file_path)
    
    async def _cleanup_temp_file_async(self, temp_path: str):
        """Async temp file cleanup"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.file_classifier.cleanup_temp_file, temp_path)
    
    async def _save_extracted_data_direct_async(self, attachment_info: Dict[str, Any], classification: Dict[str, Any], extracted_data: Dict[str, Any]) -> InvoiceData:
        """Async version of save extracted data"""
        return await sync_to_async(self._save_extracted_data_direct)(attachment_info, classification, extracted_data)
    
    async def _save_error_record_direct_async(self, attachment_info: Dict[str, Any], error_message: str, file_type: str, original_extension: str):
        """Async version of save error record"""
        await sync_to_async(self._save_error_record_direct)(attachment_info, error_message, file_type, original_extension)
    
    def _extract_attachments_from_file(self, file_path: str, file_extension: str) -> List[Dict[str, Any]]:
        """
        Extract ALL attachment URLs directly from uploaded file
        """
        try:
            # Read file into pandas DataFrame
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, header=0)
            else:  # CSV
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise Exception("Could not read CSV file with any supported encoding")
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            logger.info(f"File loaded: {len(df)} rows, columns: {list(df.columns)}")
            
            # Normalize column names (case-insensitive mapping)
            column_mapping = {
                'po no.': 'po_no',
                'po no': 'po_no',
                'po number': 'po_no',
                'grn no.': 'grn_no',
                'grn no': 'grn_no',
                'grn number': 'grn_no',
                'supplier': 'supplier',
                'vendor': 'supplier',
                'attachment-1': 'attachment_1',
                'attachment-2': 'attachment_2',
                'attachment-3': 'attachment_3',
                'attachment-4': 'attachment_4',
                'attachment-5': 'attachment_5',
            }
            
            # Create case-insensitive column mapping
            normalized_columns = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in column_mapping:
                    normalized_columns[col] = column_mapping[col_lower]
            
            if not normalized_columns:
                logger.warning(f"No matching columns found. Available columns: {list(df.columns)}")
                return []
            
            # Rename columns
            df = df.rename(columns=normalized_columns)
            
            # Extract ALL attachment URLs directly
            all_attachments = []
            
            for row_idx, row in df.iterrows():
                po_no = row.get('po_no')
                grn_no = row.get('grn_no', 'N/A')
                supplier = row.get('supplier', 'Unknown')
                
                if pd.isna(po_no) or not po_no:
                    continue
                
                po_no = str(po_no).strip()
                grn_no = str(grn_no).strip() if pd.notna(grn_no) else 'N/A'
                supplier = str(supplier).strip() if pd.notna(supplier) else 'Unknown'
                
                # Extract attachment URLs from this row
                for i in range(1, 6):
                    attachment_col = f'attachment_{i}'
                    if attachment_col in row:
                        url = row[attachment_col]
                        if pd.notna(url) and url and str(url).strip():
                            clean_url = str(url).strip()
                            if clean_url.startswith(('http://', 'https://')):
                                all_attachments.append({
                                    'url': clean_url,
                                    'po_number': po_no,
                                    'grn_number': grn_no,
                                    'supplier': supplier,
                                    'attachment_number': i,
                                    'row_number': row_idx + 1
                                })
            
            # Remove duplicates based on URL
            unique_attachments = []
            seen_urls = set()
            
            for attachment in all_attachments:
                if attachment['url'] not in seen_urls:
                    unique_attachments.append(attachment)
                    seen_urls.add(attachment['url'])
            
            logger.info(f"Extracted {len(unique_attachments)} unique attachments from {len(all_attachments)} total")
            
            return unique_attachments
            
        except Exception as e:
            logger.error(f"Error extracting attachments from file: {str(e)}")
            raise Exception(f"Failed to extract data from file: {str(e)}")
    
    def _save_extracted_data_direct(self, attachment_info: Dict[str, Any], classification: Dict[str, Any], extracted_data: Dict[str, Any]) -> InvoiceData:
        """
        Save extracted data directly (without GRN reference)
        """
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
                
                # Basic info from Excel
                po_number=attachment_info['po_number'],
                grn_number=attachment_info.get('grn_number', None),
                # Extracted invoice data
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
            
            # Parse financial fields - MAP FROM NEW NESTED SCHEMA
            invoice_totals = extracted_data.get('invoice_totals', {})
            if isinstance(invoice_totals, dict):
                financial_fields = {
                    'invoice_value_without_gst': invoice_totals.get('total_taxable_amount'),  # total_taxable_amount
                    'invoice_total_post_gst': invoice_totals.get('final_invoice_amount'),    # final_invoice_amount
                    'cgst_amount': invoice_totals.get('total_cgst'),                         # total_cgst
                    'sgst_amount': invoice_totals.get('total_sgst'),                         # total_sgst  
                    'igst_amount': invoice_totals.get('total_igst'),                         # total_igst
                    'total_gst_amount': invoice_totals.get('total_gst')                      # total_gst
                }
            else:
                financial_fields = {}
            
            # Convert to Decimal
            for field, value in financial_fields.items():
                if value:
                    try:
                        clean_value = str(value).replace(',', '').replace('₹', '').strip()
                        if clean_value:
                            setattr(invoice_data, field, Decimal(clean_value))
                    except (InvalidOperation, ValueError):
                        logger.warning(f"Could not parse {field}: {value}")
            
            # --- Failure Reason Logic ---
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
            # --- End Failure Reason Logic ---
            
            invoice_data.save()

            # Extract line items from new schema structure
            line_items = extracted_data.get('line_items', [])
            if line_items and isinstance(line_items, list):
                self._create_invoice_items(invoice_data, line_items, attachment_info)

            logger.info(f"Saved invoice data for PO {attachment_info['po_number']}, attachment {attachment_info['attachment_number']}")
            return invoice_data
    
    def _create_invoice_items(self, invoice_data: InvoiceData, line_items: List[Dict[str, Any]], attachment_info: Dict[str, Any]):
        """
        Create separate InvoiceItemData records for each item - UPDATED FOR NEW SCHEMA v3.0
        """
        try:
            items_to_create = []
            
            for idx, item in enumerate(line_items, 1):
                if not isinstance(item, dict):
                    continue
                
                item_record = InvoiceItemData(
                    invoice_data_id = invoice_data.id,
                    # MAP NEW SCHEMA FIELDS TO DATABASE FIELDS
                    item_description=item.get('item_description', ''),          
                    hsn_code=item.get('hsn_sac_code', ''),                      
                    unit_of_measurement=item.get('unit', ''),                   
                    item_sequence=idx,
                    
                    # Reference fields for easy querying
                    po_number=attachment_info['po_number'],
                    invoice_number=invoice_data.invoice_number,
                    vendor_name=invoice_data.vendor_name
                )
                
                # Parse quantity
                quantity_str = item.get('quantity', '')
                if quantity_str:
                    try:
                        clean_qty = str(quantity_str).replace(',', '').strip()
                        if clean_qty:
                            item_record.quantity = Decimal(clean_qty)
                    except (InvalidOperation, ValueError):
                        logger.warning(f"Could not parse quantity: {quantity_str}")
                
                # Parse unit price - rate_per_unit in new schema
                unit_price_str = item.get('rate_per_unit', '')
                if unit_price_str:
                    try:
                        clean_price = str(unit_price_str).replace(',', '').replace('₹', '').strip()
                        if clean_price:
                            item_record.unit_price = Decimal(clean_price)
                    except (InvalidOperation, ValueError):
                        logger.warning(f"Could not parse unit price: {unit_price_str}")
                
                # Parse item-wise invoice value - final_amount_including_gst in new schema
                item_value_str = item.get('final_amount_including_gst', '')
                if item_value_str:
                    try:
                        clean_value = str(item_value_str).replace(',', '').replace('₹', '').strip()
                        if clean_value:
                            item_record.invoice_value_item_wise = Decimal(clean_value)
                    except (InvalidOperation, ValueError):
                        logger.warning(f"Could not parse item value: {item_value_str}")
                
                # Parse individual tax details - NEW SCHEMA HAS DETAILED TAX INFO
                tax_fields = {
                    'cgst_rate': item.get('cgst_rate'),
                    'cgst_amount': item.get('cgst_amount'),
                    'sgst_rate': item.get('sgst_rate'),
                    'sgst_amount': item.get('sgst_amount'),
                    'igst_rate': item.get('igst_rate'),
                    'igst_amount': item.get('igst_amount'),
                    'total_tax_amount': item.get('total_gst_on_item'),            
                    'item_total_amount': item.get('final_amount_including_gst')   
                }
                
                for field, value in tax_fields.items():
                    if value:
                        try:
                            clean_value = str(value).replace(',', '').replace('₹', '').replace('%', '').strip()
                            if clean_value:
                                setattr(item_record, field, Decimal(clean_value))
                        except (InvalidOperation, ValueError):
                            logger.warning(f"Could not parse {field} for item {idx}: {value}")
                
                items_to_create.append(item_record)
            
            # Bulk create all items
            if items_to_create:
                InvoiceItemData.objects.bulk_create(items_to_create)
                logger.info(f"Created {len(items_to_create)} item records for invoice {invoice_data.invoice_number}")
        
        except Exception as e:
            logger.error(f"Error creating invoice items: {str(e)}")
    
    def _save_error_record_direct(self, attachment_info: Dict[str, Any], error_message: str, file_type: str, original_extension: str):
        """Save error record when processing fails (direct mode)"""
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
                
                # --- Failure Reason Logic ---
                failure_reason = classify_failure_reason(
                    file_type,
                    attachment_info.get('invoice_number', ''),
                    attachment_info.get('vendor_gst', ''),
                    '',  # total_gst not available in error context
                    ''   # invoice_total not available in error context
                )
                invoice_data.failure_reason = failure_reason
                # --- End Failure Reason Logic ---
                
                invoice_data.save()
                logger.info(f"Saved error record for PO {attachment_info['po_number']}, attachment {attachment_info['attachment_number']}")
                
        except Exception as e:
            logger.error(f"Error saving error record: {str(e)}")