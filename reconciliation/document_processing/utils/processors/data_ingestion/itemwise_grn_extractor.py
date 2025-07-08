import pandas as pd
import openpyxl
import csv
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple
from django.db import transaction
from django.core.exceptions import ValidationError
from document_processing.models import ItemWiseGrn, UploadHistory
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class ItemWiseGrnDataProcessor:
    """
    Processor for handling Item-wise GRN data from Excel/CSV files
    """
    
    def __init__(self):
        self.batch_id = None
        self.upload_history = None
        self.processed_records = 0
        self.successful_records = 0
        self.failed_records = 0
        self.errors = []
        
        # Expected column mappings (case-insensitive)
        self.column_mapping = {
            's.no.': 's_no',
            'sno': 's_no',
            'serial': 's_no',
            'type': 'type',
            'sku code': 'sku_code',
            'category': 'category',
            'sub category': 'sub_category',
            'item name': 'item_name',
            'unit': 'unit',
            'grn no.': 'grn_no',
            'grn no': 'grn_no',
            'grn number': 'grn_no',
            'hsn no.': 'hsn_no',
            'hsn no': 'hsn_no',
            'hsn code': 'hsn_no',
            'po no.': 'po_no',
            'po no': 'po_no',
            'po number': 'po_no',
            'remarks': 'remarks',
            'created by': 'created_by',
            'grn created at': 'grn_created_at',
            'grn creation date': 'grn_created_at',
            'seller invoice no': 'seller_invoice_no',
            'supplier invoice date': 'supplier_invoice_date',
            'supplier': 'supplier',
            'vendor': 'supplier',
            'concerned person': 'concerned_person',
            'pickup location': 'pickup_location',
            'pickup gstin': 'pickup_gstin',
            'pickup code': 'pickup_code',
            'pickup city': 'pickup_city',
            'pickup state': 'pickup_state',
            'delivery location': 'delivery_location',
            'delivery gstin': 'delivery_gstin',
            'delivery code': 'delivery_code',
            'delivery city': 'delivery_city',
            'delivery state': 'delivery_state',
            'price': 'price',
            'received qty': 'received_qty',
            'received quantity': 'received_qty',
            'returned qty': 'returned_qty',
            'returned quantity': 'returned_qty',
            'discount': 'discount',
            'tax': 'tax',
            'sgst tax': 'sgst_tax',
            'sgst tax amount': 'sgst_tax_amount',
            'cgst tax': 'cgst_tax',
            'cgst tax amount': 'cgst_tax_amount',
            'igst tax': 'igst_tax',
            'igst tax amount': 'igst_tax_amount',
            'cess': 'cess',
            'subtotal': 'subtotal',
            'vat(%)': 'vat_percent',
            'vat (%)': 'vat_percent',
            'vat(amount)': 'vat_amount',
            'vat (amount)': 'vat_amount',
            'item tcs(%)': 'item_tcs_percent',
            'item tcs (%)': 'item_tcs_percent',
            'item tcs(amount)': 'item_tcs_amount',
            'item tcs (amount)': 'item_tcs_amount',
            'tax amount': 'tax_amount',
            'bill tcs': 'bill_tcs',
            'delivery charges': 'delivery_charges',
            'delivery charges tax(%)': 'delivery_charges_tax_percent',
            'delivery charges tax (%)': 'delivery_charges_tax_percent',
            'additional charges': 'additional_charges',
            'inv discount': 'inv_discount',
            'invoice discount': 'inv_discount',
            'roundoff': 'round_off',
            'round off': 'round_off',
            'total': 'total',
            'attachment upload date': 'attachment_upload_date',
            'attachment-1': 'attachment_1',
            'attachment-2': 'attachment_2',
            'attachment-3': 'attachment_3',
            'attachment-4': 'attachment_4',
            'attachment-5': 'attachment_5',
        }
    
    def create_batch_id(self) -> str:
        """Generate unique batch ID for upload session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"ITEM_GRN_BATCH_{timestamp}_{unique_id}"
    
    def normalize_column_names(self, columns: List[str]) -> Dict[str, str]:
        """
        Normalize column names to match expected field names
        
        Args:
            columns: List of column names from the file
            
        Returns:
            Dictionary mapping original column names to normalized names
        """
        normalized = {}
        for col in columns:
            col_lower = col.lower().strip()
            if col_lower in self.column_mapping:
                normalized[col] = self.column_mapping[col_lower]
            else:
                # Try partial matching
                for key, value in self.column_mapping.items():
                    if key in col_lower or col_lower in key:
                        normalized[col] = value
                        break
                else:
                    logger.warning(f"Unknown column: {col}")
                    # Create a safe field name for unknown columns
                    safe_name = col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent')
                    normalized[col] = safe_name
        
        return normalized
    
    def clean_value(self, value: Any) -> Any:
        """Clean individual values, handle empty strings and dashes"""
        if pd.isna(value) or value is None:
            return None
        
        if isinstance(value, str):
            value = value.strip()
            # Convert common "empty" indicators to None
            if value in ['-', 'N/A', 'NA', '', 'null', 'NULL']:
                return None
            # Remove carriage returns
            value = value.replace('\r', '').replace('\n', ' ')
        
        return value
    
    def parse_date(self, date_value: Any) -> datetime.date:
        """
        Parse date from various formats
        
        Args:
            date_value: Date value in various formats
            
        Returns:
            Parsed date object
        """
        date_value = self.clean_value(date_value)
        if date_value is None:
            return None
            
        if isinstance(date_value, datetime):
            return date_value.date()
        
        if isinstance(date_value, str):
            # Try various date formats
            formats = [
                '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', 
                '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d',
                '%d/%m/%y', '%m/%d/%y', '%y-%m-%d',
                '%d-%m-%y', '%m-%d-%y', '%y/%m/%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_value, fmt).date()
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date: {date_value}")
            return None
        
        return None
    
    def parse_decimal(self, value: Any, field_name: str) -> Decimal:
        """
        Parse decimal values, handling various formats
        
        Args:
            value: Numeric value in various formats
            field_name: Name of the field for error reporting
            
        Returns:
            Parsed decimal value
        """
        value = self.clean_value(value)
        if value is None:
            return None
            
        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError):
                logger.warning(f"Could not convert to decimal for {field_name}: {value}")
                return None
        
        if isinstance(value, str):
            # Remove currency symbols and commas
            cleaned = value.replace(',', '').replace('\u20b9', '').replace('\u0024', '').replace(' ', '')
            try:
                return Decimal(cleaned)
            except (InvalidOperation, ValueError):
                logger.warning(f"Could not parse decimal value for {field_name}: {value}")
                return None
        
        return None
    
    def parse_integer(self, value: Any, field_name: str) -> int:
        """Parse integer values"""
        value = self.clean_value(value)
        if value is None:
            return None
            
        try:
            if isinstance(value, (int, float)):
                return int(value)
            elif isinstance(value, str):
                # Remove any non-numeric characters except decimal point
                cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
                if cleaned:
                    return int(float(cleaned))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse integer value for {field_name}: {value}")
        
        return None
    
    def is_empty_row(self, row_data: Dict[str, Any]) -> bool:
        """
        Check if a row is essentially empty (all values are None or empty)
        
        Args:
            row_data: Dictionary containing row data
            
        Returns:
            True if row is empty, False otherwise
        """
        essential_fields = ['grn_no', 'item_name', 'supplier', 'sku_code']
        
        # Check if any essential field has a meaningful value
        for field in essential_fields:
            value = self.clean_value(row_data.get(field))
            if value is not None:
                return False
        
        return True
    
    def is_duplicate_row(self, record_data: Dict[str, Any], existing_records: List[Dict[str, Any]]) -> bool:
        """
        Check if the current row is a duplicate of any existing record
        
        Args:
            record_data: Current record data
            existing_records: List of already processed records
            
        Returns:
            True if duplicate, False otherwise
        """
        key_fields = ['grn_no', 'po_no', 'sku_code', 'item_name']
        
        current_key = tuple(self.clean_value(record_data.get(field)) for field in key_fields)
        
        for existing_record in existing_records:
            existing_key = tuple(self.clean_value(existing_record.get(field)) for field in key_fields)
            if current_key == existing_key and all(val is not None for val in current_key):
                return True
        
        return False
    
    def validate_record(self, record: Dict[str, Any], row_num: int) -> Tuple[bool, List[str]]:
        """
        Validate a single record
        
        Args:
            record: Dictionary containing record data
            row_num: Row number for error reporting
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields validation (relaxed - only check for truly essential fields)
        required_fields = ['s_no']
        
        for field in required_fields:
            if not record.get(field):
                errors.append(f"Row {row_num}: Missing required field '{field}'")
        
        # Validate s_no is positive integer
        if record.get('s_no') and record['s_no'] <= 0:
            errors.append(f"Row {row_num}: Serial number must be positive")
        
        # Validate decimal amounts are non-negative (if present)
        decimal_fields = [
            'price', 'received_qty', 'returned_qty', 'discount', 'sgst_tax_amount',
            'cgst_tax_amount', 'igst_tax_amount', 'cess', 'subtotal', 'tax_amount',
            'delivery_charges', 'additional_charges', 'total'
        ]
        
        for field in decimal_fields:
            value = record.get(field)
            if value is not None and value < 0:
                errors.append(f"Row {row_num}: {field} cannot be negative")
        
        # Validate percentage fields are within reasonable range
        percentage_fields = ['tax', 'sgst_tax', 'cgst_tax', 'igst_tax', 'delivery_charges_tax_percent']
        for field in percentage_fields:
            value = record.get(field)
            if value is not None and (value < 0 or value > 100):
                errors.append(f"Row {row_num}: {field} should be between 0 and 100")
        
        # Validate GST numbers format (if present)
        gstin_fields = ['pickup_gstin', 'delivery_gstin']
        for field in gstin_fields:
            value = record.get(field)
            if value and len(str(value)) != 15:
                errors.append(f"Row {row_num}: {field} should be 15 characters long")
        
        return len(errors) == 0, errors
    
    def process_excel_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process Excel file and extract item-wise GRN data
        
        Args:
            file_path: Path to the Excel file
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Read Excel file
            logger.info(f"Reading Excel file: {filename}")
            
            # Try to detect header row
            df = pd.read_excel(file_path, header=None)
            
            # Find header row by looking for 'S.No.' or similar
            header_row = None
            for idx, row in df.iterrows():
                if any('s.no' in str(cell).lower() for cell in row if pd.notna(cell)):
                    header_row = idx
                    break
            
            if header_row is None:
                # Try first row as header
                header_row = 0
            
            # Read again with proper header
            df = pd.read_excel(file_path, header=header_row)
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            logger.info(f"Found {len(df)} data rows")
            
            return self._process_dataframe(df, filename)
            
        except Exception as e:
            logger.error(f"Error processing Excel file {filename}: {str(e)}")
            raise Exception(f"Failed to process Excel file: {str(e)}")
    
    def process_csv_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process CSV file and extract item-wise GRN data
        
        Args:
            file_path: Path to the CSV file
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Read CSV file
            logger.info(f"Reading CSV file: {filename}")
            
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
            
            logger.info(f"Found {len(df)} data rows")
            
            return self._process_dataframe(df, filename)
            
        except Exception as e:
            logger.error(f"Error processing CSV file {filename}: {str(e)}")
            raise Exception(f"Failed to process CSV file: {str(e)}")
    
    def _process_dataframe(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Process pandas DataFrame containing item-wise GRN data
        
        Args:
            df: Pandas DataFrame with data
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Create batch ID
            self.batch_id = self.create_batch_id()
            
            # Create upload history record
            self.upload_history = UploadHistory.objects.create(
                batch_id=self.batch_id,
                filename=filename,
                file_size=0,  # Will be updated later
                total_records=len(df),
                successful_records=0,
                failed_records=0,
                processing_status='processing'
            )
            
            # Normalize column names
            column_mapping = self.normalize_column_names(df.columns.tolist())
            df = df.rename(columns=column_mapping)
            
            logger.info(f"Column mapping: {column_mapping}")
            
            # Process records
            records_to_create = []
            processed_records_data = []  # For duplicate checking
            self.errors = []
            
            with transaction.atomic():
                for idx, row in df.iterrows():
                    try:
                        # Convert row to dictionary
                        record_data = row.to_dict()
                        
                        # Check if row is empty
                        if self.is_empty_row(record_data):
                            logger.info(f"Skipping empty row {idx + 1}")
                            continue
                        
                        # Parse and clean data
                        processed_record = self._parse_record(record_data, idx + 1)
                        
                        # Check for duplicates
                        if self.is_duplicate_row(processed_record, processed_records_data):
                            logger.warning(f"Skipping duplicate row {idx + 1}")
                            continue
                        
                        # Validate record
                        is_valid, validation_errors = self.validate_record(processed_record, idx + 1)
                        
                        if is_valid:
                            # Add metadata
                            processed_record['upload_batch_id'] = self.batch_id
                            processed_record['uploaded_filename'] = filename
                            
                            # Create model instance
                            grn_record = ItemWiseGrn(**processed_record)
                            records_to_create.append(grn_record)
                            processed_records_data.append(processed_record)
                            self.successful_records += 1
                        else:
                            self.errors.extend(validation_errors)
                            self.failed_records += 1
                        
                        self.processed_records += 1
                        
                    except Exception as e:
                        error_msg = f"Row {idx + 1}: Error processing record - {str(e)}"
                        logger.error(error_msg)
                        self.errors.append(error_msg)
                        self.failed_records += 1
                        self.processed_records += 1
                
                # Bulk create records
                if records_to_create:
                    ItemWiseGrn.objects.bulk_create(records_to_create, batch_size=100)
                    logger.info(f"Successfully created {len(records_to_create)} records")
                
                # Update upload history
                self.upload_history.successful_records = self.successful_records
                self.upload_history.failed_records = self.failed_records
                self.upload_history.completed_at = datetime.now()
                
                if self.failed_records == 0:
                    self.upload_history.processing_status = 'completed'
                elif self.successful_records == 0:
                    self.upload_history.processing_status = 'failed'
                else:
                    self.upload_history.processing_status = 'partial'
                
                if self.errors:
                    self.upload_history.error_details = '\n'.join(self.errors[:10])  # Store first 10 errors
                
                self.upload_history.save()
            
            # Return processing results
            return {
                'batch_id': self.batch_id,
                'total_records': self.processed_records,
                'successful_records': self.successful_records,
                'failed_records': self.failed_records,
                'errors': self.errors,
                'processing_status': self.upload_history.processing_status,
                'success_rate': self.upload_history.success_rate
            }
            
        except Exception as e:
            logger.error(f"Error processing dataframe: {str(e)}")
            if self.upload_history:
                self.upload_history.processing_status = 'failed'
                self.upload_history.error_details = str(e)
                self.upload_history.save()
            raise
    
    def _parse_record(self, record_data: Dict[str, Any], row_num: int) -> Dict[str, Any]:
        """
        Parse and clean a single record
        
        Args:
            record_data: Raw record data from DataFrame
            row_num: Row number for error reporting
            
        Returns:
            Dictionary with cleaned record data
        """
        parsed_record = {}
        
        # Parse serial number
        s_no = record_data.get('s_no')
        if s_no is not None:
            parsed_s_no = self.parse_integer(s_no, 's_no')
            parsed_record['s_no'] = parsed_s_no if parsed_s_no is not None else row_num
        else:
            parsed_record['s_no'] = row_num
        
        # Parse text fields
        text_fields = [
            'type', 'sku_code', 'category', 'sub_category', 'item_name', 'unit',
            'grn_no', 'hsn_no', 'po_no', 'remarks', 'created_by', 'seller_invoice_no',
            'supplier', 'concerned_person', 'pickup_location', 'pickup_gstin',
            'pickup_code', 'pickup_city', 'pickup_state', 'delivery_location',
            'delivery_gstin', 'delivery_code', 'delivery_city', 'delivery_state',
            'vat_percent', 'vat_amount', 'item_tcs_percent'
        ]
        
        for field in text_fields:
            value = self.clean_value(record_data.get(field))
            parsed_record[field] = value
        
        # Parse dates
        date_fields = ['grn_created_at', 'supplier_invoice_date', 'attachment_upload_date']
        for field in date_fields:
            parsed_record[field] = self.parse_date(record_data.get(field))
        
        # Parse decimal fields
        decimal_fields = [
            'price', 'received_qty', 'returned_qty', 'discount', 'tax',
            'sgst_tax', 'sgst_tax_amount', 'cgst_tax', 'cgst_tax_amount',
            'igst_tax', 'igst_tax_amount', 'cess', 'subtotal', 'tax_amount',
            'bill_tcs', 'delivery_charges', 'delivery_charges_tax_percent',
            'additional_charges', 'inv_discount', 'round_off', 'total'
        ]
        
        for field in decimal_fields:
            parsed_record[field] = self.parse_decimal(record_data.get(field), field)
        
        # Parse integer fields
        integer_fields = ['item_tcs_amount']
        for field in integer_fields:
            parsed_record[field] = self.parse_integer(record_data.get(field), field)
        
        # Parse URL fields
        url_fields = ['attachment_1', 'attachment_2', 'attachment_3', 'attachment_4', 'attachment_5']
        for field in url_fields:
            value = self.clean_value(record_data.get(field))
            # Basic URL validation
            if value and (value.startswith('http://') or value.startswith('https://')):
                parsed_record[field] = value
            else:
                parsed_record[field] = None
        
        return parsed_record