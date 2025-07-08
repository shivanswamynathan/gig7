import pandas as pd
import openpyxl
import csv
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple
from django.db import transaction
from django.core.exceptions import ValidationError
from document_processing.models import PoGrn, UploadHistory

logger = logging.getLogger(__name__)


class PoGrnDataProcessor:
    """
    Processor for handling PO-GRN data from Excel/CSV files
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
            'location': 'location',
            'po no.': 'po_number',
            'po_no': 'po_number',
            'po number': 'po_number',
            'po_creation_date': 'po_creation_date',
            'po creation date': 'po_creation_date',
            'no_item_in_po': 'no_item_in_po',
            'no item in po': 'no_item_in_po',
            'items in po': 'no_item_in_po',
            'po_amount': 'po_amount',
            'po amount': 'po_amount',
            'po_status': 'po_status',
            'po status': 'po_status',
            'supplier_name': 'supplier_name',
            'supplier name': 'supplier_name',
            'vendor name': 'supplier_name',
            'concerned person': 'concerned_person',
            'concerned_person': 'concerned_person',
            'grn_no': 'grn_number',
            'grn no.': 'grn_number',
            'grn number': 'grn_number',
            'grn_creation_date': 'grn_creation_date',
            'grn creation date': 'grn_creation_date',
            'no_item_in_grn': 'no_item_in_grn',
            'no item in grn': 'no_item_in_grn',
            'items in grn': 'no_item_in_grn',
            'received status': 'received_status',
            'received_status': 'received_status',
            'grn_subtotal': 'grn_subtotal',
            'grn subtotal': 'grn_subtotal',
            'grn_tax': 'grn_tax',
            'grn tax': 'grn_tax',
            'grn_amount': 'grn_amount',
            'grn amount': 'grn_amount',
        }
    
    def create_batch_id(self) -> str:
        """Generate unique batch ID for upload session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"BATCH_{timestamp}_{unique_id}"
    
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
                    normalized[col] = col.lower().replace(' ', '_')
        
        return normalized
    
    def parse_date(self, date_value: Any) -> datetime.date:
        """
        Parse date from various formats
        
        Args:
            date_value: Date value in various formats
            
        Returns:
            Parsed date object
        """
        if pd.isna(date_value) or date_value is None:
            return None
            
        if isinstance(date_value, datetime):
            return date_value.date()
        
        if isinstance(date_value, str):
            # Try various date formats
            formats = [
                '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d', 
                '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_value.strip(), fmt).date()
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date: {date_value}")
            return None
        
        return None
    
    def parse_numeric(self, value: Any, field_name: str) -> float:
        """
        Parse numeric values, handling various formats
        
        Args:
            value: Numeric value in various formats
            field_name: Name of the field for error reporting
            
        Returns:
            Parsed numeric value
        """
        if pd.isna(value) or value is None or value == '':
            return None
            
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove currency symbols and commas
            cleaned = value.strip().replace(',', '').replace('\u20b9', '').replace('$', '').replace(' ', '')
            try:
                return float(cleaned)
            except ValueError:
                logger.warning(f"Could not parse numeric value for {field_name}: {value}")
                return None
        
        return None
    
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
        
        # Required fields validation
        required_fields = ['s_no', 'location', 'po_number', 'po_creation_date', 'supplier_name']
        
        for field in required_fields:
            if not record.get(field):
                errors.append(f"Row {row_num}: Missing required field '{field}'")
        
        # Validate s_no is positive integer
        if record.get('s_no') and record['s_no'] <= 0:
            errors.append(f"Row {row_num}: Serial number must be positive")
        
        # Validate amounts are positive
        amount_fields = ['po_amount', 'grn_subtotal', 'grn_tax', 'grn_amount']
        for field in amount_fields:
            if record.get(field) is not None and record[field] < 0:
                errors.append(f"Row {row_num}: {field} cannot be negative")
        
        # Validate item counts are non-negative
        count_fields = ['no_item_in_po', 'no_item_in_grn']
        for field in count_fields:
            if record.get(field) is not None and record[field] < 0:
                errors.append(f"Row {row_num}: {field} cannot be negative")
        
        return len(errors) == 0, errors
    
    def process_excel_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process Excel file and extract PO-GRN data
        
        Args:
            file_path: Path to the Excel file
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Read Excel file
            logger.info(f"Reading Excel file: {filename}")
            
            # Try to detect header row (usually row 6 based on sample)
            df = pd.read_excel(file_path, header=None)
            
            # Find header row by looking for 'S.No.' or similar
            header_row = None
            for idx, row in df.iterrows():
                if any('s.no' in str(cell).lower() for cell in row if pd.notna(cell)):
                    header_row = idx
                    break
            
            if header_row is None:
                # Default to row 5 (6th row) based on sample
                header_row = 5
            
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
        Process CSV file and extract PO-GRN data
        
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
        Process pandas DataFrame containing PO-GRN data
        
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
            self.errors = []
            
            with transaction.atomic():
                for idx, row in df.iterrows():
                    try:
                        # Convert row to dictionary
                        record_data = row.to_dict()
                        
                        # Parse and clean data
                        processed_record = self._parse_record(record_data, idx + 1)
                        
                        # Validate record
                        is_valid, validation_errors = self.validate_record(processed_record, idx + 1)
                        
                        if is_valid:
                            # Add metadata
                            processed_record['upload_batch_id'] = self.batch_id
                            processed_record['uploaded_filename'] = filename
                            
                            # Create model instance
                            po_grn_record = PoGrn(**processed_record)
                            records_to_create.append(po_grn_record)
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
                    PoGrn.objects.bulk_create(records_to_create, batch_size=100)
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
        if pd.notna(s_no):
            try:
                parsed_record['s_no'] = int(float(s_no))
            except (ValueError, TypeError):
                parsed_record['s_no'] = row_num
        else:
            parsed_record['s_no'] = row_num
        
        # Parse text fields
        text_fields = ['location', 'po_number', 'po_status', 'supplier_name', 
                      'concerned_person', 'grn_number', 'received_status']
        
        for field in text_fields:
            value = record_data.get(field)
            if pd.notna(value) and value != '':
                parsed_record[field] = str(value).strip()
                # Handle special case for concerned_person
                if field == 'concerned_person' and parsed_record[field] in ['-', 'N/A', 'NA']:
                    parsed_record[field] = None
            else:
                parsed_record[field] = None if field == 'concerned_person' else ''
        
        # Parse dates
        date_fields = ['po_creation_date', 'grn_creation_date']
        for field in date_fields:
            parsed_record[field] = self.parse_date(record_data.get(field))
        
        # Parse numeric fields
        numeric_fields = {
            'no_item_in_po': int,
            'po_amount': float,
            'no_item_in_grn': int,
            'grn_subtotal': float,
            'grn_tax': float,
            'grn_amount': float
        }
        
        for field, data_type in numeric_fields.items():
            value = self.parse_numeric(record_data.get(field), field)
            if value is not None:
                parsed_record[field] = data_type(value) if data_type == int else value
            else:
                parsed_record[field] = None
        
        return parsed_record