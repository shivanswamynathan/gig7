import fitz
import json
import logging
import tempfile
import os
import time
from typing import Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from django.conf import settings
from datetime import datetime
import tiktoken

logger = logging.getLogger(__name__)

class InvoicePDFProcessor:
    """
    Invoice PDF processor using LangChain and Google Generative AI
    Now uses dynamic retry logic for Gemini API rate limits (429 errors).
    Fixed delay is no longer required or recommended.
    """
    
    def __init__(self, delay_seconds: float = 0):
        """Initialize the processor. delay_seconds is ignored (kept for backward compatibility)."""
        self.api_key = getattr(settings, 'GOOGLE_API_KEY', None) or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be set in Django settings or environment variables")
        self.model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
        self.llm = GoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1  
        )
        try:
            self.token_encoder = tiktoken.encoding_for_model("gpt-4")
        except:
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
        # Define the invoice schema
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
        # delay_seconds is now ignored; dynamic retry is used instead
        self.delay_seconds = 0
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken
        
        Args:
            text: Input text to count tokens
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.token_encoder.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using word count approximation")
            # Fallback: approximate tokens as words * 1.3
            return int(len(text.split()) * 1.3)
    
    def extract_pan_from_gst(self, gst_number: str) -> str:
        """
        Extract PAN from GST number by removing first 2 and last 3 characters
        
        Args:
            gst_number: 15-digit GST number
            
        Returns:
            10-character PAN number
        """
        if not gst_number or len(gst_number) < 15:
            return ""
        
        # Remove first 2 characters (state code) and last 3 characters (checksum + entity code)
        pan = gst_number[2:12]  # Extract characters 3-12 (0-indexed: 2-11)
        return pan
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """
        Create prompt template for invoice extraction
        
        Returns:
            PromptTemplate for invoice extraction
        """
        template = """
You are an expert invoice data extraction system. Extract structured information from the following invoice text and return it as a valid JSON object.

EXTRACTION RULES:
1. Extract ALL available information from the invoice
2. If a field is not found or unclear, use an empty string ""
3. For numerical values, extract only the number (remove currency symbols like ₹, $, etc.)
4. For dates, use DD/MM/YYYY or DD-MM-YYYY format
5. For GST numbers, extract the complete 15-digit alphanumeric code
6. For PAN numbers, extract the 10-character alphanumeric code
7. For each line item, extract ALL GST details including rates and amounts
8. For GST details, extract CGST, SGST, IGST rates and amounts as provided in the invoice. If specific breakup is not available, use empty strings.
9. Unit of measurement should be units like KG, PCS, NOS, LTR, etc.
10. HSN codes should be extracted for each item - HSN codes can be 4, 6, or 8 digits. Extract the FULL HSN code as shown in the invoice.
11.Extract the Tax Summary section data into tax_summary_by_hsn array
12. Be precise and accurate - double-check all extracted values
13. Return ONLY the JSON object, no additional text

SPECIFIC EXTRACTION GUIDELINES:

VENDOR DETAILS:
- vendor_name: Extract company/business name from header
- vendor_gst: Extract 15-digit GSTIN (format: ##AAAAA####A#A)
- vendor_pan: Extract 10-character PAN (if not shown, derive from positions 3-12 of GST number)

INVOICE INFO:
- invoice_number: Extract invoice/bill number
- invoice_date: Extract invoice date in DD/MM/YYYY format

LINE ITEMS:
For each product/service line:
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
Extract the "Tax Summary" table data:
- hsn_sac_code: HSN/SAC code (may be blank for some rows)
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
- subtotal: Amount before round-off
- round_off: Round-off amount
- final_invoice_amount: Final invoice amount

REQUIRED JSON STRUCTURE:
{schema}

INVOICE TEXT TO PROCESS:
{invoice_text}

Extract the information and return the JSON object:
"""
        
        return PromptTemplate(
            input_variables=["schema", "invoice_text"],
            template=template
        )
    
    def validate_and_clean_json(self, json_str: str) -> Dict[str, Any]:
        """
        Validate and clean the extracted JSON response
        
        Args:
            json_str: Raw JSON string from LLM
            
        Returns:
            Cleaned and validated JSON dict
        """
        try:
            # Clean the response - remove markdown formatting if present
            json_str = json_str.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            # Parse the JSON
            extracted_data = json.loads(json_str)
            
            # Validate structure against schema
            validated_data = self.invoice_schema.copy()
            
            def deep_merge(target, source):
                for key, value in source.items():
                    if key in target:
                        if isinstance(target[key], dict) and isinstance(value, dict):
                            deep_merge(target[key], value)
                        elif isinstance(target[key], list) and isinstance(value, list):
                            # For arrays, use the extracted data
                            target[key] = value
                        else:
                            target[key] = value
                    else:
                        # For unknown fields, still include them
                        target[key] = value
            
            deep_merge(validated_data, extracted_data)
            
            # Auto-extract PAN from GST if PAN is not provided
            vendor_gst = validated_data.get("vendor_details", {}).get("vendor_gst", "")
            vendor_pan = validated_data.get("vendor_details", {}).get("vendor_pan", "")
            
            if vendor_gst and not vendor_pan:
                extracted_pan = self.extract_pan_from_gst(vendor_gst)
                if extracted_pan:
                    validated_data["vendor_details"]["vendor_pan"] = extracted_pan
                    logger.info(f"Extracted PAN {extracted_pan} from GST number")
            
            # Validate and log line items structure
            line_items = validated_data.get("line_items", [])
            if isinstance(line_items, list):
                for i, item in enumerate(line_items):
                    if isinstance(item, dict):
                        item_desc = item.get('item_description', 'Unknown')
                        total_gst = item.get('total_gst_on_item', 'Not specified')
                        logger.info(f"Item {i+1}: {item_desc} - Total GST: {total_gst}")
            
            # Log tax summary extraction
            tax_summary = validated_data.get("tax_summary_by_hsn", [])
            if isinstance(tax_summary, list):
                logger.info(f"Extracted {len(tax_summary)} HSN-wise tax summary entries")
            
            return validated_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Raw JSON string: {json_str[:500]}...")
            raise ValueError(f"Invalid JSON format returned by LLM: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating JSON: {str(e)}")
            raise
    
    def _invoke_llm_with_retry(self, prompt: str) -> str:
        """
        Invoke Gemini LLM with automatic retry on 429 errors using retry_delay from error message.
        """
        max_retries = 5
        attempt = 0
        while True:
            try:
                return self.llm.invoke(prompt)
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

    def process_uploaded_file(self, uploaded_file, delay_seconds: float = 0) -> Dict[str, Any]:
        """
        Process Django uploaded file and extract invoice data.
        delay_seconds is ignored; dynamic retry is used for rate limiting.
        """
        temp_path = None
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            # Extract text from PDF
            logger.info(f"Extracting text from uploaded file: {uploaded_file.name}")
            extracted_text = self.extract_text_from_pdf(temp_path)
            
            if not extracted_text:
                raise ValueError("No text could be extracted from the PDF file")
            
            # Create and format the prompt
            prompt_template = self.create_extraction_prompt()
            formatted_prompt = prompt_template.format(
                schema=json.dumps(self.invoice_schema, indent=2),
                invoice_text=extracted_text
            )

            # Count input tokens
            input_tokens = self.count_tokens(formatted_prompt)
            logger.info(f"Input tokens: {input_tokens}")

            # No fixed delay; dynamic retry is used
            logger.info("Processing invoice data with LLM (with dynamic retry)...")
            llm_response = self._invoke_llm_with_retry(formatted_prompt)

            # Count output tokens
            output_tokens = self.count_tokens(llm_response)
            total_tokens = input_tokens + output_tokens
            logger.info(f"Output tokens: {output_tokens}, Total tokens: {total_tokens}")
            
            # Validate and clean the JSON response
            extracted_data = self.validate_and_clean_json(llm_response)
            
            # Add processing metadata
            extracted_data["_metadata"] = {
                "filename": uploaded_file.name,
                "file_size": uploaded_file.size,
                "processed_at": datetime.now().isoformat(),
                "model_used": self.model_name,
                "text_length": len(extracted_text),
                "processing_status": "success",
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            }
            
            # Log summary of extracted data
            line_items = extracted_data.get("line_items", [])
            tax_summary = extracted_data.get("tax_summary_by_hsn", [])
            vendor_name = extracted_data.get("vendor_details", {}).get("vendor_name", "Unknown")
            invoice_number = extracted_data.get("invoice_info", {}).get("invoice_number", "Unknown")
            final_amount = extracted_data.get("invoice_totals", {}).get("final_invoice_amount", "Unknown")
            
            logger.info(f"Successfully processed invoice: {uploaded_file.name}")
            logger.info(f"Vendor: {vendor_name}, Invoice: {invoice_number}, Amount: Rs.{final_amount}")
            logger.info(f"Extracted {len(line_items)} line items and {len(tax_summary)} HSN tax entries")
            logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing uploaded file {uploaded_file.name}: {str(e)}")
            
            # Return error information
            error_data = self.invoice_schema.copy()
            error_data["_metadata"] = {
                "filename": uploaded_file.name,
                "file_size": getattr(uploaded_file, 'size', 0),
                "processed_at": datetime.now().isoformat(),
                "processing_status": "failed",
                "error_message": str(e),
                "schema_version": "3.0_optimized_gst",
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            raise Exception(f"Failed to process invoice: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            
        
    
    def process_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        Process invoice from file path (for testing purposes).
        delay_seconds is ignored; dynamic retry is used for rate limiting.
        """
        try:
            # Extract text from PDF
            logger.info(f"Extracting text from file: {file_path}")
            extracted_text = self.extract_text_from_pdf(file_path)
            
            if not extracted_text:
                raise ValueError("No text could be extracted from the PDF file")
            
            # Create and format the prompt
            prompt_template = self.create_extraction_prompt()
            formatted_prompt = prompt_template.format(
                schema=json.dumps(self.invoice_schema, indent=2),
                invoice_text=extracted_text
            )
            
            # Count input tokens
            input_tokens = self.count_tokens(formatted_prompt)
            logger.info(f"Input tokens: {input_tokens}")

            # No fixed delay; dynamic retry is used
            logger.info("Processing invoice data with LLM (with dynamic retry)...")
            llm_response = self._invoke_llm_with_retry(formatted_prompt)

            # Count output tokens
            output_tokens = self.count_tokens(llm_response)
            total_tokens = input_tokens + output_tokens
            logger.info(f"Output tokens: {output_tokens}, Total tokens: {total_tokens}")
            
            # Validate and clean the JSON response
            extracted_data = self.validate_and_clean_json(llm_response)
            
            # Add processing metadata
            extracted_data["_metadata"] = {
                "file_path": file_path,
                "processed_at": datetime.now().isoformat(),
                "model_used": self.model_name,
                "text_length": len(extracted_text),
                "processing_status": "success",
                "schema_version": "3.0_optimized_gst",
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            }
            
            # Log summary of extracted data
            line_items = extracted_data.get("line_items", [])
            tax_summary = extracted_data.get("tax_summary_by_hsn", [])
            vendor_name = extracted_data.get("vendor_details", {}).get("vendor_name", "Unknown")
            invoice_number = extracted_data.get("invoice_info", {}).get("invoice_number", "Unknown")
            final_amount = extracted_data.get("invoice_totals", {}).get("final_invoice_amount", "Unknown")
            
            logger.info(f"Successfully processed invoice from: {file_path}")
            logger.info(f"Vendor: {vendor_name}, Invoice: {invoice_number}, Amount: Rs.{final_amount}")
            logger.info(f"Extracted {len(line_items)} line items and {len(tax_summary)} HSN tax entries")
            logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise Exception(f"Failed to process invoice: {str(e)}")


