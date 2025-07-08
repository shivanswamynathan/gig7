import asyncio
import logging
import json
import time
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
from django.db import transaction
from django.db.models import Sum, Count, Q
from asgiref.sync import sync_to_async
from langchain_google_genai import GoogleGenerativeAI
from django.conf import settings
import os

from document_processing.models import InvoiceData, InvoiceItemData, ItemWiseGrn, InvoiceGrnReconciliation

logger = logging.getLogger(__name__)


class LLMOnlyReconciliationProcessor:
    """
    LLM-ONLY Reconciliation: Pass complete Invoice JSON and ItemWise GRN JSON to LLM for analysis
    
    Field mappings for LLM reference:
    - po_number -> po_no
    - grn_number -> grn_no  
    - invoice_number -> seller_invoice_no
    - vendor_name -> pickup_location
    - vendor_gst -> pickup_gstin
    - invoice_date -> supplier_invoice_date
    - invoice_value_without_gst -> subtotal
    - cgst_amount -> cgst_tax_amount
    - sgst_amount -> sgst_tax_amount
    - igst_amount -> igst_tax_amount
    - invoice_total_post_gst -> total
    - invoice_date <= grn_created_at (date validation)
    """
    
    def __init__(self, delay_seconds: float = 1.0, max_concurrent: int = 10):
        self.delay_seconds = delay_seconds
        self.max_concurrent = max_concurrent
        self.setup_llm()
        
        self.stats = {
            'total_processed': 0,
            'perfect_matches': 0,
            'partial_matches': 0,
            'amount_mismatches': 0,
            'vendor_mismatches': 0,
            'date_mismatches': 0,
            'no_matches': 0,
            'errors': 0,
            'llm_analyses': 0
        }
    
    def setup_llm(self):
        """Setup LLM - REQUIRED for this processor"""
        try:
            api_key = getattr(settings, 'GOOGLE_API_KEY', None) or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required for LLM-only reconciliation")
            
            model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
            self.llm = GoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1
            )
            logger.info(f"LLM initialized for LLM-only reconciliation: {model_name}")
            
        except Exception as e:
            logger.error(f"LLM setup failed: {str(e)}")
            raise ValueError(f"LLM setup is required: {str(e)}")

    async def process_batch_async(self, invoice_ids: List[int] = None, batch_size: int = 100) -> Dict[str, Any]:
        """Process invoices using LLM-only analysis"""
        try:
            logger.info(f"Starting LLM-ONLY reconciliation")
            logger.info(f"Settings: delay={self.delay_seconds}s, concurrent={self.max_concurrent}")
            
            # Get invoices to process
            if invoice_ids:
                invoices = await sync_to_async(list)(
                    InvoiceData.objects.filter(id__in=invoice_ids, processing_status='completed')
                )
            else:
                invoices = await sync_to_async(list)(
                    InvoiceData.objects.filter(processing_status='completed')
                )
            
            total_invoices = len(invoices)
            logger.info(f"Processing {total_invoices} invoices with LLM-only analysis...")
            
            # Process in batches with semaphore
            semaphore = asyncio.Semaphore(self.max_concurrent)
            results = []
            
            for i in range(0, total_invoices, batch_size):
                batch = invoices[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} invoices")
                
                # Create tasks for this batch
                tasks = [
                    self._process_single_invoice_with_llm_only(invoice, semaphore)
                    for invoice in batch
                ]
                
                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Invoice {batch[j].id} failed: {str(result)}")
                        self.stats['errors'] += 1
                    else:
                        results.append(result)
                        self.stats['total_processed'] += 1
                
                # Log progress
                progress_pct = (self.stats['total_processed'] / total_invoices) * 100
                logger.info(f"Progress: {self.stats['total_processed']}/{total_invoices} ({progress_pct:.1f}%)")
            
            logger.info("LLM-only reconciliation completed!")
            logger.info(f"Final Stats: {self.stats}")
            
            return {
                'success': True,
                'total_processed': self.stats['total_processed'],
                'stats': self.stats,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"LLM-only batch processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }

    async def _process_single_invoice_with_llm_only(self, invoice: InvoiceData, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process single invoice using LLM-only analysis"""
        async with semaphore:
            try:
                logger.info(f"Processing invoice {invoice.id} - PO: {invoice.po_number} with LLM-only analysis")
                
                # Step 1: Find GRN matches
                grn_items = await self._find_grn_matches(invoice)
                
                if not grn_items:
                    self.stats['no_matches'] += 1
                    return await self._create_no_match_record(invoice)
                
                logger.info(f"Found {len(grn_items)} GRN items for invoice {invoice.id}")
                
                # Step 2: Get invoice items
                invoice_items = await sync_to_async(list)(
                    InvoiceItemData.objects.filter(invoice_data=invoice)
                )
                
                # Step 3: Convert to complete JSON format
                invoice_json = await self._convert_invoice_to_complete_json(invoice, invoice_items)
                grn_json = await self._convert_grn_to_complete_json(grn_items)
                
                # Step 4: LLM ANALYSIS with complete JSON data
                llm_analysis = await self._llm_complete_json_analysis(invoice_json, grn_json)
                
                if llm_analysis and llm_analysis.get('success'):
                    self.stats['llm_analyses'] += 1
                
                # Step 5: Create reconciliation record based on LLM analysis
                reconciliation = await self._create_reconciliation_from_llm_analysis(
                    invoice, grn_items, invoice_json, grn_json, llm_analysis
                )
                
                # Step 6: Update statistics based on LLM results
                if llm_analysis and llm_analysis.get('success'):
                    match_status = llm_analysis.get('overall_match_status', 'partial_match')
                    self._update_statistics(match_status)
                
                return {
                    'invoice_id': invoice.id,
                    'reconciliation_id': reconciliation.id,
                    'match_status': llm_analysis.get('overall_match_status', 'partial_match') if llm_analysis else 'no_analysis',
                    'grn_items_matched': len(grn_items),
                    'llm_analysis_success': llm_analysis.get('success', False) if llm_analysis else False,
                    'llm_discrepancies_found': len(llm_analysis.get('discrepancies', [])) if llm_analysis else 0
                }
                
            except Exception as e:
                logger.error(f"Error processing invoice {invoice.id}: {str(e)}")
                raise

    async def _find_grn_matches(self, invoice: InvoiceData) -> List[ItemWiseGrn]:
        """Find GRN matches using basic matching logic"""
        
        # Primary matching: po_number -> po_no
        if invoice.po_number:
            grn_items = await sync_to_async(list)(
                ItemWiseGrn.objects.filter(po_no=invoice.po_number)
            )
            
            if grn_items:
                logger.info(f"Found {len(grn_items)} GRN items using PO matching")
                return grn_items
        
        # Fallback: invoice_number -> seller_invoice_no
        if invoice.invoice_number:
            grn_items = await sync_to_async(list)(
                ItemWiseGrn.objects.filter(seller_invoice_no=invoice.invoice_number)
            )
            if grn_items:
                logger.info(f"Found {len(grn_items)} GRN items using invoice number matching")
                return grn_items
        
        logger.warning(f"No GRN matches found for invoice {invoice.id}")
        return []

    async def _convert_invoice_to_complete_json(self, invoice: InvoiceData, invoice_items: List[InvoiceItemData]) -> Dict[str, Any]:
        """Convert complete invoice data to JSON for LLM"""
        
        invoice_json = {
            "invoice_header": {
                "id": invoice.id,
                "po_number": invoice.po_number or "",
                "grn_number": invoice.grn_number or "",
                "invoice_number": invoice.invoice_number or "",
                "invoice_date": str(invoice.invoice_date) if invoice.invoice_date else "",
                "vendor_name": invoice.vendor_name or "",
                "vendor_gst": invoice.vendor_gst or "",
                "vendor_pan": invoice.vendor_pan or "",
                "attachment_number": invoice.attachment_number or "",
                "attachment_url": invoice.attachment_url or "",
                "file_type": invoice.file_type or "",
                "processing_status": invoice.processing_status or "",
                "created_at": invoice.created_at.isoformat() if invoice.created_at else "",
                "updated_at": invoice.updated_at.isoformat() if invoice.updated_at else ""
            },
            "invoice_financial_data": {
                "invoice_value_without_gst": str(invoice.invoice_value_without_gst or 0),
                "cgst_rate": str(invoice.cgst_rate or 0),
                "cgst_amount": str(invoice.cgst_amount or 0),
                "sgst_rate": str(invoice.sgst_rate or 0),
                "sgst_amount": str(invoice.sgst_amount or 0),
                "igst_rate": str(invoice.igst_rate or 0),
                "igst_amount": str(invoice.igst_amount or 0),
                "total_gst_amount": str(invoice.total_gst_amount or 0),
                "invoice_total_post_gst": str(invoice.invoice_total_post_gst or 0)
            },
            "invoice_line_items": []
        }
        
        # Add all invoice line items
        for item in invoice_items:
            invoice_json["invoice_line_items"].append({
                "id": item.id,
                "item_sequence": item.item_sequence,
                "item_description": item.item_description or "",
                "hsn_code": item.hsn_code or "",
                "quantity": str(item.quantity or 0),
                "unit_of_measurement": item.unit_of_measurement or "",
                "unit_price": str(item.unit_price or 0),
                "invoice_value_item_wise": str(item.invoice_value_item_wise or 0),
                "cgst_rate": str(item.cgst_rate or 0),
                "cgst_amount": str(item.cgst_amount or 0),
                "sgst_rate": str(item.sgst_rate or 0),
                "sgst_amount": str(item.sgst_amount or 0),
                "igst_rate": str(item.igst_rate or 0),
                "igst_amount": str(item.igst_amount or 0),
                "total_tax_amount": str(item.total_tax_amount or 0),
                "item_total_amount": str(item.item_total_amount or 0),
                "po_number": item.po_number or "",
                "invoice_number": item.invoice_number or "",
                "vendor_name": item.vendor_name or ""
            })
        
        return invoice_json

    async def _convert_grn_to_complete_json(self, grn_items: List[ItemWiseGrn]) -> Dict[str, Any]:
        """Convert complete GRN data to JSON for LLM"""
        
        grn_json = {
            "grn_summary": {
                "total_grn_items": len(grn_items),
                "unique_grn_numbers": list(set(item.grn_no for item in grn_items if item.grn_no)),
                "unique_po_numbers": list(set(item.po_no for item in grn_items if item.po_no)),
                "unique_suppliers": list(set(item.supplier for item in grn_items if item.supplier)),
                "total_amount_sum": str(sum(Decimal(str(item.total or 0)) for item in grn_items)),
                "subtotal_sum": str(sum(Decimal(str(item.subtotal or 0)) for item in grn_items)),
                "cgst_sum": str(sum(Decimal(str(item.cgst_tax_amount or 0)) for item in grn_items)),
                "sgst_sum": str(sum(Decimal(str(item.sgst_tax_amount or 0)) for item in grn_items)),
                "igst_sum": str(sum(Decimal(str(item.igst_tax_amount or 0)) for item in grn_items))
            },
            "grn_line_items": []
        }
        
        # Add all GRN line items
        for item in grn_items:
            grn_json["grn_line_items"].append({
                "id": item.id,
                "s_no": item.s_no,
                "type": item.type or "",
                "sku_code": item.sku_code or "",
                "category": item.category or "",
                "sub_category": item.sub_category or "",
                "item_name": item.item_name or "",
                "unit": item.unit or "",
                "grn_no": item.grn_no or "",
                "hsn_no": item.hsn_no or "",
                "po_no": item.po_no or "",
                "remarks": item.remarks or "",
                "created_by": item.created_by or "",
                "grn_created_at": str(item.grn_created_at) if item.grn_created_at else "",
                "seller_invoice_no": item.seller_invoice_no or "",
                "supplier_invoice_date": str(item.supplier_invoice_date) if item.supplier_invoice_date else "",
                "supplier": item.supplier or "",
                "concerned_person": item.concerned_person or "",
                "pickup_location": item.pickup_location or "",
                "pickup_gstin": item.pickup_gstin or "",
                "pickup_code": item.pickup_code or "",
                "pickup_city": item.pickup_city or "",
                "pickup_state": item.pickup_state or "",
                "delivery_location": item.delivery_location or "",
                "delivery_gstin": item.delivery_gstin or "",
                "delivery_code": item.delivery_code or "",
                "delivery_city": item.delivery_city or "",
                "delivery_state": item.delivery_state or "",
                "price": str(item.price or 0),
                "received_qty": str(item.received_qty or 0),
                "returned_qty": str(item.returned_qty or 0),
                "discount": str(item.discount or 0),
                "tax": str(item.tax or 0),
                "sgst_tax": str(item.sgst_tax or 0),
                "sgst_tax_amount": str(item.sgst_tax_amount or 0),
                "cgst_tax": str(item.cgst_tax or 0),
                "cgst_tax_amount": str(item.cgst_tax_amount or 0),
                "igst_tax": str(item.igst_tax or 0),
                "igst_tax_amount": str(item.igst_tax_amount or 0),
                "cess": str(item.cess or 0),
                "subtotal": str(item.subtotal or 0),
                "vat_percent": item.vat_percent or "",
                "vat_amount": item.vat_amount or "",
                "item_tcs_percent": item.item_tcs_percent or "",
                "item_tcs_amount": str(item.item_tcs_amount or 0),
                "tax_amount": str(item.tax_amount or 0),
                "bill_tcs": str(item.bill_tcs or 0),
                "delivery_charges": str(item.delivery_charges or 0),
                "delivery_charges_tax_percent": str(item.delivery_charges_tax_percent or 0),
                "additional_charges": str(item.additional_charges or 0),
                "inv_discount": str(item.inv_discount or 0),
                "round_off": str(item.round_off or 0),
                "total": str(item.total or 0),
                "attachment_upload_date": str(item.attachment_upload_date) if item.attachment_upload_date else "",
                "attachment_1": item.attachment_1 or "",
                "attachment_2": item.attachment_2 or "",
                "attachment_3": item.attachment_3 or "",
                "attachment_4": item.attachment_4 or "",
                "attachment_5": item.attachment_5 or "",
                "upload_batch_id": item.upload_batch_id or "",
                "uploaded_filename": item.uploaded_filename or "",
                "created_at": item.created_at.isoformat() if item.created_at else "",
                "updated_at": item.updated_at.isoformat() if item.updated_at else "",
                "extracted_data": item.extracted_data
            })
        
        return grn_json

    async def _llm_complete_json_analysis(self, invoice_json: Dict[str, Any], grn_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """LLM analysis with complete Invoice JSON and GRN JSON"""
        
        try:
            # Create comprehensive prompt with complete JSON data
            prompt = f"""
You are an expert invoice-GRN reconciliation analyst. You have been provided with COMPLETE JSON data for both Invoice and GRN records.

FIELD MAPPING REFERENCE (for your analysis):
- po_number ↔ po_no
- grn_number ↔ grn_no  
- invoice_number ↔ seller_invoice_no
- vendor_name ↔ pickup_location
- vendor_gst ↔ pickup_gstin
- invoice_date ↔ supplier_invoice_date
- invoice_value_without_gst ↔ subtotal (sum of all GRN items)
- cgst_amount ↔ cgst_tax_amount (sum of all GRN items)
- sgst_amount ↔ sgst_tax_amount (sum of all GRN items)
- igst_amount ↔ igst_tax_amount (sum of all GRN items)
- invoice_total_post_gst ↔ total (sum of all GRN items)
- Date validation: invoice_date <= grn_created_at

COMPLETE INVOICE JSON DATA:
{json.dumps(invoice_json, indent=2)}

COMPLETE GRN JSON DATA:
{json.dumps(grn_json, indent=2)}

ANALYSIS TASKS:
1. **Header Matching**: Compare PO numbers, GRN numbers, invoice numbers, vendor details, dates
2. **Financial Matching**: Compare all amounts using the field mappings above
3. **Line Item Analysis**: Compare invoice line items with GRN line items
4. **Date Validation**: Ensure invoice_date <= grn_created_at
5. **Overall Assessment**: Determine match quality and identify discrepancies

REQUIRED OUTPUT FORMAT (JSON):
{{
    "header_comparison": {{
        "po_match": {{"match": true/false, "invoice_value": "...", "grn_value": "...", "notes": "..."}},
        "grn_match": {{"match": true/false, "invoice_value": "...", "grn_value": "...", "notes": "..."}},
        "invoice_number_match": {{"match": true/false, "invoice_value": "...", "grn_value": "...", "notes": "..."}},
        "vendor_match": {{"match": true/false, "invoice_value": "...", "grn_value": "...", "notes": "..."}},
        "gst_match": {{"match": true/false, "invoice_value": "...", "grn_value": "...", "notes": "..."}},
        "date_validation": {{"valid": true/false, "invoice_date": "...", "grn_dates": [...], "notes": "..."}}
    }},
    "financial_comparison": {{
        "subtotal_comparison": {{"variance_amount": 0.00, "variance_pct": 0.00, "invoice_value": 0.00, "grn_value": 0.00, "match": true/false}},
        "cgst_comparison": {{"variance_amount": 0.00, "variance_pct": 0.00, "invoice_value": 0.00, "grn_value": 0.00, "match": true/false}},
        "sgst_comparison": {{"variance_amount": 0.00, "variance_pct": 0.00, "invoice_value": 0.00, "grn_value": 0.00, "match": true/false}},
        "igst_comparison": {{"variance_amount": 0.00, "variance_pct": 0.00, "invoice_value": 0.00, "grn_value": 0.00, "match": true/false}},
        "total_comparison": {{"variance_amount": 0.00, "variance_pct": 0.00, "invoice_value": 0.00, "grn_value": 0.00, "match": true/false}}
    }},
    "line_items_analysis": {{
        "invoice_items_count": 0,
        "grn_items_count": 0,
        "items_matched": 0,
        "items_missing": 0,
        "discrepancies": [
            {{"type": "...", "description": "...", "impact": "..."}}
        ]
    }},
    "discrepancies": [
        {{
            "field": "...",
            "type": "MISSING/MISMATCH/EXTRA/AMOUNT_VARIANCE/DATE_ISSUE",
            "invoice_value": "...",
            "grn_value": "...",
            "impact": "HIGH/MEDIUM/LOW",
            "suggestion": "..."
        }}
    ],
    "overall_assessment": {{
        "overall_match_status": "perfect_match/partial_match/amount_mismatch/vendor_mismatch/date_mismatch/no_match",
        "confidence_score": 0.0,
        "total_discrepancies": 0,
        "critical_issues": 0,
        "recommendation": "APPROVE/REVIEW/REJECT",
        "summary": "Brief summary of the reconciliation results"
    }}
}}

Perform comprehensive analysis and return ONLY the JSON response.
"""
            
            # Call LLM with delay
            logger.info("Sending complete JSON data to LLM for analysis...")
            response = await self._invoke_llm_with_retry_and_delay(prompt)

            # Log the raw LLM response for debugging
            po_number = invoice_json.get('invoice_header', {}).get('po_number', '')
            logger.warning(f"[PO: {po_number}] LLM raw response: {response}")
            
            # Parse the response
            analysis_result = self._parse_llm_json_response(response)
            
            logger.info(f"LLM analysis completed. Success: {analysis_result.get('success', False)}")
            return analysis_result
        
        except Exception as e:
            logger.error(f"LLM complete JSON analysis failed: {str(e)}")
            return {
                'success': False,
                'error': f"LLM analysis failed: {str(e)}",
                'overall_match_status': 'no_analysis'
            }

    async def _invoke_llm_with_retry_and_delay(self, prompt: str) -> str:
        """LLM call with retry logic and delay"""
        max_retries = 3
        attempt = 0
        
        while attempt < max_retries:
            try:
                # Apply delay before each call
                if self.delay_seconds > 0:
                    await asyncio.sleep(self.delay_seconds)
                
                # Run LLM call in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.llm.invoke, prompt)
                return result
                
            except Exception as e:
                err_msg = str(e)
                if '429' in err_msg and 'retry_delay' in err_msg:
                    import re
                    match = re.search(r'retry_delay[":\s]*([0-9.]+)', err_msg)
                    if match:
                        retry_delay = float(match.group(1))
                        logger.warning(f"Rate limit hit, waiting {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        attempt += 1
                        continue
                raise
        
        raise Exception(f"Max retries reached for LLM call")

    def _parse_llm_json_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Clean the response
            response_text = llm_response.strip()
            
            # Remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            analysis_data = json.loads(response_text)
            
            # Add success flag
            analysis_data['success'] = True
            analysis_data['raw_response'] = llm_response
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in LLM response: {e}")
            logger.error(f"Raw response (first 500 chars): {llm_response[:500]}...")
            return {
                'success': False,
                'error': f"Failed to parse LLM JSON response: {str(e)}",
                'overall_match_status': 'no_analysis',
                'raw_response': llm_response
            }

    async def _create_reconciliation_from_llm_analysis(self, invoice: InvoiceData, grn_items: List[ItemWiseGrn],
                                                  invoice_json: Dict[str, Any], grn_json: Dict[str, Any], 
                                                  llm_analysis: Optional[Dict[str, Any]]) -> InvoiceGrnReconciliation:
        """Create reconciliation record from LLM analysis results"""

        # Extract data from LLM analysis
        if llm_analysis and llm_analysis.get('success'):
            header_comp = llm_analysis.get('header_comparison', {})
            financial_comp = llm_analysis.get('financial_comparison', {})
            overall_assessment = llm_analysis.get('overall_assessment', {})

            # Extract match status and details
            match_status = overall_assessment.get('overall_match_status', 'partial_match')
            vendor_match = header_comp.get('vendor_match', {}).get('match', False)
            gst_match = header_comp.get('gst_match', {}).get('match', False)
            date_valid = header_comp.get('date_validation', {}).get('valid', False)

            # Extract all financial variances safely
            def get_variance(comp_data, key):
                section = financial_comp.get(key, {})
                return section.get('variance_amount', 0), section.get('variance_pct', 0)

            subtotal_var, subtotal_var_pct = get_variance(financial_comp, 'subtotal_comparison')
            cgst_var, cgst_var_pct = get_variance(financial_comp, 'cgst_comparison')
            sgst_var, sgst_var_pct = get_variance(financial_comp, 'sgst_comparison')
            igst_var, igst_var_pct = get_variance(financial_comp, 'igst_comparison')
            total_var, total_var_pct = get_variance(financial_comp, 'total_comparison')

            # Optional: override incorrect mismatch if total variance is basically zero
            if match_status == 'amount_mismatch' and abs(total_var) < 1e-6:
                match_status = 'perfect_match'

            # Create comprehensive reconciliation notes
            discrepancies = llm_analysis.get('discrepancies', [])
            summary = overall_assessment.get('summary', '')
            reconciliation_notes = f"LLM Analysis: {len(discrepancies)} discrepancies found. {summary}"

        else:
            # Fallback values if LLM analysis failed
            match_status = 'no_analysis'
            vendor_match = False
            gst_match = False
            date_valid = False
            subtotal_var = cgst_var = sgst_var = igst_var = total_var = 0
            subtotal_var_pct = total_var_pct = 0
            reconciliation_notes = "LLM analysis failed or unavailable"

        # Get basic aggregated data
        grn_numbers = list(set(item.grn_no for item in grn_items if item.grn_no))
        grn_vendors = list(set(item.pickup_location for item in grn_items if item.pickup_location))
        grn_gstins = list(set(item.pickup_gstin for item in grn_items if item.pickup_gstin))

        # Prepare reconciliation data
        reconciliation_data = {
            'invoice_data': invoice,
            'po_number': invoice.po_number or '',
            'grn_number': grn_numbers[0] if grn_numbers else '',
            'invoice_number': invoice.invoice_number or '',
            'match_status': match_status,

            # Vendor validation from LLM
            'vendor_match': vendor_match,
            'invoice_vendor': invoice.vendor_name or '',
            'grn_vendor': ', '.join(grn_vendors) if grn_vendors else '',

            # GST validation from LLM
            'gst_match': gst_match,
            'invoice_gst': invoice.vendor_gst or '',
            'grn_gst': ', '.join(grn_gstins) if grn_gstins else '',

            # Date validation from LLM
            'date_valid': date_valid,
            'invoice_date': invoice.invoice_date,
            'grn_date': max([item.grn_created_at for item in grn_items if item.grn_created_at], default=None),

            # Financial amounts (from original data)
            'invoice_subtotal': float(invoice.invoice_value_without_gst or 0),
            'invoice_cgst': float(invoice.cgst_amount or 0),
            'invoice_sgst': float(invoice.sgst_amount or 0),
            'invoice_igst': float(invoice.igst_amount or 0),
            'invoice_total': float(invoice.invoice_total_post_gst or 0),

            # GRN aggregated amounts
            'grn_subtotal': float(sum(Decimal(str(item.subtotal or 0)) for item in grn_items)),
            'grn_cgst': float(sum(Decimal(str(item.cgst_tax_amount or 0)) for item in grn_items)),
            'grn_sgst': float(sum(Decimal(str(item.sgst_tax_amount or 0)) for item in grn_items)),
            'grn_igst': float(sum(Decimal(str(item.igst_tax_amount or 0)) for item in grn_items)),
            'grn_total': float(sum(Decimal(str(item.total or 0)) for item in grn_items)),

            # Variances
            'subtotal_variance': float(subtotal_var),
            'cgst_variance': float(cgst_var),
            'sgst_variance': float(sgst_var),
            'igst_variance': float(igst_var),
            'subtotal_variance_pct': float(subtotal_var_pct),
            'total_variance': float(total_var),
            'total_variance_pct': float(total_var_pct),

            # Summary info
            'total_grn_line_items': len(grn_items),
            'matching_method': 'llm_complete_json_analysis',
            'is_auto_matched': True,
            'reconciliation_notes': reconciliation_notes
        }

        # Store assessment summary inside notes (optional but useful)
        if llm_analysis and llm_analysis.get('success'):
            reconciliation_data['reconciliation_notes'] += f"\n\nLLM Summary:\n{json.dumps(overall_assessment, indent=2)}"

        # Create and return the record
        reconciliation = await sync_to_async(InvoiceGrnReconciliation.objects.create)(**reconciliation_data)

        logger.info(f"Created LLM-based reconciliation record {reconciliation.id} for invoice {invoice.id}")
        return reconciliation


    def _update_statistics(self, match_status: str):
        """Update processing statistics"""
        if match_status == 'perfect_match':
            self.stats['perfect_matches'] += 1
        elif match_status == 'partial_match':
            self.stats['partial_matches'] += 1
        elif match_status == 'amount_mismatch':
            self.stats['amount_mismatches'] += 1
        elif match_status == 'vendor_mismatch':
            self.stats['vendor_mismatches'] += 1
        elif match_status == 'date_mismatch':
            self.stats['date_mismatches'] += 1

    async def _create_no_match_record(self, invoice: InvoiceData) -> Dict[str, Any]:
        """Create no-match record"""
        reconciliation_data = {
            'invoice_data': invoice,
            'po_number': invoice.po_number or '',
            'invoice_number': invoice.invoice_number or '',
            'match_status': 'no_grn_found',
            'total_grn_line_items': 0,
            'is_auto_matched': True,
            'matching_method': 'llm_complete_json_analysis',
            'reconciliation_notes': 'No matching GRN records found for LLM analysis'
        }
        
        reconciliation = await sync_to_async(InvoiceGrnReconciliation.objects.create)(**reconciliation_data)
        
        return {
            'invoice_id': invoice.id,
            'reconciliation_id': reconciliation.id,
            'match_status': 'no_grn_found',
            'grn_items_matched': 0,
            'llm_analysis_success': False
        }


# Main function to run LLM-only reconciliation
async def run_llm_only_reconciliation(invoice_ids: List[int] = None, delay_seconds: float = 1.0, 
                                    max_concurrent: int = 10, batch_size: int = 100) -> Dict[str, Any]:
    """
    Main function to run LLM-only reconciliation with complete JSON data
    
    Args:
        invoice_ids: Optional list of invoice IDs
        delay_seconds: Delay between LLM calls (default 1.0)
        max_concurrent: Max concurrent processes (default 10)
        batch_size: Batch size for processing (default 100)
    """
    processor = LLMOnlyReconciliationProcessor(
        delay_seconds=delay_seconds,
        max_concurrent=max_concurrent
    )
    
    return await processor.process_batch_async(
        invoice_ids=invoice_ids,
        batch_size=batch_size
    )