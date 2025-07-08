import asyncio
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from document_processing.utils.invoice_recon import run_llm_only_reconciliation
from document_processing.models import InvoiceData, ItemWiseGrn
from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class LLMReconciliationAPI(View):
    """
    SINGLE LLM-ONLY Reconciliation Endpoint
    
    Passes complete Invoice JSON and ItemWise GRN JSON to LLM for analysis
    
    Field mappings provided to LLM:
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
    - Date validation: invoice_date <= grn_created_at
    """
    
    def post(self, request):
        """
        POST: Start LLM-only reconciliation with complete JSON data
        
        Request Body (JSON):
        {
            "invoice_ids": [1, 2, 3] (optional - if not provided, processes all),
            "max_concurrent": 5 (optional - default: 15),
            "batch_size": 50 (optional - default: 50),
            "delay_seconds": 0 (optional - default: 0, uses dynamic retry)
        }
        
        Note: delay_seconds is optional. The system uses automatic retry with 
        dynamic delays based on LLM API rate limiting responses.
        
        Response:
        {
            "success": true,
            "message": "LLM reconciliation completed",
            "data": {
                "processing_summary": {...},
                "match_statistics": {...},
                "field_mappings_used": {...}
            }
        }
        """
        return asyncio.run(self._async_post(request))
    
    async def _async_post(self, request):
        try:
            # Parse parameters
            if request.content_type == 'application/json':
                import json
                body = json.loads(request.body.decode('utf-8'))
                invoice_ids = body.get('invoice_ids', None)
                delay_seconds = float(body.get('delay_seconds', 0))  # Default to 0 - use dynamic retry
                max_concurrent = int(body.get('max_concurrent', 15))
                batch_size = int(body.get('batch_size', 50))
            else:
                # Form data support
                invoice_ids_str = request.POST.get('invoice_ids', None)
                if invoice_ids_str:
                    import json
                    invoice_ids = json.loads(invoice_ids_str)
                else:
                    invoice_ids = None
                delay_seconds = float(request.POST.get('delay_seconds', 0))  # Default to 0
                max_concurrent = int(request.POST.get('max_concurrent', 5))
                batch_size = int(request.POST.get('batch_size', 50))
            
            # Validate parameters (delay_seconds is optional now)
            if delay_seconds < 0 or delay_seconds > 10:
                return JsonResponse({
                    'success': False,
                    'error': 'delay_seconds must be between 0 and 10 (0 = use dynamic retry)'
                }, status=400)
            
            if max_concurrent < 1 or max_concurrent > 20:
                return JsonResponse({
                    'success': False,
                    'error': 'max_concurrent must be between 1 and 20'
                }, status=400)
            
            if batch_size < 5 or batch_size > 200:
                return JsonResponse({
                    'success': False,
                    'error': 'batch_size must be between 5 and 200'
                }, status=400)
            
            # Get data counts
            total_invoices = await sync_to_async(InvoiceData.objects.filter(processing_status='completed').count)()
            total_grn_items = await sync_to_async(ItemWiseGrn.objects.count)()
            
            if invoice_ids:
                invoices_to_process = len(invoice_ids)
            else:
                invoices_to_process = total_invoices
            
            logger.info(f"Starting LLM-ONLY reconciliation")
            logger.info(f"Data: {total_invoices} invoices, {total_grn_items} GRN items")
            logger.info(f"Settings: {invoices_to_process} to process, delay={delay_seconds}s (0=dynamic retry), concurrent={max_concurrent}")
            
            # Cost warning
            if invoices_to_process > 100:
                logger.warning(f"⚠️ Processing {invoices_to_process} invoices with LLM - consider API costs!")
            
            # Run LLM-only reconciliation
            result = await run_llm_only_reconciliation(
                invoice_ids=invoice_ids,
                delay_seconds=delay_seconds,
                max_concurrent=max_concurrent,
                batch_size=batch_size
            )
            
            if result['success']:
                return JsonResponse({
                    'success': True,
                    'message': f"✅ LLM reconciliation completed: {result['total_processed']} invoices processed",
                    'data': {
                        'processing_summary': {
                            'total_processed': result['total_processed'],
                            'invoices_available': total_invoices,
                            'grn_items_available': total_grn_items,
                            'llm_analyses_performed': result['stats'].get('llm_analyses', 0),
                            'success_rate': f"{result['total_processed']}/{invoices_to_process}"
                        },
                        'match_statistics': {
                            'perfect_matches': result['stats'].get('perfect_matches', 0),
                            'partial_matches': result['stats'].get('partial_matches', 0),
                            'amount_mismatches': result['stats'].get('amount_mismatches', 0),
                            'vendor_mismatches': result['stats'].get('vendor_mismatches', 0),
                            'date_mismatches': result['stats'].get('date_mismatches', 0),
                            'no_matches': result['stats'].get('no_matches', 0),
                            'errors': result['stats'].get('errors', 0)
                        },
                        'processing_config': {
                            'method': 'LLM Complete JSON Analysis',
                            'delay_seconds': delay_seconds,
                            'max_concurrent': max_concurrent,
                            'batch_size': batch_size
                        },
                        'field_mappings_used_by_llm': {
                            'po_number': 'po_no',
                            'grn_number': 'grn_no',
                            'invoice_number': 'seller_invoice_no',
                            'vendor_name': 'pickup_location',
                            'vendor_gst': 'pickup_gstin',
                            'invoice_date': 'supplier_invoice_date',
                            'invoice_value_without_gst': 'subtotal (sum)',
                            'cgst_amount': 'cgst_tax_amount (sum)',
                            'sgst_amount': 'sgst_tax_amount (sum)',
                            'igst_amount': 'igst_tax_amount (sum)',
                            'invoice_total_post_gst': 'total (sum)',
                            'date_validation': 'invoice_date <= grn_created_at'
                        },
                        'llm_json_structure': {
                            'invoice_sections_sent': [
                                'invoice_header (complete invoice data)',
                                'invoice_financial_data (all amounts)',
                                'invoice_line_items (all line items with details)'
                            ],
                            'grn_sections_sent': [
                                'grn_summary (aggregated data)',
                                'grn_line_items (all GRN items with complete details)'
                            ],
                            'llm_analysis_requested': [
                                'header_comparison',
                                'financial_comparison',
                                'line_items_analysis',
                                'discrepancies identification',
                                'overall_assessment'
                            ]
                        }
                    }
                }, status=200)
            else:
                return JsonResponse({
                    'success': False,
                    'error': f"❌ LLM reconciliation failed: {result['error']}",
                    'stats': result['stats']
                }, status=500)
                
        except Exception as e:
            logger.error(f"❌ Error in LLM reconciliation API: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'LLM reconciliation failed: {str(e)}'
            }, status=500)