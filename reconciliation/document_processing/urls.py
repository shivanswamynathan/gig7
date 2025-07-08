from django.urls import path
from .views import views,po_grn_views,itemwise_grn_views,attachment_api_views,invoice_recon_views,invoice_recon_logical_views

app_name = 'document_processing'

urlpatterns = [
    path('api/process-invoice/', views.ProcessInvoiceAPI.as_view(), name='process_invoice'),

    # PO-GRN data processing
    path('api/process-po-grn/', po_grn_views.ProcessPoGrnAPI.as_view(), name='process_po_grn'),

    # Item-wise GRN data processing
    path('api/process-itemwise-grn/', itemwise_grn_views.ProcessItemWiseGrnAPI.as_view(), name='process_itemwise_grn'),

    # Process ItemWiseGRN file and automatically extract attachments
    path('api/process-grn-file-and-attachments/', attachment_api_views.ProcessItemWiseGRNAndAttachmentsAPI.as_view(), name='process_grn_file_and_attachments'),

    # Process attachments from GRN table
    path('api/process-attachments-from-grn-table/', attachment_api_views.ProcessAttachmentsFromGrnTableAPI.as_view(), name='process_attachments_from_grn_table'),

    path('api/llm-reconciliation/', invoice_recon_views.LLMReconciliationAPI.as_view(), name='llm_reconciliation'),

    path('api/reconciliation/', invoice_recon_logical_views.LLMReconciliationAPI.as_view(), name='reconciliation'),
]