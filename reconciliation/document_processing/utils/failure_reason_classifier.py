def classify_failure_reason(file_type, invoice_number, vendor_gst, total_gst, invoice_total):
    """
    Classifies the failure reason for invoice extraction based on business rules.
    Returns one of: 'unsupported format', 'poor image quality', 'partial missing', or None.
    """
    supported_types = ['pdf_text', 'pdf_image', 'image']
    if file_type not in supported_types:
        return 'unsupported format'
    important_fields = [invoice_number, vendor_gst, total_gst, invoice_total]
    empty_fields = [f for f in important_fields if not f]
    if len(empty_fields) == 4:
        return 'poor image quality'
    elif len(empty_fields) >= 2:
        return 'partial missing'
    return None
