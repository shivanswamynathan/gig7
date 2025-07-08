from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    path('document-processing/', include('document_processing.urls')),
    
]