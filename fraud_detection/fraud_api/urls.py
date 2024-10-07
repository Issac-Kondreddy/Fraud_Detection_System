from django.urls import path
from .views import index, predict_fraud, predict_batch

urlpatterns = [
    path('', index, name='index'),  # Frontend root
    path('predict_fraud/', predict_fraud, name='predict_fraud'),  # Single transaction
    path('predict_batch/', predict_batch, name='predict_batch'),  # Batch prediction
]
