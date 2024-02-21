from django.urls import path
from .views import TransformerLmView

urlpatterns = [
    path('', TransformerLmView.as_view(), name='transformer_lm'),
]