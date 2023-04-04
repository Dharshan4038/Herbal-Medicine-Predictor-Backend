from django.urls import re_path
from herbs_predictor import views

urlpatterns = [
    re_path(r'^herbs/$',views.herbsApi),
]
