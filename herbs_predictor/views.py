from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse
from herbs_predictor import herb_prediction


@csrf_exempt
def herbsApi(request):
    if request.method == 'POST':
        symptom_data = JSONParser().parse(request)
        print(symptom_data)
        herb = herb_prediction.seeHerb(symptom_data)
        return JsonResponse(herb,safe=False)
