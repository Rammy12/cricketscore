from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('predict_t20_men/',views.predict_t20score,name='predict_t20score'),
    path('predict_ipl/',views.predict_iplscore,name='predict_iplscore'),
    path('predict_odismen/',views.predict_odiscore,name='predict_odiscore'),
]