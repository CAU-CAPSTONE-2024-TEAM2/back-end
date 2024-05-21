from django.urls import path, include
from rest_framework.routers import DefaultRouter
from ddobagi_api.views import *

router = DefaultRouter()
router.register(r'questions', QuestionViewSet, basename='questions')
router.register(r'feedbacks', FeedbackViewSet, basename='feedbacks')

urlpatterns = [
    path('', include(router.urls)),
    path('levels/', UserProgressAPIView.as_view(), name='levels'),
    path('upload/', FileUploadAPIView.as_view(), name='upload'),
]