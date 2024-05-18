from django.urls import path, include
from rest_framework_simplejwt.views import TokenRefreshView
from rest_framework import urls, routers

from user.views import *

app_name = 'user'


urlpatterns = [
    path('auth/refresh', TokenRefreshView.as_view(), name='auth_refresh'),
    path('auth/', include('dj_rest_auth.urls')),
    path('auth/registration/', include('dj_rest_auth.registration.urls')),
    path('kakao/login/', kakao_login, name='kakao_login'),
    path('kakao/callback/', kakao_callback, name='kakao_callback'),
    path('kakao/login/finish/', KakaoLogin.as_view(), name='kakao_login_finish'),
]