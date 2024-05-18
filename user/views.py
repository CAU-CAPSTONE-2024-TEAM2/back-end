import os
from json import JSONDecodeError
from django.http import JsonResponse
import requests
from rest_framework import status
from django.shortcuts import render, redirect
from dj_rest_auth.registration.views import SocialLoginView
from allauth.socialaccount.providers.kakao import views as kakao_view
from allauth.socialaccount.providers.oauth2.client import OAuth2Client

BASE_URL = 'http://localhost:8000/'
KAKAO_CALLBACK_URI = BASE_URL + 'user/kakao/callback/'


def kakao_login(request):
    client_id = os.environ.get('SOCIAL_AUTH_KAKAO_CLIENT_ID')
    return redirect(
        f"https://kauth.kakao.com/oauth/authorize?client_id={client_id}&redirect_uri={KAKAO_CALLBACK_URI}&response_type=code")


def kakao_callback(request):
    client_id = os.environ.get('SOCIAL_AUTH_KAKAO_CLIENT_ID')
    code = request.GET.get('code')

    token_request = requests.get(
        f"https://kauth.kakao.com/oauth/token?grant_type=authorization_code&client_id={client_id}&redirect_uri={KAKAO_CALLBACK_URI}&code={code}")
    token_response_json = token_request.json()

    error = token_response_json.get("error", None)
    if error is not None:
        raise JSONDecodeError(error)

    access_token = token_response_json.get("access_token")

    data = {'access_token': access_token, 'code': code}
    accept = requests.post(f"{BASE_URL}user/kakao/login/finish/", data=data)
    accept_status = accept.status_code

    if accept_status != 200:
        return JsonResponse({'err_msg': 'failed to signin'}, status=accept_status)

    accept_json = accept.json()
    accept_json.pop('user', None)
    return JsonResponse(accept_json)


class KakaoLogin(SocialLoginView):
    adapter_class = kakao_view.KakaoOAuth2Adapter
    callback_url = KAKAO_CALLBACK_URI
    client_class = OAuth2Client
