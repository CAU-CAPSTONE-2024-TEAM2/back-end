from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from ddobagi_api.models import *
from ddobagi_api.serializers import *


class QuestionViewSet(viewsets.ModelViewSet):
    queryset = Question.objects.all()
    serializer_class = QuestionSerializer

    def get_queryset(self):
        level_id = self.request.query_params.get('level', None)
        print("level_id", level_id)
        if level_id is not None:
            return self.queryset.filter(level_id=level_id)
        return self.queryset


class FeedbackViewSet(viewsets.ModelViewSet):
    queryset = Feedback.objects.all()
    serializer_class = FeedbackSerializer


class UserProgressAPIView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        levels = Level.objects.all()
        serializer = LevelProgressSerializer(levels, many=True, context={'request': request})
        return Response(serializer.data)


class FileUploadAPIView(APIView):
    permission_classes = (IsAuthenticated,)
    parser_classes = (MultiPartParser, FormParser)
    serializer = FileUploadSerializer

    def post(self, request):
        serializer = self.serializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
