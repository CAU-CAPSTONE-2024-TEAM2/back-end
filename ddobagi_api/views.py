import uuid

from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from ddobagi_api.models import *
from ddobagi_api.serializers import *
from ddobagi_api.prediction import *


class QuestionViewSet(viewsets.ModelViewSet):
    queryset = Question.objects.all()
    serializer_class = QuestionSerializer
    permission_classes = (IsAuthenticated,)

    def get_queryset(self):
        level_id = self.request.query_params.get('level', None)
        grammar_class_id = self.request.query_params.get('grammar_class', None)
        if level_id is not None:
            return self.queryset.filter(level_id=level_id)
        if grammar_class_id is not None:
            return self.queryset.filter(grammar_class_id=grammar_class_id)
        return self.queryset


class CategoryViewSet(viewsets.ModelViewSet):
    serializer_class = CategorySerializer
    queryset = Category.objects.all()

    def get_queryset(self):
        return self.queryset.all()


class FeedbackViewSet(viewsets.ModelViewSet):
    queryset = Feedback.objects.all()
    serializer_class = FeedbackSerializer


class UserProgressAPIView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        levels = Level.objects.all()
        serializer = LevelProgressSerializer(levels, many=True, context={'request': request})
        return Response(serializer.data)


class UserGrammarClassProgressAPIView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        categories = Category.objects.all()
        serializer = GrammarProgressSerializer(categories, many=True, context={'request': request})
        return Response(serializer.data)


class FileUploadAPIView(APIView):
    permission_classes = (IsAuthenticated,)
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = FileUploadSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            file = serializer.save()
            file_path = file.file.path
            question_id = file.question_id
            question_word = Question.objects.get(id=question_id).word
            category_id = Question.objects.get(id=question_id).grammar_class_id
            mfcc_file = './data_augmented/' + question_word

            # 파일 경로
            mfcc_a_file = mfcc_file + '/A.pkl'
            mfcc_b_file = mfcc_file + '/B.pkl'
            voice_x_path = file_path

            # 데이터 로드
            max_len = 100
            mfcc_a_group = load_mfcc_from_file(mfcc_a_file, max_len)
            mfcc_b_group = load_mfcc_from_file(mfcc_b_file, max_len)
            mfcc_x = extract_mfcc(voice_x_path, max_len)

            # 훈련 데이터 생성
            mfcc_group = mfcc_a_group + mfcc_b_group
            y_train = [0] * len(mfcc_a_group) + [1] * len(mfcc_b_group)

            # 거리 행렬 계산
            distance_matrix = compute_distance_matrix(mfcc_group)

            # 새로운 샘플에 대한 거리 계산
            distance_to_x = compute_distance_to_sample(mfcc_group, mfcc_x)
            distance_to_a = compute_distance_to_sample(mfcc_a_group, mfcc_x)

            # K-NN 분류기 생성 및 학습
            knn = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
            knn.fit(distance_matrix, y_train)

            # 예측
            y_pred = knn.predict(distance_to_x.reshape(1, -1))
            prob = knn.predict_proba(distance_to_x.reshape(1, -1))

            # 유사도 점수 계산
            similarity_score = calculate_similarity_score(distance_to_a)

            print(f"Score: {similarity_score}")

            # 결과 출력
            if y_pred[0] == 0:
                print("Voice X is more similar to Voice Group A.")
                chosen_pronounciation = Question.objects.get(id=question_id).correct_pronounciation
                # new_solve = UserSolve(question_id=question_id, user=request.user, solved=True)
                # new_solve.save()
                UserSolve.objects.update_or_create(
                    question_id=question_id,
                    user=request.user,
                    solved=True,
                    defaults={
                        'accuracy': similarity_score,
                    }
                )
            else:
                print("Voice X is more similar to Voice Group B.")
                chosen_pronounciation = Question.objects.get(id=question_id).incorrect_pronounciation

            print(f"Probability: {prob[0]}")

            # 기준 MFCC 배열을 하나만 사용하여 그래프 저장
            new_uuid = uuid.uuid4()
            mfcc_a_filepath = f"media/graphs/{new_uuid}mfcc_a.png"
            mfcc_x_filepath = f"media/graphs/{new_uuid}mfcc_x.png"
            save_mfcc_graph(mfcc_a_group[0], 'Answer', mfcc_a_filepath)
            save_mfcc_graph(mfcc_x, 'User', mfcc_x_filepath)

            data = {
                "id": str(new_uuid),
                "accuracy": similarity_score,
                "answer": Question.objects.get(id=question_id).correct_pronounciation,
                "chosen_pronounciation": chosen_pronounciation,
                "correct_pronounciation_graph": mfcc_a_filepath,
                "user_pronounciation": mfcc_x_filepath,
                "feedback": Category.objects.get(id=category_id).description,
            }

            default_storage.delete(file.file.path)
            file.delete()

            return JsonResponse(data)

        print(serializer.errors)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
