from rest_framework import serializers
from ddobagi_api.models import *


class QuestionSerializer(serializers.ModelSerializer):
    accuracy = serializers.SerializerMethodField()

    class Meta:
        model = Question
        fields = ['id', 'question_number', 'word', 'correct_pronounciation', 'incorrect_pronounciation', 'level', 'grammar_class', 'accuracy']

    def get_accuracy(self, obj):
        user = self.context['request'].user
        try:
            accuracy = UserSolve.objects.get(user=user, question=obj).accuracy
            return accuracy
        except UserSolve.DoesNotExist:
            return 0


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = '__all__'


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = Feedback
        fields = '__all__'


class LevelProgressSerializer(serializers.ModelSerializer):
    progress = serializers.SerializerMethodField()

    class Meta:
        model = Level
        fields = ['id', 'name', 'description', 'progress']

    def get_progress(self, obj):
        user = self.context['request'].user
        total_questions = Question.objects.filter(level_id=obj.id).count()
        solved_questions = UserSolve.objects.distinct().values_list('question_id', 'user_id').filter(user=user, question__level=obj, solved=True).count()
        if total_questions == 0:
            return 0

        return (solved_questions / total_questions) * 100


class GrammarProgressSerializer(serializers.ModelSerializer):
    progress = serializers.SerializerMethodField()

    class Meta:
        model = Category
        fields = ['id', 'name', 'progress']

    def get_progress(self, obj):
        user = self.context['request'].user
        total_questions = Question.objects.filter(grammar_class_id=obj.id).count()
        solved_questions = UserSolve.objects.distinct().values_list('question_id', 'user_id').filter(user=user, question__grammar_class=obj, solved=True).count()
        if total_questions == 0:
            return 0

        print('total_questions', total_questions)
        print('solved_questions', solved_questions)
        return (solved_questions / total_questions) * 100


class FileUploadSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadFile
        fields = ['id', 'file', 'question', 'uploaded_at']

    def create(self, validated_data):
        user = self.context['request'].user
        return UploadFile.objects.create(user=user, **validated_data)
