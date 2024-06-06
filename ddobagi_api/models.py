import os
import uuid
from django.utils import timezone

from django.db import models
from django.conf import settings


# Create your models here.
class Level(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()

    def __str__(self):
        return self.name


class Category(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    description = models.TextField()

    def __str__(self):
        return self.name


class Question(models.Model):
    level = models.ForeignKey(Level, on_delete=models.CASCADE)
    question_number = models.IntegerField()
    word = models.CharField(max_length=255)
    correct_pronounciation = models.CharField(max_length=255)
    incorrect_pronounciation = models.CharField(max_length=255)
    grammar_class = models.ForeignKey(Category, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"{self.level.name} - {self.question_number}: {self.word}"


class UserSolve(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    solved = models.BooleanField(default=False)
    accuracy = models.FloatField(default=0)
    updated_at = models.DateTimeField(auto_now=True)


class Feedback(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    accuracy = models.FloatField()
    chosen_pronounciation = models.CharField(max_length=255)
    correct_pronounciation_graph = models.ImageField(upload_to="graphs/")
    user_pronounciation_graph = models.ImageField(upload_to="graphs/")

    def __str__(self):
        return f"{self.user.username} - {self.question.word} - Accuracy: {self.accuracy}"


class UploadFile(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
