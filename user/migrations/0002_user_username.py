# Generated by Django 5.0.4 on 2024-05-19 05:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='username',
            field=models.CharField(max_length=255, null=True, unique=True),
        ),
    ]
