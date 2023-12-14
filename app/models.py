from django.db import models
from django.contrib.auth.models import User

class Users(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    address = models.CharField(max_length=255, default="")
    def __str__(self):
        return self.user.username
    
class Category(models.Model):
    name = models.CharField(max_length=255, default="")
    def __str__(self):
        return f"{self.name}"
    
class Post(models.Model):
    title = models.TextField(default="")
    abstract = models.TextField(default="")
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    label = models.ForeignKey(Category, on_delete=models.CASCADE)
    def __str__(self):
        return f"{self.label}"
    