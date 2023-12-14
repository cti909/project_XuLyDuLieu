from django.urls import path, include
from . import views

urlpatterns = [
    # path('predict/', views.predict, name='predict'),
    # path('',views.demo, name = "demo"),
    path('', views.home, name="home"),
    path('login/', views.login_view, name="login"),
    path('signup/', views.signup, name="signup"),
    path('handle_login', views.handle_login, name="handle_login"),
    path('hanle_logout', views.handle_logout, name="handle_logout"),
    path('hanle_signup', views.handle_signup, name="handle_signup"),
    path('post', views.handle_view_post, name='post'),
    path('add-post/', views.handle_view_add_post, name="add-post"),
    # path('handle_post', views.handle_post, name='handle_post'),
    
    path('post/predict', views.predict_new_post, name='predict'),
    path('post/<int:id>', views.post_detail, name="post_detail")
]
