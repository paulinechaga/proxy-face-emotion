# from django.urls import re_path
# from . import detect

# websocket_urlpatterns = [
#     re_path(r'ws/emotion_detection/$', detect.EmotionDetectionConsumer.as_asgi()),
# ]
# routing.py

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path
from . import detect

websocket_urlpatterns = [
    path('ws/emotion_detection/', detect.EmotionDetectionConsumer.as_asgi()),
]

application = ProtocolTypeRouter({
    'websocket': AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})
