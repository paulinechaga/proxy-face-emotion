"""
ASGI config for geeksforgeeks project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
"""

# import os

# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recogn.settings')

# application = get_asgi_application()
import os
import django
from channels.routing import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recogn.settings')
django.setup()
application = get_asgi_application()
