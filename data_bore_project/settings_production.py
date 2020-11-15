from data_bore_project.settings import *

import dj_database_url
import django_heroku

DATABASES['default'] = dj_database_url.config()

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')

DEBUG = False

ALLOWED_HOSTS = ['*']

django_heroku.settings(locals())