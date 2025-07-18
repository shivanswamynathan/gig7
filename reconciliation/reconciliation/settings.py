"""
Django settings for reconciliation project.

Generated by 'django-admin startproject' using Django 5.2.3.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.2/ref/settings/
"""

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-hy&@#2*#huv0rubixkshqpwf9*$v7ee#4e9uh(b+42r#j@)3^='

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["*", "localhost"]

# Add media file handling
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'


import os
from pathlib import Path
import logging
from dotenv import load_dotenv
load_dotenv() 

# Configure early logging to capture database connection attempts
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / 'database_connection.log')
    ]
)

logger = logging.getLogger(__name__)

# Database connection logging
logger.info("=" * 80)
logger.info("STARTING DATABASE CONFIGURATION")
logger.info("=" * 80)


# Log environment file location
env_file_path = BASE_DIR.parent / '.env'
logger.info(f"Looking for .env file at: {env_file_path}")
logger.info(f".env file exists: {env_file_path.exists()}")

if env_file_path.exists():
    with open(env_file_path, 'r') as f:
        env_content = f.read()
        logger.info("Contents of .env file:")
        for line_num, line in enumerate(env_content.split('\n'), 1):
            if line.strip() and not line.startswith('#'):
                if 'PASSWORD' in line:
                    # Mask password for security
                    logger.info(f"  Line {line_num}: {line.split('=')[0]}=***MASKED***")
                else:
                    logger.info(f"  Line {line_num}: {line}")

# Get database configuration with detailed logging
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

logger.info("DATABASE ENVIRONMENT VARIABLES:")
logger.info(f"  DB_NAME: '{DB_NAME}' (type: {type(DB_NAME)})")
logger.info(f"  DB_USER: '{DB_USER}' (type: {type(DB_USER)})")
logger.info(f"  DB_PASSWORD: {'***SET***' if DB_PASSWORD else 'NOT SET'} (length: {len(DB_PASSWORD) if DB_PASSWORD else 0})")
logger.info(f"  DB_HOST: '{DB_HOST}' (type: {type(DB_HOST)})")
logger.info(f"  DB_PORT: '{DB_PORT}' (type: {type(DB_PORT)})")

# Check for missing values
missing_vars = []
if not DB_NAME:
    missing_vars.append('DB_NAME')
if not DB_USER:
    missing_vars.append('DB_USER')
if not DB_PASSWORD:
    missing_vars.append('DB_PASSWORD')
if not DB_HOST:
    missing_vars.append('DB_HOST')
if not DB_PORT:
    missing_vars.append('DB_PORT')

if missing_vars:
    logger.error(f"MISSING ENVIRONMENT VARIABLES: {', '.join(missing_vars)}")
else:
    logger.info("ALL DATABASE ENVIRONMENT VARIABLES ARE SET")



# Google AI Configuration
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

logger.info("GOOGLE AI CONFIGURATION:")
logger.info(f"  GEMINI_MODEL: '{GEMINI_MODEL}'")
logger.info(f"  GOOGLE_API_KEY: {'***SET***' if GOOGLE_API_KEY else 'NOT SET'}")



# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'document_processing',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'reconciliation.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'reconciliation.wsgi.application'



# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases


logger.info("CONFIGURING DATABASE CONNECTION:")

# Fallback values for development
DB_NAME_FINAL = DB_NAME or 'recon_db'
DB_USER_FINAL = DB_USER or 'postgres'
DB_PASSWORD_FINAL = DB_PASSWORD or 'Entrans'
DB_HOST_FINAL = DB_HOST or 'localhost'
DB_PORT_FINAL = DB_PORT or '5432'

logger.info("FINAL DATABASE CONFIGURATION (after fallbacks):")
logger.info(f"  Database Name: '{DB_NAME_FINAL}'")
logger.info(f"  Username: '{DB_USER_FINAL}'")
logger.info(f"  Host: '{DB_HOST_FINAL}'")
logger.info(f"  Port: '{DB_PORT_FINAL}'")

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST'),
        'PORT': os.getenv('DB_PORT'),
        'CONN_MAX_AGE': 600,
    }
}

logger.info("DATABASE DICTIONARY CREATED:")
logger.info(f"  ENGINE: {DATABASES['default']['ENGINE']}")
logger.info(f"  NAME: {DATABASES['default']['NAME']}")
logger.info(f"  USER: {DATABASES['default']['USER']}")
logger.info(f"  HOST: {DATABASES['default']['HOST']}")
logger.info(f"  PORT: {DATABASES['default']['PORT']}")

""" 
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
"""


# Database connection pooling 
DATABASE_CONNECTION_POOLING = {
    'default': {
        'BACKEND': 'django_db_pool.backends.postgresql',
        'POOL_OPTIONS': {
            'INITIAL_CONNS': 1,
            'MAX_CONNS': 20,
            'MIN_CACHED_CONNS': 0,
            'MAX_CACHED_CONNS': 50,
            'MAX_LIFETIME': 3600,
        }
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Logging configuration for debugging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[{levelname}] {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '[{levelname}] {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'invoice_processing.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'db_file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'database_operations.log',
            'formatter': 'verbose',
        },
        'itemwise_grn_file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'itemwise_grn_processing.log',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'document_processing': {
            'handlers': ['file', 'console', 'db_file','itemwise_grn_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'document_processing.reconciliation': {
        'handlers': ['file', 'console'],  
        'level': 'INFO',
        'propagate': False,
        },
        'django.db.backends': {
            'handlers': ['db_file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}


# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB

# Additional settings for large file processing
DATA_UPLOAD_MAX_NUMBER_FIELDS = 10000  