import os

class Config(object):
    """Parent configuration class."""
    DEBUG = False
    CSRF_ENABLED = True
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:postgres@localhost:5432/apitest'

class DevelopmentConfig(Config):
    """Configurations for Development."""
    DEBUG = False

class TestingConfig(Config):
    """Configurations for Testing, with a separate test database."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'postgresql://postgres:postgres@localhost:5432/apitest'
    DEBUG = True


app_config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
}
