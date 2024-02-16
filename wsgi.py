from app_app import app  # Replace 'your_flask_app' with the actual name of your Flask app
from gunicorn.app.base import BaseApplication
from six import iteritems

class StandaloneApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(StandaloneApplication, self).__init__()

    def load_config(self):
        config = {key: value for key, value in iteritems(self.options) if key in self.cfg.settings and value is not None}
        for key, value in iteritems(config):
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == '__main__':
    options = {
        'workers': 4,
    }
    StandaloneApplication(app, options).run()
