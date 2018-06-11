"""Interface for gunicorn.

Run a server locally with:
> cd humantree
> gunicorn wsgi:app.server
"""
from server import app

if __name__ == "__main__":
    app.run()
