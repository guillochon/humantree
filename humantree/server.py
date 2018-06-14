"""Basic Flask server for HumanTree.org."""
import os

import numpy as np

from flask import Flask, render_template, send_from_directory
from humantree import HumanTree

app = Flask(__name__)


@app.route('/')
def index():
    """Return the index page."""
    return render_template('index.html')


@app.route('/process', methods=['GET', 'POST'])
def process():
    """Process request."""
    from flask import request

    global ht

    """Process an address."""
    if request.method == 'POST':
        address = request.form.get('address')
        print(address)
        return ht.get_image_from_address(address).split('/')[-1]


@app.route('/metrics', methods=['GET', 'POST'])
def metrics():
    """Return metrics."""
    from flask import request
    import json

    global ht

    """Process an address."""
    if request.method == 'POST':
        image = request.form.get('image')
        data = [float(x) for x in image.split(',')]
        fraction = np.sum(data) / (255 * 512.0 * 512)
        return json.dumps({'fraction': fraction})


@app.route('/favicon.ico')
def favicon():
    """Return favicon."""
    return send_from_directory(
        os.path.join(app.root_path, 'static'), 'favicon.ico',
        mimetype='image/vnd.microsoft.icon')


@app.route('/<path:path>')
def send_file(path):
    """Send a file to the user."""
    return send_from_directory('', path)


ht = HumanTree()
if __name__ == '__main__':
    app.run_server(debug=True)
