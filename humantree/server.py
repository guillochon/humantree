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
        address = request.form.get('address')
        data = np.array([float(x) for x in image.split(',')])
        data[data < 255 / 2.0] = 0
        data[data > 255 / 2.0] = 1
        fraction = np.sum(1 - data) / (512.0 * 512)
        state = ht.get_state(address)
        eprice = ht.get_electricity_price(state) / 100
        hdd = ht.get_degree_days(state, 'heating')
        cdd = ht.get_degree_days(state, 'cooling')

        # Gross approximation: kwh usage = 0.5 * dd.
        cost = 0.5 * (hdd + cdd) * eprice

        # Savings: Assume 365 * 5 dd for full tree coverage.
        savings = 365 * 5 * eprice * fraction

        return json.dumps({
            'fraction': fraction,
            'cost': cost,
            'savings': savings
        })


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


ht = HumanTree(load_canopy_polys=False)
if __name__ == '__main__':
    app.run_server(debug=True)
