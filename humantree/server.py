"""Basic Flask server for HumanTree.org."""
import os
import json

import numpy as np

from flask import Flask, render_template, send_from_directory, request
from humantree import HumanTree

app = Flask(__name__)

# Define some globals here.
with open('../recaptcha.key', 'r') as f:
    secret = f.readline()

votes_path = os.path.join('..', 'votes.json')

if os.path.isfile(votes_path):
    with open(votes_path, 'r') as f:
        votes = json.load(f)
else:
    votes = {}

ips = {}


@app.route('/')
def index():
    """Return the index page."""
    return render_template('index.html')


@app.route('/help', methods=['GET', 'POST'])
def help():
    """Return the help us page."""
    import requests
    import time
    from glob import glob

    """Process captcha."""
    ip = request.remote_addr
    captcha = (time.time() - ips.get(ip, 0.0)) > 86400.

    response = None
    success = True
    if request.method == 'POST':
        if captcha:
            response = request.form.get('g-recaptcha-response')
            if response is not None:
                r = requests.post(
                    'https://www.google.com/recaptcha/api/siteverify', data={
                        'secret': secret,
                        'response': response,
                        'remoteip': ip
                    })
                resp = r.json()
                success = True if resp.get('success') else False
        id = request.form.get('mask-num')

        if success and id:
            if captcha:
                ips[ip] = time.time()
            if id not in votes:
                votes[id] = {'good': 0, 'okay': 0, 'bad': 0}
            if 'good' in request.form:
                votes[id]['good'] += 1
            elif 'okay' in request.form:
                votes[id]['okay'] += 1
            else:
                votes[id]['bad'] += 1
            with open(votes_path, 'w') as f:
                json.dump(votes, f)

    captcha = ip not in ips

    ids = ','.join(sorted([x.split('/')[
        -1].split('-')[0] for x in glob('parcels/*-outline.png')]))
    return render_template(
        'help.html', ids=ids, captcha=captcha, success=success)


@app.route('/about')
def about():
    """Return the about page."""
    return render_template('about.html')


@app.route('/method')
def method():
    """Return the method page."""
    return render_template('method.html')


@app.route('/process', methods=['GET', 'POST'])
def process():
    """Process an address."""
    global ht

    if request.method == 'POST':
        address = request.form.get('address')
        print(address)
        fpath = ht.get_image_from_address(address)
        if fpath is not None:
            return fpath.split('/')[-1]
        return ''


@app.route('/metrics', methods=['GET', 'POST'])
def metrics():
    """Return metrics."""
    global ht

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

        house_value = 0.0
        value_increase = 0.0
        try:
            zill = ht.get_zillow(address, ht.get_zip(address))
        except Exception:
            sqft = 1000.0
        else:
            sqft = (
                float(
                    zill.extended_data.finished_sqft) if zill.has_extended_data
                else 1000.0)

            if zill.zestimate.amount is not None:
                house_value = float(zill.zestimate.amount)
                value_increase = 0.1 * house_value * fraction

        # Gross approximation: kwh usage = 0.5 * dd.
        cost = 0.6 * (hdd + cdd) * eprice * sqft / 1000.0

        # Savings: Assume 0.66 * 365 * 5 dd for full tree coverage.
        # Explanation: 66% of days require either heating or cooling, trees
        # heat/cool 5 degrees in area around them.
        savings = 0.66 * 365 * 5 * eprice * fraction

        noise_abatement = 10 * fraction

        return json.dumps({
            'fraction': fraction,
            'cost': cost,
            'savings': savings,
            'house_value': house_value,
            'value_increase': value_increase,
            'noise_abatement': noise_abatement
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


def before_request():
    """Clear cache when HTML changes."""
    app.jinja_env.cache = {}


ht = HumanTree(load_canopy_polys=False)
if __name__ == '__main__':
    app.before_request(before_request)
    app.run_server(debug=True)
