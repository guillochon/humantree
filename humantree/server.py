"""Basic Flask server for HumanTree.org."""
import os
import json
import logging

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

logger = logging.getLogger('gunicorn.error')
logger.setLevel(logging.INFO)


@app.route('/')
def index():
    """Return the index page."""
    return render_template('index.html', places_api_key=ht._google_key)


@app.route('/help', methods=['GET', 'POST'])
def help():
    """Return the help us page."""
    import requests
    import time
    from glob import glob

    """Process captcha."""
    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0]
    else:
        ip = request.remote_addr
    logger.info('IP: {}'.format(ip))
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
        logger.info(address)
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
        try:
            eprice = ht.get_electricity_price(state) / 100
        except Exception:
            eprice = None
        try:
            hdd = ht.get_degree_days(state, 'heating')
            cdd = ht.get_degree_days(state, 'cooling')
        except Exception:
            hdd = None
            cdd = None

        house_value = 0.0
        value_increase = 0.0
        one_tree = 0.0
        one_tree_frac = 0.05
        try:
            zadd = address.replace(', USA', '')
            zill = ht.get_zillow(zadd, ht.get_zip(address))
            # upd = ht.get_updated_prop_details(zill.zpid)
        except Exception:
            sqft = 1000.0
        else:
            sqft = (
                float(
                    zill.extended_data.finished_sqft) if (
                        zill.has_extended_data and
                        zill.extended_data.finished_sqft is not None)
                else 1000.0)

            one_tree_frac = 0.05

            if zill.zestimate.amount is not None:
                house_value = float(zill.zestimate.amount)
                max_value_increase = 0.1 * house_value
                value_increase = max_value_increase * fraction
                one_tree_value = max_value_increase * one_tree_frac

        # Gross approximation: kwh usage = 0.5 * dd.
        if hdd is not None and cdd is not None and eprice is not None:
            cost = 0.6 * (hdd + cdd) * eprice * sqft / 1000.0
        else:
            cost = 0.0

        # Savings: Assume 0.66 * 365 * 5 dd for full tree coverage.
        # Explanation: 66% of days require either heating or cooling, trees
        # heat/cool 5 degrees in area around them.
        if eprice is not None:
            max_savings = 0.66 * 365 * 5 * eprice
            savings = max_savings * fraction
            one_tree_savings = max_savings * one_tree_frac
        else:
            savings = 0.0
            max_savings = 0.0
            one_tree_savings = 0.0

        max_noise_abatement = 10
        noise_abatement = max_noise_abatement * np.log10(1.0 + 9.0 * fraction)
        one_tree_noise = max_noise_abatement * np.log10(1.0 + 9.0 * one_tree_frac)

        return json.dumps({
            'fraction': fraction,
            'cost': cost,
            'savings': savings,
            'one_tree_savings': one_tree_savings,
            'max_savings': max_savings,
            'house_value': house_value,
            'value_increase': value_increase,
            'max_value_increase': max_value_increase,
            'one_tree_value': one_tree_value,
            'noise_abatement': noise_abatement,
            'max_noise_abatement': max_noise_abatement,
            'one_tree_noise': one_tree_noise
        })


@app.route('/favicon.ico')
def favicon():
    """Return favicon."""
    return send_from_directory(
        os.path.join(app.root_path, 'static'), 'favicon.ico',
        mimetype='image/vnd.microsoft.icon')


# This route ignored by nginx, only used locally.
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
