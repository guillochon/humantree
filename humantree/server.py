"""Basic Flask server for HumanTree.org."""
import json
import logging
import os

import numpy as np

from flask import Flask, jsonify, render_template, request, send_from_directory
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
        radius = ht.get_address_radius(address)
        if fpath is not None:
            return jsonify({
                'path': fpath.split('/')[-1],
                'radius': radius
            })
        return jsonify({})


@app.route('/metrics', methods=['GET', 'POST'])
def metrics():
    """Return metrics."""
    import rasterio
    from scipy.misc import imsave
    from rasterio.features import shapes
    from shapely.geometry import shape
    import warnings

    global ht

    warnings.filterwarnings(
        "ignore", message="Dataset has no geotransform set")

    if request.method == 'POST':
        image = request.form.get('image')
        address = request.form.get('address')
        head = request.form.get('head')
        data = np.array([float(x) for x in image.split(',')])

        try:
            zill = ht.get_zillow(address)
        except Exception:
            zill = None

        radius = ht.get_zill_radius(zill)

        data[data < 255 / 2.0] = 0
        data[data > 255 / 2.0] = 1

        n = int(np.round(np.sqrt(data.shape[0])))
        data = np.reshape(data, (n, n))

        mask_path = os.path.join(app.root_path, 'queries', head + '-mask.png')
        if not os.path.exists(mask_path):
            imsave(mask_path,
                   np.repeat((data * 255)[..., np.newaxis], 3, axis=2).astype(
                       np.uint8))

        # Make outline image from mask
        with rasterio.open(mask_path) as src:
            image = src.read(1)
            mask = image != 255
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v)
                in enumerate(
                    shapes(image, mask=mask, transform=src.affine)))
        shapes = [shape(x['geometry']) for x in list(results)]
        outline_path = os.path.join(
            app.root_path, 'queries', head + '-outline.png')
        if not os.path.exists(outline_path):
            ht.make_mask_from_polys(
                shapes, outline_path, buff=0.25, reverse_y=True)

        a, b = int(np.floor(data.shape[0] / 2.0)
                   ), int(np.floor(data.shape[1] / 2.0))
        # Warning: 70 was eyeballed, need to get exact scale!
        r = int(np.round(n * radius / 70.0))

        y, x = np.ogrid[-a:n - a, -b:n - b]
        mask = x * x + y * y <= r * r

        full = np.ones((n, n))[mask]
        fraction = np.sum(1 - data[mask]) / np.sum(full)

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
        max_value_increase = 0.0
        one_tree_value = 0.0
        one_tree_frac = 0.05
        sqft = ht._DEFAULT_SQFT

        if zill is not None:
            sqft = ht.get_sqft(zill)
            if sqft is None:
                sqft = ht._DEFAULT_SQFT

            one_tree_frac = 0.05

            if zill.zestimate.amount is not None:
                house_value = float(zill.zestimate.amount)
                max_value_increase = 0.1 * house_value
                value_increase = max_value_increase * fraction
                one_tree_value = max_value_increase * one_tree_frac

        # Gross approximation: kwh usage = 0.5 * dd.
        if hdd is not None and cdd is not None and eprice is not None:
            cost = 0.6 * (hdd + cdd) * eprice * sqft / ht._DEFAULT_SQFT
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
        one_tree_noise = max_noise_abatement * np.log10(
            1.0 + 9.0 * one_tree_frac)

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


ht = HumanTree(load_canopy_polys=False, logger=logger)
if __name__ == '__main__':
    app.before_request(before_request)
    app.run_server(debug=True)
