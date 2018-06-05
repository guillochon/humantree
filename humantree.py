"""Find trees on a property, make suggestions for new trees."""
import json

import numpy as np
import requests
from tqdm import tqdm


def download_file(url, fname=None):
    """Download file if it doesn't exist locally."""
    response = requests.get(url, stream=True)

    if fname is None:
        fname = url.split('/')[-1]

    block_size = 1024
    total_size = int(response.headers.get('content-length', 0))
    with open(fname, 'wb') as handle:
        for data in tqdm(
            response.iter_content(1024),
            total=np.ceil(total_size // block_size),
                unit='KB', unit_scale=True):
            handle.write(data)
    return fname


parcels_url = (
    "https://raw.githubusercontent.com/cambridgegis/cambridgegis_data"
    "/master/Assessing/FY2018/FY2018_Parcels/ASSESSING_ParcelsFY2018.geojson")

fname = download_file(parcels_url)

with open(fname, 'r') as f:
    parcels = json.load(f)

print(parcels)
