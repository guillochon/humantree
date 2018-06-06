"""Find trees on a property, make suggestions for new trees."""
import json
import os
import pprint

import numpy as np
import requests
from scipy import misc
from tqdm import tqdm

import googlemaps
from shapely.geometry import Point, Polygon

pp = pprint.PrettyPrinter(indent=4)


class HumanTree(object):
    """Count trees, make suggestions where new trees should be added."""

    _PARCELS_URL = (
        "https://raw.githubusercontent.com/cambridgegis/cambridgegis_data"
        "/master/Assessing/FY2018/FY2018_Parcels/"
        "ASSESSING_ParcelsFY2018.geojson")

    def __init__(self):
        """Initialize, loading data."""
        self._parcels_fname = self.download_file(self._PARCELS_URL)

        with open(self._parcels_fname, 'r') as f:
            self._parcels = json.load(f)

        self._parcel_polygons = [
            x.get('geometry', {}).get('coordinates', [])
            for x in self._parcels.get('features', [])]

        # print(self._parcel_polygons)

        self._parcel_polygons = list(filter(None, ([
            [y for y in x if len(y) >= 3] for x in self._parcel_polygons])))

        self._parcel_polygons = [[Polygon(y) for y in x if len(y) >= 3]
                                 for x in self._parcel_polygons]

        with open('google.key', 'r') as f:
            self._google_key = f.readline().strip()

        self._client = googlemaps.Client(key=self._google_key)

    def download_file(self, url, fname=None):
        """Download file if it doesn't exist locally."""
        if fname is None:
            fname = url.split('/')[-1]
        if not os.path.exists(fname):
            response = requests.get(url, stream=True)
            block_size = 1024
            total_size = int(response.headers.get('content-length', 0))
            with open(fname, 'wb') as handle:
                for data in tqdm(
                    response.iter_content(1024),
                    total=np.ceil(total_size // block_size),
                        unit='KB', unit_scale=True):
                    handle.write(data)
        return fname

    def find_poly(self, address):
        """Count trees on a property."""
        lon, lat = self.get_coordinates(address)
        pt = Point(lon, lat)

        for polys in self._parcel_polygons:
            for poly in polys:
                if poly.contains(pt):
                    print('Success!')
                    result = poly
                    break

        return result

    def get_poly_images(self):
        """Retrieve images of all polygons on Google Maps."""
        zoom = 20
        imgsize = 640
        croppix = 20
        rearth = 6371000.0
        pattern = (
            "https://maps.googleapis.com/maps/api/staticmap?"
            "visible={},{}&zoom={}&maptype=satellite&size={}x{}"
            "&key={}")
        if not os.path.isdir('parcels'):
            os.mkdir('parcels')
        for pi, polys in enumerate(tqdm(self._parcel_polygons)):
            if pi > 100:
                break
            poly = polys[0]
            mlon, mlat = poly.centroid.coords[0]
            query_url = pattern.format(
                mlat, mlon, zoom, imgsize, imgsize, self._google_key)
            # print(query_url)
            # Calculate physical size in meters of image.
            physical_size = (imgsize - 2 * croppix) * 156543.03392 * np.cos(
                mlat * np.pi / 180) / (2 ** zoom)
            min_lon, min_lat, max_lon, max_lat = [
                x * np.pi / 180 for x in poly.bounds]
            dlat = max_lat - min_lat
            dlon = max_lon - min_lon
            aa = np.sin(
                dlat / 2) ** 2 + np.cos(min_lat) * np.cos(max_lat) * np.sin(
                    dlon / 2) ** 2
            cc = 2.0 * np.arctan2(np.sqrt(aa), np.sqrt(1.0 - aa))
            dd = rearth * cc
            if dd > physical_size:
                # print('Lot {} too large, skipping.'.format(pi))
                continue
            fname = str(pi).zfill(5) + '.png'
            fpath = os.path.join('parcels', fname)
            crop_image = False if os.path.exists(fname) else True
            self.download_file(query_url, fname=fpath)
            if crop_image:
                pic = misc.imread(fpath)
                pic = pic[croppix:-croppix, croppix:-croppix]
                misc.imsave(fpath, pic)

    def get_coordinates(self, address):
        """Get lat/lon from address using Geocode API."""
        result = self._client.geocode(address)

        location = result[0].get('geometry', {}).get('location', {})

        lon = location.get('lng')
        lat = location.get('lat')

        return (lon, lat)
