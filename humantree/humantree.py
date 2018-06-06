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
from shapely.errors import TopologicalError
from pypolyline.util import encode_coordinates, decode_polyline

pp = pprint.PrettyPrinter(indent=4)


class HumanTree(object):
    """Count trees, make suggestions where new trees should be added."""

    _TO_RAD = np.pi / 180

    _PARCELS_URL = (
        "https://raw.githubusercontent.com/cambridgegis/cambridgegis_data"
        "/master/Assessing/FY2018/FY2018_Parcels/"
        "ASSESSING_ParcelsFY2018.geojson")

    _TREE_CANOPY_URL = (
        "https://raw.githubusercontent.com/cambridgegis/cambridgegis_data/"
        "master/Environmental/Tree_Canopy_2014/"
        "ENVIRONMENTAL_TreeCanopy2014.topojson")

    def __init__(self):
        """Initialize, loading data."""
        # Load parcel data.
        self._parcels_fname = self.download_file(self._PARCELS_URL)

        with open(self._parcels_fname, 'r') as f:
            self._parcels = json.load(f)

        self._parcel_polygons = [
            x.get('geometry', {}).get('coordinates', [])
            for x in self._parcels.get('features', [])]

        self._parcel_polygons = list(filter(None, ([
            [y for y in x if len(y) >= 3] for x in self._parcel_polygons])))

        self._parcel_polygons = [[Polygon(y) for y in x if len(y) >= 3]
                                 for x in self._parcel_polygons]

        # Load canopy data.
        self._canopy_fname = self.download_file(self._TREE_CANOPY_URL)

        with open(self._canopy_fname, 'r') as f:
            self._canopies_topo = json.load(f)

        self._canopy_polygons = [
            x.get('geometry', {}).get('coordinates', [])
            for x in self._canopies_topo.get('features', [])]

        # Convert canopies to polygons.
        scale = np.array(self._canopies_topo['transform']['scale'])
        trans = np.array(self._canopies_topo['transform']['translate'])
        arcs = [[
            (y + trans).tolist() for y in (
                np.cumsum(x, axis=0) * scale)] for x in self._canopies_topo[
                    'arcs']]

        self._canopies = []
        for cano in tqdm(self._canopies_topo['objects'][
                'ENVIRONMENTAL_TreeCanopy2014']['geometries']):
            arc_ids = cano['arcs'][0]
            print(cano['arcs'])

            poly = []
            for aid in arc_ids:
                lid = aid[0] if isinstance(aid, list) else aid
                pos = np.sign(lid)
                lid = np.abs(lid)
                larcs = arcs[lid - 1]
                if pos:
                    poly.extend(larcs)
                else:
                    poly.extend(reversed(larcs))
                print(aid, pos, larcs)
            poly = np.unique(poly, axis=0)
            if poly.shape[0] < 3:
                continue
            cpoly = Polygon(poly)
            if not cpoly.is_valid:
                print(cpoly)
                print(cpoly.exterior.type)
                print(cpoly.exterior.is_valid)
                print(encode_coordinates([(
                    y, x) for x, y in cpoly.exterior.coords], 5))
                raise ValueError('invalid poly')
            self._canopies.append(cpoly)

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
        cropsize = imgsize - 2 * croppix
        rearth = 6371000.0
        pattern = (
            "https://maps.googleapis.com/maps/api/staticmap?"
            "center={},{}&zoom={}&maptype=satellite&size={}x{}"
            "&key={}")
        if not os.path.isdir('parcels'):
            os.mkdir('parcels')
        for pi, polys in enumerate(tqdm(self._parcel_polygons)):
            if pi > 10:
                break
            poly = polys[0]
            mlon, mlat = poly.centroid.coords[0]
            query_url = pattern.format(
                mlat, mlon, zoom, imgsize, imgsize, self._google_key)
            # print(query_url)
            # Calculate physical size in meters of image.
            physical_size = cropsize * 156543.03392 * np.cos(
                mlat * np.pi / 180) / (2 ** zoom)
            min_lon, min_lat, max_lon, max_lat = poly.bounds
            bp = self.get_static_map_bounds(
                mlat, mlon, zoom, cropsize, cropsize)
            bound_poly = Polygon([
                bp[0], [bp[0][0], bp[1][1]], bp[1], [bp[1][0], bp[0][1]]])
            ipolys = []
            for canpoly in self._canopies:
                if not canpoly.intersects(bound_poly):
                    # print('no overlap')
                    continue
                try:
                    ipolys.append(canpoly.intersection(bound_poly))
                except TopologicalError:
                    pass
                    # print('fail')
                    # print(e, canpoly)
                else:
                    pass
                    # print('success')
            print(ipolys)
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

    def get_static_map_bounds(self, lat, lng, zoom, sx, sy):
        """Get bounds of a static map from Google.

        From https://stackoverflow.com/questions/12507274/
        how-to-get-bounds-of-a-google-static-map
        """
        # lat, lng - center
        # sx, sy - map size in pixels

        # 256 pixels - initial map size for zoom factor 0
        sz = 256 * 2 ** zoom

        # resolution in degrees per pixel
        res_lat = np.cos(lat * self._TO_RAD) * 360. / sz
        res_lng = 360. / sz

        d_lat = res_lat * sy / 2
        d_lng = res_lng * sx / 2

        return ((lng - d_lng, lat - d_lat), (lng + d_lng, lat + d_lat))
