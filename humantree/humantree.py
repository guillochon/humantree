"""Find trees on a property, make suggestions for new trees."""
import json
import os
import pprint

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MPPoly
from scipy import misc
from tqdm import tqdm

import googlemaps
from pypolyline.util import encode_coordinates
from shapely.geometry import Point, Polygon
from glob import glob

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
        # self._canopy_fname = self.download_file(self._TREE_CANOPY_URL)

        self._canopy_fname = 'ENVIRONMENTAL_TreeCanopy2014.json'
        with open(self._canopy_fname, 'r') as f:
            self._canopies = json.load(f)

        self._canopy_polygons = [
            x.get('geometry', {}).get(
                'coordinates', []) for x in self._canopies.get(
                    'features', []) if x.get('geometry', {}) is not None]

        self._canopy_polygons = list(filter(None, ([
            [y for y in x if len(y) >= 3] for x in self._canopy_polygons])))

        self._canopy_polygons = [[Polygon(y) for y in x if len(y) >= 3]
                                 for x in self._canopy_polygons]

        # self._canopy_polygons = [
        #     x.get('geometry', {}).get('coordinates', [])
        #     for x in self._canopies_topo.get('features', [])]
        #
        # # Convert canopies to polygons.
        # scale = np.array(self._canopies_topo['transform']['scale'])
        # trans = np.array(self._canopies_topo['transform']['translate'])
        # arcs = [[
        #     (y + trans).tolist() for y in (
        #         np.cumsum(x, axis=0) * scale)] for x in self._canopies_topo[
        #             'arcs']]
        #
        # self._canopies = []
        # for ci, cano in enumerate(tqdm(self._canopies_topo['objects'][
        #         'ENVIRONMENTAL_TreeCanopy2014']['geometries'])):
        #     arc_ids = cano['arcs'][0]
        #     # print(cano['arcs'])
        #
        #     poly = []
        #     for ai, aid in enumerate(arc_ids):
        #         lid = aid[0] if isinstance(aid, list) else aid
        #         pos = np.sign(lid)
        #         si = 1 if ai else 0
        #         if pos:
        #             larcs = arcs[lid]
        #             poly.extend(larcs[si:])
        #         else:
        #             larcs = arcs[~lid]
        #             poly.extend(larcs[si:])
        #         # print(aid, pos, larcs)
        #     u, ind = np.unique(poly, axis=0, return_index=True)
        #     poly = u[np.argsort(ind)]
        #     if poly.shape[0] < 3:
        #         continue
        #     # Hack for now.
        #     cpoly = Polygon(poly)  # .convex_hull
        #     if not cpoly.is_valid:
        #         # print(cpoly)
        #         # print(cpoly.exterior.type)
        #         # print(cpoly.exterior.is_valid)
        #         print(encode_coordinates([(
        #             y, x) for x, y in cpoly.exterior.coords], 5))
        #         continue
        #         # raise ValueError('invalid poly')
        #     else:
        #         self._canopies.append(cpoly)

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

    def get_poly_images(self, limit=None, purge=False):
        """Retrieve images of all polygons on Google Maps."""
        zoom = 20
        imgsize = 640
        croppix = 20
        dpi = 100.0
        cropsize = imgsize - 2 * croppix
        # rearth = 6371000.0
        pattern = (
            "https://maps.googleapis.com/maps/api/staticmap?"
            "center={},{}&zoom={}&maptype=satellite&size={}x{}"
            "&key={}")
        if not os.path.isdir('parcels'):
            os.mkdir('parcels')
        if purge:
            files = glob(os.path.join('parcels', '*'))
            for f in files:
                os.remove(f)
        self._train_count = 0
        for pi, polys in enumerate(tqdm(self._parcel_polygons, total=limit)):
            if limit is not None and pi >= limit:
                break
            poly = polys[0]
            mlon, mlat = poly.centroid.coords[0]
            # print(query_url)
            # Calculate physical size in meters of image.
            # physical_size = cropsize * 156543.03392 * np.cos(
            #     mlat * np.pi / 180) / (2 ** zoom)
            min_lon, min_lat, max_lon, max_lat = poly.bounds
            bp = self.get_static_map_bounds(
                mlat, mlon, zoom, cropsize, cropsize)

            bound_poly = Polygon([
                bp[0], [bp[0][0], bp[1][1]], bp[1], [bp[1][0], bp[0][1]]])
            if not bound_poly.contains(poly):
                print('Lot {} not fully contained in image, skipping.'.format(
                    pi))
                continue

            fname = str(pi).zfill(5)
            fpath = os.path.join('parcels', fname + '-mask.png')
            if not os.path.exists(fpath):
                ipolys = []
                success_count = 0
                for canpoly in self._canopy_polygons:
                    cp = canpoly[0]
                    if cp.disjoint(bound_poly):
                        # print('no overlap')
                        continue
                    try:
                        ipolys.append(cp.intersection(bound_poly))
                    except Exception:
                        try:
                            for geo in cp.buffer(
                                    0).intersection(bound_poly).geoms:
                                ipolys.append(geo)
                        except Exception as e:
                            print(e, cp)
                    else:
                        success_count += 1
                print(success_count)
                # print(ipolys)

                fig = plt.figure()
                ax = fig.gca()
                plt.axis('off')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                patches = [MPPoly(
                    x.exterior.coords) for x in ipolys if hasattr(
                        x, 'exterior')]
                pc = PatchCollection(
                    patches, alpha=1, facecolors='black', edgecolors=None,
                    antialiased=False)
                ax.autoscale_view(True, True, True)
                plt.gray()
                ax.add_collection(pc)
                ax.set_xlim(bp[0][0], bp[1][0])
                ax.set_ylim(bp[0][1], bp[1][1])
                fig.subplots_adjust(
                    bottom=0, left=0, top=cropsize / dpi, right=cropsize / dpi)
                fig.set_size_inches(1, 1)
                plt.savefig(fpath, bbox_inches='tight', dpi=dpi, pad_inches=0)

            fpath = os.path.join('parcels', fname + '.png')
            crop_image = False if os.path.exists(fpath) else True
            query_url = pattern.format(
                mlat, mlon, zoom, imgsize, imgsize, self._google_key)
            self.download_file(query_url, fname=fpath)
            if crop_image:
                pic = misc.imread(fpath)
                npic = pic[croppix:-croppix, croppix:-croppix]
                misc.imsave(fpath, npic)

            self._train_count += 1

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

    def training_data(self):
        """Return image and mask for training image segmentation."""
        parcel_paths = list(sorted(glob(os.path.join('parcels', '*.png'))))
        mask_paths = list(sorted(glob(os.path.join('parcels', '*-mask.png'))))

        images = []
        masks = []
        for i in range(self._train_count):
            images.append(misc.imread(parcel_paths[i])[:, :, :3])
            mask = np.rint(
                misc.imread(mask_paths[i])[:, :, 0] / 255).astype(int)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def train(self):
        """Train DNN for image segmentation."""
        from tf_unet import unet
        from tf_unet.image_util import SimpleDataProvider

        net = unet.Unet(layers=6, features_root=16)
        trainer = unet.Trainer(
            net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
        data, label = self.training_data()
        generator = SimpleDataProvider(data, label, channels=3)
        trainer.train(
            generator, "./unet_trained", training_iters=20, epochs=10,
            display_step=2)
