"""Find trees on a property, make suggestions for new trees."""
import json
import os
import pprint
from glob import glob

import numpy as np
import requests
from scipy import misc
from tqdm import tqdm

# from pypolyline.util import encode_coordinates
from skimage.io import imsave
from skimage.transform import resize

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
    _PATTERN = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        "center={},{}&zoom={}&maptype=satellite&size={}x{}"
        "&key={}")
    _LAT_OFFSET = 2.e-5  # The LIDAR data is slightly offset from the images.
    _SMOOTH = 1.0
    _SCALED_SIZE = 512
    _ZOOM = 20
    _IMGSIZE = 640
    _CROPPIX = 20
    _DPI = 100.0
    _SBUFF = 2.e-6  # buffer size for smoothing tree regions
    _POINT_BUFF = 2.e-5
    _REGIONS = {
        'ENC': ['IL', 'IN', 'MI', 'OH', 'WI'],
        'WNC': ['IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
        'PAC': ['CA', 'OR', 'WA', 'AK'],
        'MTN': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY'],
        'NEC': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
        'MAC': ['NY', 'NJ', 'PE', 'WV'],
        'SAC': ['MD', 'DE', 'DC', 'WV', 'VA', 'NC', 'SC', 'GA', 'FL'],
        'ESC': ['KY', 'TN', 'MS', 'AL'],
        'WSC': ['TX', 'OK', 'AR', 'LA']
    }

    def __init__(self, **kwargs):
        """Initialize, loading data."""
        import eia
        import googlemaps
        import zillow
        from shapely.geometry import Polygon

        load_canopy_polys = kwargs.get('load_canopy_polys', True)

        self._dir_name = os.path.dirname(os.path.realpath(__file__))

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

        with open(os.path.join(self._dir_name, '..', 'google.key'), 'r') as f:
            self._google_key = f.readline().strip()
        self._google_client = googlemaps.Client(key=self._google_key)

        with open(os.path.join(self._dir_name, '..', 'eia.key'), 'r') as f:
            self._eia_key = f.readline().strip()
        self._eia_client = eia.API(self._eia_key)

        with open(os.path.join(self._dir_name, '..', 'zillow.key'), 'r') as f:
            self._zillow_key = f.readline().strip()
        self._zillow_client = zillow.ValuationApi()

        self._cropsize = self._IMGSIZE - 2 * self._CROPPIX

        # Load canopy data.
        if not load_canopy_polys:
            return

        self._canopy_fname = os.path.join(
            self._dir_name, '..', 'geo', 'ENVIRONMENTAL_TreeCanopy2014.json')
        with open(self._canopy_fname, 'r') as f:
            self._canopies = json.load(f)

        raw_canpols = [
            x.get('geometry', {}).get(
                'coordinates', []) for x in self._canopies.get(
                    'features', []) if x.get('geometry', {}) is not None]

        raw_canpols = list(filter(None, ([
            [y for y in x if len(y) >= 3] for x in raw_canpols])))

        raw_canpols = [[Polygon([(
            a, b + self._LAT_OFFSET) for a, b in y]) for y in x if len(
                y) >= 3] for x in raw_canpols]

        self._canopy_polygons = []
        for canpoly in tqdm(raw_canpols, desc='Extracting canopy polygons'):
            cps = [x.buffer(0) for x in canpoly]
            cps = self._canopy_polygons.extend([a for b in [
                list(x.geoms) if 'multi' in str(type(
                    x)).lower() else [x] for x in cps] for a in b])

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
        from shapely.geometry import Point

        lon, lat = self.get_coordinates(address)
        pt = Point(lon, lat)

        result = None
        for polys in self._parcel_polygons:
            for poly in polys:
                if poly.contains(pt):
                    print('Success!')
                    result = poly
                    break

        return result, pt

    def get_bound_poly(self, poly):
        """Get bounding polygon for property."""
        from shapely.geometry import Polygon

        mlon, mlat = poly.centroid.coords[0]
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        bp = self.get_static_map_bounds(
            mlat, mlon, self._ZOOM, self._cropsize, self._cropsize)

        return Polygon([
            bp[0], [bp[0][0], bp[1][1]], bp[1], [
                bp[1][0], bp[0][1]]]), mlat, mlon, bp

    def get_image(
        self, poly, mlat, mlon, zoom=None, imgsize=None, fname=None,
            target_dir=None):
        """Get satellite image from polygon boundaries."""
        import uuid

        zoom = self._ZOOM if zoom is None else zoom
        imgsize = self._IMGSIZE if imgsize is None else imgsize
        target_dir = 'parcels' if target_dir is None else target_dir

        if fname is None:
            fname = str(uuid.uuid4())

        tdir = os.path.join(self._dir_name, '..', target_dir)
        if not os.path.isdir(tdir):
            os.mkdir(tdir)
        fpath = os.path.join(self._dir_name, '..', tdir, fname + '.png')
        process_image = False if os.path.exists(fpath) else True
        query_url = self._PATTERN.format(
            mlat, mlon, zoom, imgsize, imgsize,
            self._google_key)
        self.download_file(query_url, fname=fpath)
        if process_image:
            pic = misc.imread(fpath)
            npic = pic[self._CROPPIX:-self._CROPPIX,
                       self._CROPPIX:-self._CROPPIX]
            npic = resize(
                npic, (self._SCALED_SIZE, self._SCALED_SIZE, 3),
                preserve_range=False, mode='constant')
            misc.imsave(fpath, npic)

        return fpath

    def get_image_from_address(self, address):
        """Get an image from an address."""
        if not address:
            raise ValueError('Invalid address `{}`!'.format(address))

        poly, pt = self.find_poly(address)
        if poly is None:
            poly = pt.buffer(self._POINT_BUFF)
        print(poly.exterior.coords, pt.coords)
        bound_poly, mlat, mlon, __ = self.get_bound_poly(poly)
        fpath = self.get_image(bound_poly, mlat, mlon, target_dir='queries')
        pic = misc.imread(fpath)
        with open(fpath.replace('.png', '.json'), 'w') as f:
            json.dump(pic.tolist(), f, separators=(',', ':'), indent=0)
        return fpath

    def get_poly_images(self, limit=None, purge=False):
        """Retrieve images of all polygons on Google Maps."""
        from shapely.ops import cascaded_union
        from matplotlib.patches import Polygon as MPPoly
        from matplotlib.collections import PatchCollection
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        # rearth = 6371000.0
        if not os.path.isdir('parcels'):
            os.mkdir('parcels')
        if purge:
            files = glob(os.path.join('parcels', '*'))
            for f in files:
                os.remove(f)
        self._train_count = 0
        lots_skipped = 0
        for pi, polys in enumerate(tqdm(self._parcel_polygons, total=limit)):
            if limit is not None and pi >= limit:
                break
            poly = polys[0]
            bound_poly, mlat, mlon, bp = self.get_bound_poly(poly)
            if not bound_poly.contains(poly):
                lots_skipped += 1
                continue

            fname = str(pi).zfill(5)

            fpaths = [os.path.join(
                self._dir_name, '..', 'parcels', fname + '-' + suffix +
                '.png') for suffix in ['mask', 'outline']]
            for fpath in fpaths:
                if not os.path.exists(fpath):
                    ipolys = []
                    success_count = 0
                    for cp in self._canopy_polygons:
                        if cp.disjoint(bound_poly):
                            continue
                        try:
                            ipolys.append(cp.intersection(bound_poly).buffer(
                                self._SBUFF).buffer(self._SBUFF).buffer(
                                    self._SBUFF).buffer(self._SBUFF))
                        except Exception as ee:
                            print(ee)
                        else:
                            success_count += 1

                    merged_polys = cascaded_union(ipolys)

                    if 'Multi' not in str(type(merged_polys)):
                        merged_polys = [merged_polys]

                    fig = plt.figure()
                    ax = fig.gca()
                    plt.axis('off')
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    patches = [MPPoly(
                        x.exterior.coords) for x in merged_polys if hasattr(
                            x, 'exterior')]
                    if 'mask' in fpath:
                        pc = PatchCollection(
                            patches, alpha=1, facecolors='black',
                            edgecolors='none', antialiased=False)
                        plt.gray()
                    else:
                        pc = PatchCollection(
                            patches, facecolors=(1, 0, 1, 0.2),
                            edgecolors='magenta', antialiased=True,
                            linewidth=4)
                    ax.autoscale_view(True, True, True)
                    ax.add_collection(pc)
                    ax.set_xlim(bp[0][0], bp[1][0])
                    ax.set_ylim(bp[0][1], bp[1][1])
                    fig.subplots_adjust(
                        bottom=0, left=0, top=self._SCALED_SIZE / self._DPI,
                        right=self._SCALED_SIZE / self._DPI)
                    fig.set_size_inches(1, 1)
                    if 'mask' in fpath:
                        plt.savefig(fpath, bbox_inches='tight', dpi=self._DPI,
                                    pad_inches=0)
                    else:
                        plt.savefig(fpath, bbox_inches='tight', dpi=self._DPI,
                                    pad_inches=0, transparent=True)
                    plt.close()

                    # If image is wrong size
                    pic = misc.imread(fpath)
                    shape = pic.shape
                    if (shape[0] != self._SCALED_SIZE or
                            shape[1] != self._SCALED_SIZE):
                        dx = (shape[0] - self._SCALED_SIZE) / 2.0
                        dy = (shape[1] - self._SCALED_SIZE) / 2.0
                        dxm, dxp = int(np.ceil(dx)), int(np.floor(dx))
                        dym, dyp = int(np.ceil(dy)), int(np.floor(dy))
                        npic = pic[dxm:-dxp, dym:-dyp]
                        misc.imsave(fpath, npic)

            self.get_image(bound_poly, mlat, mlon, fname=fname)

            self._train_count += 1

        print('Training on {} lots, skipped {} because they were '
              'too large.'.format(self._train_count, lots_skipped))

    def get_state(self, address):
        """Get lat/lon from address using Geocode API."""
        result = self._google_client.geocode(address)

        acs = result[0].get('address_components', {})
        state = [x for x in acs if 'administrative_area_level_1' in x.get(
            'types')][0].get('short_name')

        return state

    def get_zip(self, address):
        """Get zip from address using Geocode API."""
        result = self._google_client.geocode(address)

        acs = result[0].get('address_components', {})
        pc = [x for x in acs if 'postal_code' in x.get(
            'types')][0].get('short_name')

        return pc

    def get_electricity_price(self, state):
        """Get price of electricity per kwh."""
        series = 'ELEC.PRICE.{}-ALL.M'.format(state)
        result = self._eia_client.data_by_series(series)

        result = [(int(k.replace(' ', '')), v) for k, v in result[
            list(result.keys())[0]].items()]
        result = sorted(result)[-1][1]
        return result

    def get_degree_days(self, state, type='cooling'):
        """Get degree days for a given state."""
        region = [k for k, v in self._REGIONS.items() if state in v][0]

        series = 'STEO.ZW{}D_{}.A'.format(
            'C' if type == 'cooling' else 'H', region)
        result = self._eia_client.data_by_series(series)

        result = [(int(k.replace(' ', '')), v) for k, v in result[
            list(result.keys())[0]].items()]
        result = sorted(result)[-1][1]
        return result

    def get_zillow(self, address, postal_code):
        """Get square feet of a property."""
        return self._zillow_client.GetDeepSearchResults(
            self._zillow_key, address, postal_code, True)

    def get_coordinates(self, address):
        """Get lat/lon from address using Geocode API."""
        result = self._google_client.geocode(address)

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

    def get_data(self, fractions=(0.0, 0.8)):
        """Return image and mask for training image segmentation."""
        parcel_paths = list(sorted([
            x for x in glob(os.path.join(
                'parcels', '*.png')) if (
                    'mask' not in x and 'outline' not in x)]))
        mask_paths = list(sorted(glob(os.path.join('parcels', '*-mask.png'))))

        images = []
        masks = []
        min_i = 0 if fractions[0] == 0.0 else int(np.floor(
            fractions[0] * self._train_count)) + 1
        max_i = int(np.floor(fractions[1] * self._train_count))
        indices = list(range(min_i, max_i))
        pids = []
        for i in indices:
            image = misc.imread(parcel_paths[i])[:, :, :3]
            images.append(image)
            mask = misc.imread(mask_paths[i])[:, :, [0]]
            masks.append(mask)
            pids.append(int(parcel_paths[i].split('/')[-1].split('.')[0]))

        return np.array(images), np.array(masks), pids

    def get_unet(self):
        """Construct UNet."""
        from keras.optimizers import Adam
        from keras.models import Model
        from keras.layers import (Conv2D, Conv2DTranspose, Input, MaxPooling2D,
                                  concatenate)

        inputs = Input((self._SCALED_SIZE, self._SCALED_SIZE, 3))
        # inputs = Input((512, 512, 3))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
            2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
            2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
            2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
            2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        model.summary()

        model.compile(optimizer=Adam(lr=1e-4),
                      loss='binary_crossentropy',
                      metrics=['binary_crossentropy', 'acc'])

        return model

    def preprocess(self, imgs, channels=3, label='images'):
        """Put images in the appropriate format."""
        imgs_p = np.ndarray(
            (imgs.shape[0], self._SCALED_SIZE, self._SCALED_SIZE, channels),
            dtype=np.uint8)
        for i in tqdm(range(
                imgs.shape[0]), desc='Converting {}'.format(label)):
            imgs_p[i] = imgs[i].astype(np.uint8)

        return imgs_p

    def prepare_data(self):
        """Get images and masks ready for UNet."""
        imgs_train, imgs_mask_train, __ = self.get_data(fractions=(0.0, 0.8))

        imgs_train = self.preprocess(imgs_train, 3)
        imgs_mask_train = self.preprocess(imgs_mask_train, 1)

        self._imgs_train = imgs_train.astype('float32')
        self._imgs_mean = np.mean(self._imgs_train)  # mean for data centering
        self._imgs_std = np.std(self._imgs_train)  # std for data normalization

        self._imgs_train -= self._imgs_mean
        self._imgs_train /= self._imgs_std

        self._imgs_mask_train = imgs_mask_train.astype('float32')
        self._imgs_mask_train /= 255.  # scale masks to [0, 1]

    def notice(self, txt=''):
        """Print a notice for the user."""
        print('-' * 30)
        print(txt)
        print('-' * 30)

    def train(self):
        """Train DNN for image segmentation."""
        from keras import backend as K
        from keras.callbacks import ModelCheckpoint

        self.notice('Loading and preprocessing train data...')

        self.prepare_data()

        with open(os.path.join(self._dir_name, '..', 'meta.json'), 'w') as f:
            json.dump([float(self._imgs_mean), float(self._imgs_std)], f)

        self.notice('Creating and compiling model...')
        model = self.get_unet()
        model_checkpoint = ModelCheckpoint(
            'weights.h5', monitor='val_loss', save_best_only=True)

        self.notice('Fitting model...')

        tbCallBack = K.callbacks.TensorBoard()

        model.fit(
            self._imgs_train, self._imgs_mask_train, batch_size=8,
            epochs=40, verbose=1, shuffle=True, validation_split=0.2,
            callbacks=[model_checkpoint, tbCallBack])

    def predict_test(self):
        """Test trained UNet."""
        self.notice('Creating and compiling model...')
        model = self.get_unet()

        self.notice('Loading and preprocessing test data...')
        imgs_test, masks_test, ids_test = self.get_data(fractions=(0.8, 1.0))
        imgs_test = self.preprocess(imgs_test, 3)

        imgs_test = imgs_test.astype('float32')
        imgs_test -= self._imgs_mean
        imgs_test /= self._imgs_std

        self.notice('Loading saved weights...')
        model.load_weights('weights.h5')

        self.notice('Predicting masks on test data...')
        imgs_mask_test = model.predict(imgs_test, verbose=1)
        np.save('imgs_mask_test.npy', imgs_mask_test)

        self.notice('Saving predicted masks to files...')
        pred_dir = 'preds'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        for image, id in zip(imgs_mask_test, ids_test):
            image = (image[:, :, 0] * 255.).astype(np.uint8)
            imsave(os.path.join(
                pred_dir, str(id).zfill(5) + '-pred.png'), image)
