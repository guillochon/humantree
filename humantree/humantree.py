"""Find trees on a property, make suggestions for new trees."""
import json
import os
import pprint
from glob import glob

import numpy as np
import requests
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MPPoly
from scipy import misc
from tqdm import tqdm

import googlemaps
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (Conv2D, Conv2DTranspose, Input, MaxPooling2D,
                          concatenate)
from keras.models import Model
from keras.optimizers import Adam
# from pypolyline.util import encode_coordinates
from shapely.geometry import Point, Polygon
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

    _LAT_OFFSET = 2.e-5  # The LIDAR data is slightly offset from the images.

    _SMOOTH = 1.0

    _SCALED_SIZE = 512

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
        self._canopy_fname = 'ENVIRONMENTAL_TreeCanopy2014.json'
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
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        zoom = 20
        imgsize = 640
        croppix = 20
        dpi = 100.0
        sbuff = 2.e-6  # buffer size for smoothing tree regions
        self._cropsize = imgsize - 2 * croppix
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
        lots_skipped = 0
        for pi, polys in enumerate(tqdm(self._parcel_polygons, total=limit)):
            if limit is not None and pi >= limit:
                break
            poly = polys[0]
            mlon, mlat = poly.centroid.coords[0]
            min_lon, min_lat, max_lon, max_lat = poly.bounds
            bp = self.get_static_map_bounds(
                mlat, mlon, zoom, self._cropsize, self._cropsize)

            bound_poly = Polygon([
                bp[0], [bp[0][0], bp[1][1]], bp[1], [bp[1][0], bp[0][1]]])
            if not bound_poly.contains(poly):
                lots_skipped += 1
                continue

            fname = str(pi).zfill(5)
            fpath = os.path.join('parcels', fname + '-mask.png')
            if not os.path.exists(fpath):
                ipolys = []
                success_count = 0
                for cp in self._canopy_polygons:
                    if cp.disjoint(bound_poly):
                        continue
                    try:
                        ipolys.append(cp.intersection(bound_poly).buffer(
                            sbuff).buffer(sbuff).buffer(sbuff).buffer(sbuff))
                    except Exception as ee:
                        print(ee)
                    else:
                        success_count += 1

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
                    bottom=0, left=0, top=self._SCALED_SIZE / dpi,
                    right=self._SCALED_SIZE / dpi)
                fig.set_size_inches(1, 1)
                plt.savefig(fpath, bbox_inches='tight', dpi=dpi, pad_inches=0)
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

            fpath = os.path.join('parcels', fname + '.png')
            process_image = False if os.path.exists(fpath) else True
            query_url = pattern.format(
                mlat, mlon, zoom, imgsize, imgsize, self._google_key)
            self.download_file(query_url, fname=fpath)
            if process_image:
                pic = misc.imread(fpath)
                npic = pic[croppix:-croppix, croppix:-croppix]
                npic = resize(
                    npic, (self._SCALED_SIZE, self._SCALED_SIZE, 3),
                    preserve_range=False, mode='constant')
                misc.imsave(fpath, npic)

            self._train_count += 1

        print('Training on {} lots, skipped {} because they were '
              'too large.'.format(self._train_count, lots_skipped))

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

    def get_data(self, fractions=(0.0, 0.8)):
        """Return image and mask for training image segmentation."""
        parcel_paths = list(sorted([
            x for x in glob(os.path.join(
                'parcels', '*.png')) if 'mask' not in x]))
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

    def dice_coef(self, y_true, y_pred):
        """Return the Dice coefficient."""
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + self._SMOOTH) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + self._SMOOTH)

    def dice_coef_loss(self, y_true, y_pred):
        """Return loss function (negative of the Dice coefficient)."""
        return -self.dice_coef(y_true, y_pred)

    def get_unet(self):
        """Construct UNet."""
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

    def preprocess(self, imgs, channels=3):
        """Put images in the appropriate format."""
        imgs_p = np.ndarray(
            (imgs.shape[0], self._SCALED_SIZE, self._SCALED_SIZE, channels),
            dtype=np.uint8)
        for i in tqdm(range(imgs.shape[0]), desc='Processing images'):
            imgs_p[i] = imgs[i].astype(np.uint8)
            # imgs_p[i] = resize(
            #     imgs[i], (self._SCALED_SIZE, self._SCALED_SIZE, channels),
            #     preserve_range=True, mode='constant')

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
        import pickle

        self.notice('Loading and preprocessing train data...')

        self.prepare_data()

        with open('meta.pkl', 'wb') as f:
            pickle.dump([self._imgs_mean, self._imgs_std], f)

        self.notice('Creating and compiling model...')
        model = self.get_unet()
        model_checkpoint = ModelCheckpoint(
            'weights.h5', monitor='val_loss', save_best_only=True)

        self.notice('Fitting model...')

        model.fit(
            self._imgs_train, self._imgs_mask_train, batch_size=8,
            epochs=40, verbose=1, shuffle=True, validation_split=0.2,
            callbacks=[model_checkpoint])

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
