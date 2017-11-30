import numpy as np
from skimage.feature import corner_orientations
from matplotlib import pyplot as plt
from skimage import io
from skimage.measure import label
from skimage.measure import regionprops
from skimage.measure import moments
from skimage.measure import moments_central
from skimage.measure import moments_normalized
from skimage.measure import moments_hu
import numpy as np
from matplotlib.patches import Rectangle


def get_hog_features(samples):
    print 'getting hog features...'
    hog_features = []
    for sample in samples:
        sample = np.array(sample)
        hog_features.append(corner_orientations(sample))
    hog_features = np.array(hog_features)
    print "hog features = ", hog_features.shape
    return hog_features


def get_window_pixels(samples, window):
    print 'getting window_pixel_count...'
    window_pixel_count = []
    temp = samples[0]
    r, c = temp.shape
    count = 1
    for sample in samples:
        sample_count = []
        sample = np.array(sample)
        i = 0
        while i <= (r - window):
            j = 0
            while j <= (c - window):
                temp = sample[i:i + window, j:j + window]
                pixels = 0
                rr, cc = temp.shape
                for m in range(rr):
                    for n in range(cc):
                        if temp[m, n] != 0:
                            pixels += 1
                sample_count.append(pixels)
                j += window
            i += window
        window_pixel_count.append(sample_count)
        count += 1
    return window_pixel_count


def getwhitepixels(samples):
    print 'getting white_pixels...'
    length, height, width = samples.shape
    white_pixels = []
    for sample in samples:
        count = 0
        for i in range(height):
            for j in range(width):
                if sample[i, j] != 0:
                    count += 1
        white_pixels.append(count)
    return white_pixels


def getwhitepixels_byrows(samples):
    print 'getting white_pixels_row_wise...'
    length, height, width = samples.shape
    white_pixels_row_wise = []
    for sample in samples:
        white_pixels = []
        for i in range(height):
            count = 0
            for j in range(width):
                if sample[i, j] != 0:
                    count += 1
            white_pixels.append(count)
        white_pixels_row_wise.append(white_pixels)
    return white_pixels_row_wise


def getwhitepixels_bycols(samples):
    print 'getting white_pixels_col_wise...'
    length, height, width = samples.shape
    white_pixels_col_wise = []
    for sample in samples:
        sample = sample.transpose()
        white_pixels = []
        for i in range(width):
            count = 0
            for j in range(height):
                if sample[i, j] != 0:
                    count += 1
            white_pixels.append(count)
        white_pixels_col_wise.append(white_pixels)
    return white_pixels_col_wise


def get_hu_moments(samples):
    print "getting hu moments..."
    features = []
    for sample in samples:
        '''
        sample = np.array(sample)
        th = 200
        img_binary = (sample < th).astype(np.double)
        img_label = label(img_binary, background=255)
        regions = regionprops(img_label)
        if regions == []:
            print "no regions"
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            roi = img_binary[minr:maxr, minc:maxc]

        '''
        sample = np.array(sample)
        sample = sample.astype(np.double)
        m = moments(sample)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = moments_central(sample, cr, cc)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        features.append(hu)
    return features
