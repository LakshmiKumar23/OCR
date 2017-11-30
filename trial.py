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

file = open('sample.txt')
img = file.readlines()

sample_array = []
for line in img:
    for i in range(28):
        if line[i] == ' ':
            sample_array.append(0)
        elif line[i] == '#':
            sample_array.append(255)
        elif line[i] == '+':
            sample_array.append(225)
        else:
            continue

sample_array = np.array(sample_array).reshape(28, 28)
img = sample_array

'''
# Canny edge detector
img = img.astype(np.uint8)
edges = cv2.Canny(img, 100, 200)
print edges
'''


# hu moments
features = []
# Binarization by Thresholding
th = 200
img_binary = (img < th).astype(np.double)
# Connected Component Analysis
img_label = label(img_binary, background=255)
# Computing Hu Moments, Removing Small Components and Displaying Component Bounding Boxes
regions = regionprops(img_label)
io.imshow(img_binary)
ax = plt.gca()
for props in regions:
    minr, minc, maxr, maxc = props.bbox
    height = maxr - minr
    width = maxc - minc
    print height, width
    ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    roi = img_binary[minr:maxr, minc:maxc]
    m = moments(roi)
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]
    mu = moments_central(roi, cr, cc)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    features.append(hu)
    print features
plt.title('Bounding Boxes')
io.show()
