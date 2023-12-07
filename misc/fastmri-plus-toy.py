import os
import pandas as pd
import numpy as np
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import h5py

# %matplotlib inline

annotation_path = '/home/xsd618/fastmri-plus/Annotations/brain.csv'
fastmri_path = '/home/xsd618/data/fastMRI_test_subset_brain/multicoil_test/'

for f in os.walk(fastmri_path):
    for fastmri_file in f[2]:
        fastmri_file = fastmri_file[:-3]
        print(fastmri_file)

        # Labels for this file
        df = pd.read_csv(annotation_path, index_col=None, header=0)
        labels_for_file = df.loc[df['file'] == fastmri_file]
        labels_for_file['label'].unique()

        print(labels_for_file['label'].unique())


fastmri_file = 'file_brain_AXFLAIR_200_6002558'
#fastmri_file = 'file_brain_AXFLAIR_200_6002585'
#fastmri_file = 'file_brain_AXFLAIR_200_6002549'

df = pd.read_csv(annotation_path, index_col=None, header=0)
labels_for_file = df.loc[df['file'] == fastmri_file]
labels_for_file['label'].unique()

datafile = os.path.join(fastmri_path, fastmri_file + '.h5')
f = h5py.File(datafile,'r')
img_data = f['reconstruction_rss'][:]
img_data = img_data[:, ::-1, :]  # flipped up down


def plot_bounding_box(image, labels):
    plotted_image = ImageDraw.Draw(image)
    for label in labels:
        _, _, _, x0, y0, w, h, label_txt = label
        x1 = x0 + w
        y1 = y0 + h
        plotted_image.rectangle(((x0,y0), (x1,y1)), outline="white")
        plotted_image.text((x0, max(0, y0 - 10)), label_txt, fill= "white")
    return np.array(image)


# Display an imageslice with annotations
slice_choice = 6
scale = 0.5

labels_for_slice = labels_for_file.loc[labels_for_file['slice'] == slice_choice].values.tolist()
arrimg = np.squeeze(img_data[slice_choice,:,:])
image_2d_scaled = (np.maximum(arrimg,0) / (scale*arrimg.max())) * 255.0
image_2d_scaled[image_2d_scaled > 255] = 255
image_2d_scaled = Image.fromarray(np.uint8(image_2d_scaled))
print(labels_for_slice)
annotated_img = plot_bounding_box(image_2d_scaled, labels_for_slice)
plt.figure(figsize = (5,5))
plt.imshow(annotated_img,'gray')
plt.show()



