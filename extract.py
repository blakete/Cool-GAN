import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, help="Path to model .h5 file")
args = parser.parse_args()

# extract face index from given numpy file and append to saved faces
cool_latent_vecs = {}
cool_latent_vecs["data/latent_vectors/faces/generator_model_100//2021-12-07_21:10:14_gan_images.npy"] = [20,21,23,28,29,47]
cool_latent_vecs["data/latent_vectors/faces/generator_model_100/2021-12-07_20:45:41_gan_images.npy"] = [4, 6]
cool_latent_vecs["data/latent_vectors/faces/generator_model_100//2021-12-07_20:55:35_gan_images.npy"] = [3,10,16,17,18,30,42,48,50,55,73,79,85]



cool_vecs = []
for key in cool_latent_vecs.keys():
    vec_idxs = cool_latent_vecs[key]
    vecs = np.load(key)
    for i in vec_idxs:
        cool_vecs.append(vecs[i])
cool_vecs = np.asarray(cool_vecs)

def interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
	ratios = np.linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return np.asarray(vectors)

# interpolate cool vecs
n_steps = 70
n_samples = cool_vecs.shape[0]
trippy = np.zeros(((n_samples-1)*n_steps, 100))
for i in range(0, n_samples-1):
    interpolated = interpolate_points(cool_vecs[i], cool_vecs[i+1], n_steps=n_steps)
    trippy[i*n_steps:(i+1)*n_steps,:] = interpolated

# load model
import tensorflow as tf
generator = tf.keras.models.load_model(args.model_path)
# forward pass cool_vecs through model
# gan_images = generator.predict(cool_vecs)
gan_images = generator.predict(trippy)


# plot images
def scale_img(img):
	t_img = img + abs(np.amin(img))
	s_img = t_img / np.amax(t_img)
	return s_img

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
implot = ax.imshow(gan_images[0, :, :, :])

for i in range((n_samples-1)*n_steps):
    # scale images to [0,1]
    img = gan_images[i, :, :, :]
    implot.set_data(scale_img(img))
    fig.canvas.flush_events()
    time.sleep(0.01)
