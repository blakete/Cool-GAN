import matplotlib.pyplot as plt
import numpy as np
from random import randint
import time
from numpy.random import randn
import argparse
from datetime import datetime 
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type=str, help="Path to model .h5 file")
args = parser.parse_args()

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
	ratios = np.linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = (1.0 - ratio) * p1 + ratio * p2
		vectors.append(v)
	return np.asarray(vectors)

# generate latent points
n_samples = 100
n_steps = 1
latent_dim = 100
latent_points = generate_latent_points(latent_dim, n_samples)
# interpolate points in latent space
trippy = np.zeros(((n_samples-1)*n_steps, latent_dim))
for i in range(0, n_samples-1):
    interpolated = interpolate_points(latent_points[i], latent_points[i+1], n_steps=n_steps)
    trippy[i*n_steps:(i+1)*n_steps,:] = interpolated

# load model 
# forward pass latent_points through model
import tensorflow as tf
generator = tf.keras.models.load_model(args.model_path)
print("Predicting")
gan_images = generator.predict(trippy)

# TODO save trippy numpy to file and create dataset of high quality face vectors
now = datetime.now()
now = now.strftime("%Y-%m-%d_%H:%M:%S")
save_dir = f"data/latent_vectors/faces/{args.model_path.split('/')[-1].split('.')[0]}/"
os.makedirs(save_dir, exist_ok=True)
save_file = f"{save_dir}/{now}_gan_images.npy"
with open(save_file, 'wb') as f:
	np.save(f, trippy)
print(f"Save file: {save_file}")

def scale_img(img):
	t_img = img + abs(np.amin(img))
	s_img = t_img / np.amax(t_img)
	return s_img

# plot images successively
print("Plotting")
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
implot = ax.imshow(gan_images[0, :, :, :])

for i in range((n_samples-1)*n_steps):
	print(f"Face index: {i-1}")
	time.sleep(1.0)
	# scale images to [0,1]
	img = gan_images[i, :, :, :]
	implot.set_data(scale_img(img))
	fig.canvas.flush_events()




