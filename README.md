# Cool-GAN
This GAN was made to generate a cool profile photo. 

## Requirements
* Docker
* Nvidia-Docker
* Cuda GPU

## Install
* Clone this repository
```
$ cd Cool-GAN
$ ./build.sh
$ ./run.sh
```

## Usage
### Pre-trained models
```
$ cd Cool-GAN
$ python plotting.py --model_path models/faces/generator_model_100.h5
```
* Enjoy your Cool GAN faces :)

### Training models
* Once inside the Docker container
```
$ python mnist-gan.py # to train mnist GAN
```
OR
```
$ python face-gan.py # to train celebrity faces GAN
```
* Refer to pre-trained models usage instructions to test your trained gan

## References
* https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
* https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
