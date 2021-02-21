import math
import random
from imageio import imread
from skimage.color import rgb2gray
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from scipy.ndimage.filters import convolve
from . import sol5_utils


# kernel size of convolution
KERNEL_SIZE = (3, 3)
# representation code for a gray scale image
GRAY_REP = 1


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: The filename of an image on disk (could be grayscale or RGB).
    :param representation: Representation code, either 1 or 2 defining whether the output should be a grayscale
            image (1) or an RGB image (2).
    :return: an image represented by a matrix of type np.float64 with intensities normalized to [0,1]
    """
    rgb_img = imread(filename).astype("float64")
    if representation == GRAY_REP:
        rgb_img = rgb2gray(rgb_img)
    return rgb_img / 255


def build_im_dict(filenames, representation):
    images = dict.fromkeys(set(filenames))
    # images = images.fromkeys(filenames)
    for file in images.keys():
        # loading as a grayscale image in the [0,1] range (use read_image)
        images[file] = read_image(file, representation)
    return images


def corrupt_set(images, corruption_func):
    # we should instead first sample a larger random crop, of size 3 × crop_size, apply the corruption function
    # on it, and then take a random crop of the requested size from both original larger crop and the corrupted copy.
    cor_images = [None] * len(images)
    for i, im in enumerate(images):
        # corrupting the entire image with corruption_func(im)
        cor_images[i] = corruption_func(im)
    return np.array(cor_images)


def get_valid_indices(crop_size, src_im):
    indices = []
    for im in src_im:
        im_height, im_width = im.shape
        crop_height = np.random.choice(max(im_height - crop_size[0], 1))
        crop_width = np.random.choice(max(im_width - crop_size[1], 1))
        indices.append((crop_height, crop_width))
    return indices


def get_cropped_set(src_im, indices, crop_size):
    cropped_images = [None] * len(src_im)
    for i in range(len(src_im)):
        crop_init_y, crop_init_x = indices[i]
        cropped_images[i] = src_im[i][crop_init_y:crop_init_y + crop_size[0], crop_init_x:crop_init_x + crop_size[1]]
    return np.array(cropped_images)


def read_subset_image(filenames, representation):
    images = [None] * len(filenames)
    for i, file in enumerate(filenames):
        # loading as a grayscale image in the [0,1] range (use read_image)
        images[i] = read_image(file, representation)
    return np.array(images)


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """

    :param filenames: A list of filenames of clean images
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
                            and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: data_generator:
    -a Python’s generator object which outputs random tuples of the form (source_batch, target_batch),
    where each output variable is an array of shape (batch_size, height, width, 1).
    -source_batch is their respective randomly corrupted version according to corruption_func(im).
    -target_batch is made of clean images,
    """
    im_dict = build_im_dict(filenames, GRAY_REP)

    def data_generator():
        # Each image in the batch should be picked at random from the given list of filenames
        while True:
            # get random set of images
            src_im = read_subset_image(random.sample(list(im_dict), min(batch_size, len(filenames))), GRAY_REP)
            src_im = src_im - 0.5
            # get sets of cropped original images and corrupted images in size of 3 * crop_size
            triple_size = (crop_size[0] * 3, crop_size[1] * 3)
            triple_src_crops = get_cropped_set(src_im, get_valid_indices(triple_size, src_im), triple_size)
            triple_corrupt_crops = corrupt_set(triple_src_crops, corruption_func)
            triple_corrupt_crops = triple_corrupt_crops - 0.5
            # get sets of cropped original images and corrupted images in size of crop_size
            indices = get_valid_indices(crop_size, triple_src_crops)
            src_crops = get_cropped_set(triple_src_crops, indices, crop_size)
            corrupt_crops = get_cropped_set(triple_corrupt_crops, indices, crop_size)
            src_crops, corrupt_crops = src_crops.reshape(*src_crops.shape, 1), corrupt_crops.reshape(*corrupt_crops.shape, 1)
            yield (corrupt_crops, src_crops)
    return data_generator()


def resblock(input_tensor, num_channels):
    """
    Constructs a single residual block.
    :param input_tensor: a symbolic input tensor.
    :param num_channels: the number of channels for each of its convolutional layers.
    :return: output_tensor: the symbolic output tensor.
    """
    convolved = Conv2D(num_channels, KERNEL_SIZE, padding = 'same')(input_tensor)
    relu = Activation('relu')(convolved)
    convolved = Conv2D(num_channels, KERNEL_SIZE, padding = 'same')(relu)
    added = Add()([input_tensor, convolved])
    return Activation('relu')(added)


def build_nn_model(height, width, num_channels, num_res_blocks):
    """

    :param height: wanted height dimension of output.
    :param width: wanted width dimension of output.
    :param num_channels: wanted number of output channels.
    :param num_res_blocks: wanted number of residual blocks.
    :return: model: an untrained Keras model with input dimension the shape of (height, width, 1)4.
                    All convolutional layers (including residual blocks) with number of output channels equal to
                    num_channels, except the last convolutional layer which should have a single output channel.
                    The number of residual blocks should be equal to num_res_blocks.
    """
    input_tensor = Input((height, width, 1))
    convolved = Conv2D(num_channels, KERNEL_SIZE, padding = 'same')(input_tensor)
    relu = Activation('relu')(convolved)
    for _ in range(num_res_blocks):
        relu = resblock(relu, num_channels)
    convolved = Conv2D(1, KERNEL_SIZE, padding = 'same')(relu)
    output = Add()([input_tensor, convolved])
    return Model(inputs = input_tensor, outputs = output)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Divides the images into a training set and validation set, using an 80-20 split.
    From each set, generates a dataset with the given batch size and corruption function.
    Compiles the model and then calls fit_generator to actually train the model.
    :param model: A general neural network model for image restoration.
    :param images: A list of file paths pointing to image files. You should assume these paths are complete, and
                    should append anything to them.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
                    and returns a randomly corrupted version of the input image.
    :param batch_size: The size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: The number of update steps in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    """
    # Divides the images into a training set and validation set, using an 80-20 split.
    images = set(images)
    training_set_paths = random.sample(images, int(0.8 * len(images)))
    test_set_paths = list(images.difference(set(training_set_paths)))
    # From each set, generates a dataset with the given batch size and corruption function.
    test_gen = load_dataset(test_set_paths, batch_size, corruption_func, model.input_shape[1:3])
    training_gen = load_dataset(training_set_paths, batch_size, corruption_func, model.input_shape[1:3])
    # Call to the compile() method of the model using the “mean squared error” loss and ADAM optimizer.
    model.compile(loss = 'mean_squared_error', optimizer = Adam(beta_2 = 0.9))
    # Call fit_generator to actually train the model.
    model.fit_generator(training_gen, steps_per_epoch = steps_per_epoch, epochs = num_epochs,
                        validation_data = test_gen, validation_steps = num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    Create a new model that fits the size of the input image and has the same weights as the given base model.
    Use the predict() method to restore the image.
    Clip the results to [0, 1].
    :param corrupted_image: A grayscale image of shape (height, width) and with values in the [0, 1] range of
        type float64 (as returned by read_image), corrupted by the same corruption function encountered during training
    :param base_model: A neural network trained to restore small patches.
        The input and output of the network are images with values in the [−0.5, 0.5].
    :return: restored_image: the restored image.
    """
    corrupted_image = corrupted_image[:, :, np.newaxis] - 0.5
    input_tensor = Input(corrupted_image.shape)
    output_tensor = base_model(input_tensor)
    new_model = Model(inputs = input_tensor, outputs = output_tensor)
    restored = new_model.predict(corrupted_image[np.newaxis, ...])
    restored = restored[0, :, :, 0]
    restored = restored + 0.5
    return restored.astype("float64")


# =========== Denoising ===========

def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Randomly sample a value of sigma, uniformly distributed between min_sigma and max_sigma.
    Add to every pixel of the input image a zero-mean gaussian random variable with standard
    deviation equal to sigma.
    Normalize the values- round each pixel to the nearest fraction i/255 and clip to [0, 1].
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal variance
                    of the gaussian distribution.
    :return: corrupted: the corrupted image
    """
    sigma = np.random.uniform(min_sigma, max_sigma, 1)[0]
    return np.round((image + np.random.normal(0, sigma, image.shape)) * 255) / 255


def noise(image):
    return add_gaussian_noise(image, 0, 0.2)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    Builds an appropriate model, and returns the trained model.
    :param num_res_blocks: number of residual blocks.
    :param quick_mode: if true, different running parameters for train_model
    :return: model: the trained model
    """
    _model = build_nn_model(24, 24, 48, num_res_blocks)
    if not quick_mode:
        train_model(_model, sol5_utils.images_for_denoising(), noise, 100, 100, 5, 1000)
    else:
        train_model(_model, sol5_utils.images_for_denoising(), noise, 10, 3, 2, 30)
    return _model

# =========== Deblurring ===========


def add_motion_blur(image, kernel_size, angle):
    """
    Simulates motion blur on the given image.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return: corrupted: the blurred image.
    """
    blur_kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, blur_kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    Randomly choosing a blur an blurs the given image.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: corrupted: the blurred image.
    """
    # samples an angle at uniform from the range [0, π),
    angle = np.random.uniform(0, math.pi, 1)[0]
    # choses a kernel size at uniform from list_of_kernel_sizes
    kernel_size = list_of_kernel_sizes[np.random.randint(0, len(list_of_kernel_sizes), 1)[0]]
    # applying the previous function with the given image and the randomly sampled parameters.
    blurred = add_motion_blur(image, kernel_size, angle)
    # Before returning the image it should be rounded to the nearest fraction i/255 and clipped to [0, 1].
    return np.round(blurred * 255) / 255


def blur(image):
    return random_motion_blur(image, [7])


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    learns the blurring network.
    :param num_res_blocks: the number of residual blocks.
    :param quick_mode: if true, configurations are different.
    :return: model: the learned model
    """
    _model = build_nn_model(16, 16, 32, num_res_blocks)
    if not quick_mode:
        train_model(_model, sol5_utils.images_for_deblurring(), blur, 100, 100, 10, 1000)
    else:
        train_model(_model, sol5_utils.images_for_deblurring(), blur, 10, 3, 2, 30)
    return _model



