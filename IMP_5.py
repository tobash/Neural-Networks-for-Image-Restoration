
from . import sol5_utils
from tensorflow.python.keras.layers import Input,Conv2D,Add,Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

import numpy as np

from scipy.ndimage.filters import convolve
from scipy.misc import imread
from skimage.color import rgb2gray


MAX_GRAY_LEVEL = 256
GRAYSCALE_MODE = 1


def read_image(filename, registration):

    """
    the function reads an image and represents it in a given type
    notice - a gray scale image will always be returned as a flat image (not YIQ)
    :param filename: the file's name...
    :param registration: 1 for gray scale mode. 2 for RGB mode
    :return: an image of the given registration.
    """

    if registration == GRAYSCALE_MODE:
        image = imread(filename)
        image = rgb2gray(image)
        image = image.astype(np.float64)
    else:
        image = imread(filename)
        image /= MAX_GRAY_LEVEL
    float_image = image.astype(np.float64)
    return float_image


def take_patch(im,x,y,x_range,y_range):

    '''
    Helper function that takes a patch out of a given image
    :param im: the image that the patch should be taken from
    :param x: x starting point
    :param y: y starting point
    :param x_range: size by the x parameter
    :param y_range: size by the y parameter
    :return: the patch
    '''

    return im[y:y+y_range,x:x+x_range]



def load_dataset(filenames, batch_size, corruption_func, crop_size):

    """
    A function building a dataset of pairs of patches comprising (i) an
    original, clean and sharp, image patch with (ii) a corrupted version of
    same patch. Given a set of images, each time picking a random image,
    applying a random corruption, and extracting a random patch.
    :param filenames: list of filenames of clean images
    :param batch_size: size of the batch of images for each iteration of
    Stochastic Gradient Descent
    :param corruption_func: function receiving a numpy’s array representation
    of an image as a single argument, and returns a randomly corrupted version
    of the input image
    :param crop_size: tuple (height, width) specifying the crop size of the
    patches to extract
    :return: generator object which outputs random tuples of the form
    (source_batch, target_batch)
    """

    im_dict = {}
    source_im = np.zeros([batch_size, crop_size[0], crop_size[1], 1])  # check
    target_im = np.zeros([batch_size, crop_size[0], crop_size[1], 1])  # check

    while True:

        file_index = np.random.randint(len(filenames),size = batch_size)

        for i in range(file_index.shape[0]):
            name_im = filenames[file_index[i]]

            if name_im in im_dict:
                im = im_dict[name_im]
            else:
                im = read_image(filenames[file_index[i]], GRAYSCALE_MODE)
                im_dict[name_im] = im

            x_val_3 = np.random.randint(im.shape[1] - crop_size[1]*3)
            y_val_3 = np.random.randint(im.shape[0] - crop_size[0]*3)

            im = take_patch(im,x_val_3,y_val_3,crop_size[1]*3,crop_size[0]*3)
            corrupt_im = corruption_func(im)

            x_val = np.random.randint(im.shape[1]- crop_size[1])
            y_val = np.random.randint(im.shape[0]- crop_size[0])
            source_im[i] = take_patch(corrupt_im, x_val, y_val, crop_size[1], crop_size[0])[:,:,None]
            target_im[i] = take_patch(im,x_val, y_val, crop_size[1], crop_size[0])[:,:,None]

        yield (source_im-0.5,target_im-0.5)


def resblock(input_tensor, num_channels):

    '''
    The function takes as input a symbolic input tensor and the number of
    channels for each of its convolutional layers, and returns the symbolic
    output tensor.
    :param input_tensor: a symbolic input tensor
    :param num_channels: the number of output channels at each convolution
    layer.
    :return: symbolic output tensor
    '''

    b = Conv2D(num_channels, (3, 3), padding="same")(input_tensor)
    b = Activation('relu')(b)
    b = Conv2D(num_channels, (3, 3), padding="same")(b)
    b = Add()([input_tensor, b])
    b = Activation('relu')(b)

    return b

def build_nn_model(height, width, num_channels, num_res_blocks):

    '''
    The function builds an untrained Keras model with input dimension the shape
    of (height, width, 1) and the corresponding output.
    :param height: input height
    :param width: input width
    :param num_channels: number of output channels except the last convolutional
    layer which have a single output channel
    :param num_res_blocks: The number of residual blocks
    :return: the build nn model
    '''

    a = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding="same")(a)
    b = Activation('relu')(b)
    for i in range(num_res_blocks):
        b = resblock(b, num_channels)
    b = Conv2D(1, (3, 3), padding="same")(b)
    b = Add()([a, b])
    model = Model(inputs=a, outputs=b)

    return model

def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):

    '''
    A function that train a neural network model on a given training set,
    dividing the images into a training set and validation set, using an 80-20
    split.
    :param model: general neural network model for image restoration
    :param images: list of file paths pointing to image files.
    :param corruption_func: function receiving a numpy’s array representation
    of an image as a single argument, and returns a randomly corrupted version
    of the input image
    :param batch_size: size of the batch of examples for each iteration of SGD
    :param steps_per_epoch: number of update steps in each epoch
    :param num_epochs: number of epochs for which the optimization will run
    :param num_valid_samples: number of samples in the validation set to test
    on after every epoch
    '''

    crop_param = model.input_shape[1:3]
    training = images[int(len(images)*0.2):]
    training_generator = load_dataset(training, batch_size, corruption_func,crop_param)
    validation = images[:int(len(images)*0.2)]
    validation_generator = load_dataset(validation,batch_size,corruption_func,crop_param) #todo should use num_valid_samples instead of batch size?
    model.compile(optimizer=Adam(beta_2=0.9),loss='mean_squared_error')
    model.fit_generator(training_generator ,steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs, validation_data=validation_generator
                        ,validation_steps=num_valid_samples)

    return model



def restore_image(corrupted_image, base_model):

    '''
    A function that restore full images of any size, creating a new network for each
    given image, then uses the new model to restore the given full image.
    :param corrupted_image: a grayscale image of shape (height, width) and with
     values in the [0, 1] range of type float64
    :param base_model: a neural network trained to restore small patches
    :return: restored image
    '''

    corrupted_im = corrupted_image[:,:,None]
    a = Input(shape=(corrupted_im.shape))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    n_im = new_model.predict(corrupted_im[None,:,:,:]-0.5).astype(np.float64)
    n_im = n_im.reshape(n_im.shape[1],n_im.shape[2])
    n_im = np.clip(n_im + 0.5,0,1)
    return n_im


def add_gaussian_noise(image, min_sigma, max_sigma):

    '''
    A function that randomly sample a value of sigma, uniformly distributed
    between min_sigma and max_sigma, followed by adding to every pixel of the
    input image a zero-mean gaussian random variable with standard deviation
    equal to sigma
    :param image: grayscale image with values in the [0, 1] range of type float64
    :param min_sigma: a non-negative scalar value representing the minimal
     variance of the gaussian distribution
    :param max_sigma: a non-negative scalar value larger than or equal to
    min_sigma, representing the maximal variance of the gaussian distribution
    :return: the corrupted image
    '''

    sigma = np.random.uniform(min_sigma,max_sigma)
    vals = np.random.normal(0,sigma,image.shape)
    corrupt_im = np.round((image + vals)*255)/255
    corrupt_im = np.clip(corrupt_im, 0, 1)
    return corrupt_im



def learn_denoising_model(num_res_blocks=5, quick_mode=False):

    '''

    A function that builds and train a network which expect patches of size 24×24,
    using 48 channels for all but the last layer and a corruption function of
    gaussian noise.
    :param num_res_blocks: a default value of 5.
    :param quick_mode: a default value of quick mode to train the model.
    :return: a trained model for denoising image with gaussian noise
    '''

    img_lst = sol5_utils.images_for_denoising()
    new_corrupt_func = lambda img:add_gaussian_noise(img,0,0.2)
    model = build_nn_model(24, 24, 48, num_res_blocks)

    if not quick_mode:
        trained_model = train_model(model,img_lst,new_corrupt_func,100,100,5,1000)
    else:
        trained_model = train_model(model, img_lst, new_corrupt_func,10,3,2,30)

    return trained_model


def add_motion_blur(image, kernel_size, angle):

    '''
    A function that simulate motion blur on the given image using a square
    kernel of size kernel_size where the line has the given angle in radians,
    measured relative to the positive horizontal axis.
    :param image: a grayscale image with values in the [0, 1] range of type float64
    :param kernel_size: an odd integer specifying the size of the kernel
    :param angle: an angle in radians in the range [0, π)
    :return: the corrupted image with the motion blur.
    '''

    kernel = sol5_utils.motion_blur_kernel(kernel_size,angle)
    corrupt_im = convolve(image,kernel)
    return corrupt_im

def random_motion_blur(image, list_of_kernel_sizes):

    '''
    The function samples an angle at uniform from the range [0, π),
    and choses a kernel size at uniform from the list list_of_kernel_sizes,
    followed by applying the previous function with the given image and the
    randomly sampled parameters.
    :param image:a grayscale image with values in the [0, 1] range of type float64
    :param list_of_kernel_sizes: a list of odd integers
    :return: the corrupted image with the motion blur.
    '''

    angel = np.random.uniform(0,np.pi)
    corrupt_im = add_motion_blur(image,np.random.choice(list_of_kernel_sizes),angel)
    corrupt_im = np.round( corrupt_im * 255) / 255

    corrupt_im = np.clip(corrupt_im,0,1)
    return corrupt_im

def learn_deblurring_model(num_res_blocks=5, quick_mode=False):

    '''
    A function that builds and train a network in order to resrore images with
    a motion blur. The network expect patches of size 16×16, and have 32
    channels in all layers except the last one.
    :param num_res_blocks: a default value of 5.
    :param quick_mode: a default value of quick mode to train the model.
    :return: a trained model for deblurring image with motion blur.
    '''

    img_lst = sol5_utils.images_for_deblurring()
    new_corrupt_func = lambda img: random_motion_blur(img,[7])
    model = build_nn_model(16, 16, 32, num_res_blocks)

    if not quick_mode:
        trained_model = train_model(model, img_lst, new_corrupt_func, 100, 100, 10, 1000)
    else:
        trained_model = train_model(model, img_lst, new_corrupt_func, 10, 3, 2, 30)

    return trained_model
