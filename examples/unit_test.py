from sol5 import *
import sol5_utils
import matplotlib.pyplot as plt
import time

REP_GRAY = 1


def show_im(im, is_gray = False, text = ""):
    if is_gray:
        if im.ndim == 3:
            plt.imshow(im[:, :, 0], cmap = 'gray')
        else:
            plt.imshow(im, cmap = 'gray')
    else:
        plt.imshow(im)
    plt.title(text)
    plt.show()


def test_load_dataset():
    CROP_SIZE = 300
    filenames = sol5_utils.images_for_denoising()
    batch_size = 1
    corruption_func = lambda im: add_gaussian_noise(im, 0, 0.2)
    crop_size = (CROP_SIZE, CROP_SIZE)
    generator = load_dataset(filenames, batch_size, corruption_func, crop_size)
    source_batch, target_batch = next(generator)
    assert source_batch.shape == (batch_size, CROP_SIZE, CROP_SIZE, 1)
    assert target_batch.shape == (batch_size, CROP_SIZE, CROP_SIZE, 1)
    show_im(source_batch[0], True)
    show_im(target_batch[0], True)


def test_add_gaussian_noise():
    name = sol5_utils.images_for_denoising()[np.random.randint(100)]
    im = read_image(name, REP_GRAY)
    show_im(im, True)
    im = add_gaussian_noise(im, 0, 0.2)
    show_im(im, True)
    print(f'add_gaussian_noise to image {name}')


def test_random_motion_blur():
    name = sol5_utils.images_for_deblurring()[np.random.randint(100)]
    im = read_image(name, REP_GRAY)
    show_im(im, True)
    im = random_motion_blur(im, [7])
    show_im(im, True)
    print(f'random_motion_blur on image {name}')


def test_add_motion_blur():
    name = sol5_utils.images_for_denoising()[np.random.randint(100)]
    im = read_image(name, REP_GRAY)
    show_im(im, True)
    im = add_motion_blur(im, 7, np.pi / 2)
    show_im(im, True)
    print(f'add_motion_blur on image {name}')


def weights_file_name(num_res_block = 5, quick_mode = False, type = 'no_type'):
    return sol5_utils.relpath(f"weights/{type}_blocks-{num_res_block}_quickmode-{quick_mode}")


def learn_denoising_model_and_save(num_res_block = 5, quick_mode = False):
    start = time.time()
    model = learn_denoising_model(num_res_block, quick_mode)
    model.save_weights("datasets/model_weights_denoising")
    print(f'time = {(time.time() - start)} seconds')


def test_learn_denoising_model(num_res_block = 5, quick_mode = False):
    model = build_nn_model(24, 24, 48, num_res_block)
    model.load_weights("datasets/model_weights_denoising")

    name = sol5_utils.images_for_denoising()[np.random.randint(100)]
    image = read_image(name, REP_GRAY)

    cur_image = add_gaussian_noise(image, 0, 0.2)

    fixed_im = restore_image(cur_image, model)

    difference = cur_image - fixed_im
    show_im(image, True, 'original')
    show_im(cur_image, True, 'currpted')
    show_im(fixed_im, True, 'fixed')
    print(f'avarage change is {np.average(difference)}')
    print(f'Standard deviation of change is {np.std(difference)}')


def learn_deblurring_model_and_save(num_res_block = 5, quick_mode = False):
    start = time.time()
    model = learn_deblurring_model(num_res_block, quick_mode)
    model.save_weights("datasets/model_weights_deblurring")
    print(f'time = {(time.time() - start)} seconds')


def test_learn_deblurring_model(num_res_block = 5, quick_mode = False):
    model = build_nn_model(16, 16, 32, num_res_block)
    model.load_weights("datasets/model_weights_deblurring")

    name = sol5_utils.images_for_deblurring()[np.random.randint(100)]
    image = read_image(name, REP_GRAY)

    cur_image = random_motion_blur(image, [7])

    fixed_im = restore_image(cur_image, model)

    show_im(image, True, 'original')
    show_im(cur_image, True, 'currpted')
    show_im(fixed_im, True, 'fixed')



# test_load_dataset()
# test_add_gaussian_noise()
# test_random_motion_blur()

#
# learn_denoising_model_and_save(5, False)
# test_learn_denoising_model()
#
#
# learn_deblurring_model_and_save(5, False)
# test_learn_deblurring_model()
