from sol5 import *
import matplotlib.pyplot as plt
from unit_test import show_im


# err = []
# err_arr = []
# print('#########################   depth effect in deblurring  #########################')
# for _ in range(10):
#     for i in range(1, 6):
#         print('#########################   depth: ' + str(i) + '   #########################')
#         _model = learn_deblurring_model(i, False)
#         err.append(_model.history.history['val_loss'][-1])
#         # _model.save_weights("datasets/model_weights_deblurring")
#     print(err)
#     err_arr.append(err)
# print(err_arr)

print('#########################   depth effect in denoising   #########################')
err = []
im = read_image("C:/Users/user/Documents/2nd/IMPR/ex5-daniel.brown1/examples/0000018_1_original.png", 1)
for i in range(1, 6):
    print('#########################   depth: ' + str(i) + '   #########################')
    _model = learn_denoising_model(i, False)
    err.append(_model.history.history['val_loss'][-1])
    show_im(im, True, "depth: " + str(i))
    print(err)

plt.plot(err)
plt.show()

# res = [2077.008544921875, 2472.668212890625, 1630.447021484375, 1612.78466796875, 1808.88525390625]