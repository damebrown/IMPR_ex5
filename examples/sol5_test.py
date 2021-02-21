# from sol5 import *
import matplotlib.pyplot as plt

# model = learn_denoising_model(quick_mode= True)
# image = sol5.read_image('monkey_selfie.jpg', 1)
# corrupted = sol5.corrupt(image)
# restored_image = sol5.restore_image(corrupted, model)
# plt.figure()
# plt.imshow(restored_image)
# plt.show()

# debluring_minimal_error = [1554.47509765625, 1441.8397216796875, 1334.9981689453125, 1255.4984130859375, 1270.0421142578125]
# fig = plt.figure()
# plt.plot(debluring_minimal_error)
# plt.show()
# # plt.close(fig)
# # # debluring_minimal_error = [1426.9825439453125, 1434.0023193359375, 1701.814697265625, 1421.3165283203125, 1651.056396484375]
# # # fig = plt.figure()
# # # plt.plot(debluring_minimal_error)
# # # plt.show()
# # plt.close(fig)
# # # debluring_minimal_error = [1916.774658203125, 1414.17138671875, 1253.007568359375, 1642.3594970703125, 1282.948486328125]
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # # debluring_minimal_error = [1517.9730224609375, 1514.2779541015625, 1065.859375, 1332.20556640625, 1214.9210205078125]
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # # debluring_minimal_error = [1520.3388671875, 1446.27880859375, 1435.0975341796875, 1545.0107421875, 1435.10888671875]
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # # debluring_minimal_error = [1997.09033203125, 1254.394287109375, 1565.0758056640625, 1613.784423828125, 1451.7401123046875]
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # # debluring_minimal_error = [1962.8509521484375, 1420.5035400390625, 1691.83203125, 1365.58984375, 1425.57177734375]
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # # debluring_minimal_error = [1412.803955078125, 1500.2652587890625, 1335.4317626953125, 1158.7230224609375, 1433.3470458984375]
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # # debluring_minimal_error = [1474.7166748046875, 1580.5015869140625, 1563.1285400390625, 1362.920166015625, 1495.2049560546875]
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # debluring_minimal_error = [1242.9176025390625, 1499.9669189453125, 1326.6339111328125, 1327.78857421875, 1246.5875244140625]
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # debluring_minimal_error = [1358.1396484375, 1644.4317626953125, 1553.557861328125, 1439.2344970703125, 1264.752685546875],
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # debluring_minimal_error = [1382.434326171875, 1318.16259765625, 1165.3721923828125, 1284.5775146484375, 1280.37451171875],
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
# # debluring_minimal_error = [1607.1121826171875, 1585.4769287109375, 1231.1710205078125, 1514.2431640625, 1468.3018798828125],
# # fig = plt.figure()
# # plt.plot(debluring_minimal_error)
# # plt.show()
# # plt.close(fig)
#
# image = read_image("examples/163004_2_corrupted_0.10.png", 1)
#
#
# bool = True
# if bool:
#     dn_model = learn_denoising_model()
#     dn_model.save("datasets/model_denoising")
# else:
#     dn_model = Model()
#     dn_model.load_weights("datasets/model_weights_denoising")
#     dn_model.save("datasets/model_denoising")
#     # dn_model = load_model("datasets/model_weights_denoising")
# dn_res = restore_image(image, dn_model)
# plt.imshow(dn_res, cmap = 'gray')
# plt.imsave('dn_restored', dn_res)
# plt.show()
#
#
# db_im = read_image("examples/0000018_2_corrupted.png", 1)
# if bool:
#     db_model = learn_deblurring_model()
#     db_model.save("datasets/model_deblurring")
# else:
#     db_model = Model()
#     db_model.load_weights("datasets/model_weights_deblurring")
#     db_model.save("datasets/model_weights_deblurring")
#     # db_model = load_model("datasets/model_weights_deblurring")
# db_res = restore_image(db_im, db_model)
# plt.imsave('db_restored', db_res)
# plt.imshow(db_res, cmap = 'gray')
# plt.show()
#



# db = [None, 0.03105640970170498, 0.02610638365149498, 0.023839259520173073, 0.023171432316303253, 0.02186286263167858]
dn = [None, 0.00395783968269825, 0.003426595591008663, 0.0032076260540634394, 0.003152331104502082, 0.0033155817072838545]

# plt.plot(db)
# plt.title("deblurring")
# plt.show()
plt.plot(dn)
plt.title("denoising")
plt.show()
