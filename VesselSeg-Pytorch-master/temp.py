from lib.pre_processing import my_PreProc, rgb2gray, dataset_normalized, clahe_equalized
import matplotlib.pyplot as plt
import numpy as np
import cv2

imgs = r"C:\Users\hasee\Desktop\image58.tif"

imgp = cv2.imread(imgs)

img_new2 = imgp[:, :, ::-1]

plt.imshow(img_new2)
plt.show()
imgp = np.expand_dims(img_new2, 0)
imgp = np.transpose(imgp, (0, 3, 1, 2))
img_r = rgb2gray(imgp)
img_z = dataset_normalized(img_r)
train_imgs = clahe_equalized(img_z)
img_p = img_r[0]
img_p = np.reshape(img_p, (img_p.shape[1], img_p.shape[2]))
plt.imshow(img_p,cmap='gray')

plt.show()
print("")


#readme img_p是处理后的图 img r是灰度图  img z是正则化 train imgs是直方图均衡化,想要看哪个就在imp=后边加哪个
