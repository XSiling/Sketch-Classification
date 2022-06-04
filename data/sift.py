import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import pickle


def sift(img, standard):
    sift = cv2.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(img, None)
    if standard:
        descriptor = StandardScaler().fit_transform(descriptor)
    # print(descriptor.shape)
    # print('discriptor: ', descriptor)
    return descriptor


train_data = np.load('./dataset/train_data.npy')
# train_img = train_data[0]
# sift_feature = []
# sift_feature.append(sift(train_data[0], 1))
# sift_feature.append(sift(train_data[1], 1))
# print(sift_feature)



train_data_sift = []
for train_img in train_data:
    print(train_img.shape)
    sift_feature = sift(train_img, 0)
    train_data_sift.append(sift_feature)

f = open('./dataset/train_data_sift.pkl', 'wb')
pickle.dump(train_data_sift, f)
f.close()
# obj = pickle.load(f)
# print(obj)


# valid_data = np.load('./dataset/valid_data.npy')
# valid_data_sift = []
# for valid_img in valid_data:
#     print(valid_img.shape)
#     sift_feature = sift(valid_img, 0)
#     valid_data_sift.append(sift_feature)
# valid_data_sift = np.array(valid_data_sift)
# np.save('./dataset/valid_data_sift.npy', valid_data_sift)


