import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

frame1 = np.array( Image.open("1/0001.png") )
# frame1 = cv2.imread("1/0001.png")

detector = cv2.AKAZE_create()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)

# frame2 = cv2.imread("1/0010.png")
frame2 = np.array( Image.open("1/0010.png") )

# detector = cv2.AKAZE_create()
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)

# plt.imshow(gray1)

#Brute-Force
bf = cv2.BFMatcher()
#matchesの要素数は、queryDescriptorsの要素数に一致
#上位k個の特徴点を返す
matches = bf.knnMatch(queryDescriptors = descriptors1, trainDescriptors = descriptors2, k=2)

good = sorted(matches, key = lambda x : x[1].distance)

img3 = cv2.drawMatchesKnn(frame1, keypoints1, frame2, keypoints2, good[:10], None, flags=2)
# plt.figure(figsize=(15,15))
# plt.imshow(img3)

src_points = np.float32([ keypoints1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_points = np.float32([ keypoints2[m[1].trainIdx].pt for m in good ]).reshape(-1,1,2)
#src_points/dst_pointsは、good_matchesからindexでアクセスしたquery/trainのkeypoint座標である
#(求めたいMの順にsrc_points/dst_pointsの入力順を決める)
M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
size=(1080,925)
frame2_trans = cv2.warpPerspective(frame2, np.linalg.inv(M), size)
#frame1_trans = cv2.warpPerspective(frame1, M, size)

plt.imshow(frame2_trans)
Image.fromarray(frame2_trans).save("fire.png")
