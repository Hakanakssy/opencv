import cv2
import numpy as np
filename = 'kizkulesi.jpg'
img = cv2.imread(filename)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Harris_detector = cv2.cornerHarris(np.float32(gray_img), 2, 3, 0.04)
img[Harris_detector > 0.01 * Harris_detector.max()] = [0, 0, 255]
cv2.imshow('Harris Kose isaretleri', img)
cv2.waitKey()