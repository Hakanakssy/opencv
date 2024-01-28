import cv2
import numpy as np

# Görüntüyü yükle
img = cv2.imread('kup.jpg', cv2.IMREAD_GRAYSCALE)

# Laplace operatörü için kernel oluştur
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

# Laplace operatörü ile kenarları tespit et
edges = cv2.filter2D(img, -1, kernel)

# Görüntüyü göster
cv2.imshow('Original Image', img)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
