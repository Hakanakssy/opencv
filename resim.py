import cv2
import numpy as np  #ilgili kütühanelerin tanımlanması

img = cv2.imread('resim.png') #kullanılacak görselin tanımlanması

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #görselin gri tonlama hale getirilmesi

dst = cv2.cornerHarris(gray,2,1,0) # blok boyutu 2, piksel çekirdek boyutu 1, harris algoritma parametresi 0
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,1)) #daha önceki denemelerde elips ve dikdörtgen çekirdek denedikten sonra haç şekli işlem yaptığım iki görselde de kanaat getirdiğim en iyi sonucu aldım.
dst = cv2.dilate(dst, kernel) #tespit edilen köşelerin etrafındaki alanı genişletmek için fonksiyonun tanımlanması.

threshold = 0.01*dst.max() #tespit edilen köşelerin işaretlenmesi
for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
        if dst[i,j] > threshold:
            cv2.circle(img,(j,i),1,(0,0,255),1)


#cv2.imshow('Gri tonlama',gray) #gri tonlamalı görseli görmek için çıktı kodu
cv2.imshow('Harris Edge Detection',img) # çalışmanın sonucu
cv2.waitKey(0) #açılan ekranın bekleme tanımı
cv2.destroyAllWindows() #açılan ekranların kapanışı
