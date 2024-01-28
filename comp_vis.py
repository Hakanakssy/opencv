import cv2

# Görüntü işleme algoritması
tracker = cv2.TrackerCSRT_create()

# Video dosyasını yükle
video = cv2.VideoCapture("video.mp4")

# İlk kareyi al
ok, frame = video.read()

# Nesnenin konumunu seç
bbox = cv2.selectROI("Frame", frame, False)

# Nesneyi takip etmek için algoritmayı başlat
tracker.init(frame, bbox)

while True:
    # Bir sonraki kareyi al
    ok, frame = video.read()
    if not ok:
        break

    # Nesneyi takip et
    ok, bbox = tracker.update(frame)

    # Nesneyi çevreleyen dikdörtgeni çiz
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
    else:
        cv2.putText(frame, "Kayıp", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Görüntüyü ekranda göster
    cv2.imshow("Frame", frame)

    # Q tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bellekleri temizle
video.release()
cv2.destroyAllWindows()