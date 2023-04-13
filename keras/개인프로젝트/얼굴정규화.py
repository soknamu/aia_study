import cv2
path = "d:study_data/asian_data/2/"
# 이미지 로드
img = cv2.imread(path + '00000A02.jpg')

# 얼굴 검출 모델 로드
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 이미지에서 얼굴 영역 검출
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 얼굴 크기를 조정하여 정규화된 이미지 생성
for (x, y, w, h) in faces:
    roi = img[y:y+h, x:x+w]  # 얼굴 영역 추출
    roi_resized = cv2.resize(roi, (128, 128))  # 얼굴 크기 조정
    cv2.imshow('roi_resized', roi_resized)
    cv2.waitKey(0)

cv2.destroyAllWindows()
