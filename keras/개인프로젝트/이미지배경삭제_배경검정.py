# rembg 패키지에서 remove 클래스 불러오기
from rembg import remove 

# PIL 패키지에서 Image 클래스 불러오기
from PIL import Image 
path = "./asian_data/2/"

# Lenna.png파일 불러오기
img = Image.open(path + "00010A02.jpg") 

# 배경 제거하기
out = remove(img) 

# 변경된 이미지 저장하기
out = out.convert('RGB')
out.save(path + "00010A02_remove.jpg")

