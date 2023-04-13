import os
import numpy as np
from PIL import Image
from rembg import remove

for i in range(5,6):
    input_dir = f"d:study_data/_data/asian_data_predict/{i}/"
    output_dir = f"d:study_data/_data/asian_data_predict/{i}/"

    # 존재하지 않는 경우 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 입력 디렉토리의 모든 파일을 반복
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            
            # 이미지 불러오기
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # 배경 삭제
            out = remove(img)

            # 출력 이미지를 RGBA 모드로 변환
            out = out.convert('RGBA')

            # 이미지에서 NumPy 배열 만들기
            arr = np.array(out)

            # 임계값 미만의 픽셀에 대해 알파 채널을 0으로 설정
            arr[(arr[:,:,0] < 1) & (arr[:,:,1] < 1) & (arr[:,:,2] < 1)] = [0, 0, 0, 0]

            # 수정된 배열에서 새 Image 개체생성
            out = Image.fromarray(arr)

            # 투명한 배경으로 이미지 저장
            output_path = os.path.join(output_dir, filename.split(".")[0] + "_transparent.png")
            out.save(output_path)