import numpy as np
import cv2
import matplotlib.pyplot as plt

# 원본 이미지
img= cv2.imread("PATH", 0)
# 90도 시계방향으로 회전시킨 이미지
img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 


#필터 (3x3필터 & 가버필터)
#3x3 gabor필터
filter3x3 = np.array([[1, 1, 1],[1, -9, 1],[1, 1, 1]])
#3x3 gabor필터
gabor = cv2.getGaborKernel((3,3), 1, 1, 15, 0, 0, cv2.CV_32F)
    
    
#컨볼루션 연산 실행 함수
def doConvolution(image, kernel):
    xFilter, yFilter = kernel.shape   #필터 크기
    xImage, yImage = image.shape         #이미지 크기
    
    x = xImage - xFilter + 1
    y = yImage - yFilter + 1
    result = np.zeros((x, y))            #필터 적용 후 이미지 저장 공간
    
    for i in range(x):
        for j in range(y):
           result[i][j] = np.sum(kernel * image[i: i + xFilter, j: j + yFilter])
    return result


#필터 적용
#3x3 커스텀필터
img_f = doConvolution(img,filter3x3)
img_rot_f = doConvolution(img_rot, filter3x3)
#3x3 gabor 필터
img_g = doConvolution(img,gabor)
img_rot_g = doConvolution(img_rot, gabor)


#결과물 출력
plt.subplot(551), plt.axis('off'), plt.imshow(img)
plt.subplot(552), plt.axis('off'), plt.imshow(img_f, cmap='gray')
plt.subplot(553), plt.axis('off'), plt.imshow(img_rot_f, cmap='gray')
plt.subplot(554), plt.axis('off'), plt.imshow(img_g, cmap='gray')
plt.subplot(555), plt.axis('off'), plt.imshow(img_rot_g, cmap='gray')

plt.show()