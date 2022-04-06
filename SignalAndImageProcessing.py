'''import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
def main(): 
    img = cv2.imread('image.jpg')
    img_color = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    # View image
    cv2.imshow('YCBCR',img)
    cv2.imshow('RGB',img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()
'''
'''import cv2
import numpy as np
from matplotlib import pyplot as plt 
 
##### Gray Histogram ###
def gray_hst(): 
     image = cv2.imread('image.jpg')
     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
     plt.plot(histogram, color='k')
     cv2.imshow("Original", image)
     plt.show()
     cv2.waitKey(0) 
     cv2.destroyAllWindows() 
 
#### RGB Histogram ####
def rgb_hst(): 
    image = cv2.imread('image.jpg', -1)
    b, g, r = cv2.split(image)
    cv2.imshow("Original", image)
    cv2.imshow("Blue", b),
    cv2.imshow("Green", g)
    cv2.imshow("Red", r) 
    plt.hist(b.ravel(), 256, [0, 256])
    plt.hist(g.ravel(), 256, [0, 256])
    plt.hist(r.ravel(), 256, [0, 256]) 
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
 
#### Equalization ####
def equ(): 
    img = cv2.imread('image.jpg',0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ))
    plt.plot(res)
    cv2.imshow("Equalization", res)
    plt.show()
    cv2.imwrite('res.png',res)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
 
#### Stretching ####
def stret(): 
    img = cv2.imread('image.jpg')
    original = img.copy()
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)
    cv2.imshow("Original", original)
    cv2.imshow("Output", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

while 1: 
    print("Choose your option : ")
    print("1. Gray Histogram")
    print("2. RGB Histogram")
    print("3. Histogram Equalization")
    print("4. Histogram Stretching") 
    
    ch = int(input())
    if ch == 1:
        gray_hst()
    elif ch == 2:
        rgb_hst()
    elif ch == 3: 
        equ()
    elif ch == 4:
        stret()
    else: 
        print("Invalid Option !!!") '''
'''
import cv2 
import numpy as np
from matplotlib import pyplot as plt

def median(): 
    img = cv2.imread('image.png') 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    blur = cv2.medianBlur(img,3) 
    plt.subplot(121),plt.imshow(img),plt.title('Original') 
    plt.xticks([]), plt.yticks([]) 
    plt.subplot(122),plt.imshow(blur),plt.title('Median') 
    plt.xticks([]), plt.yticks([]) 
    plt.show() 

def gauss(): 
    img = cv2.imread('image.png') 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    blur = cv2.GaussianBlur(img,(5,5),0) 
    plt.subplot(121),plt.imshow(img),plt.title('Original') 
    plt.xticks([]), plt.yticks([]) 
    plt.subplot(122),plt.imshow(blur),plt.title('Gausian') 
    plt.xticks([]), plt.yticks([]) 
    plt.show() 

def mean(): 
    img = cv2.imread('image.png') 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    blur = cv2.blur(img,(7,7)) 
    plt.subplot(121),plt.imshow(img),plt.title('Original') 
    plt.xticks([]), plt.yticks([]) 
    plt.subplot(122),plt.imshow(blur),plt.title('Mean') 
    plt.xticks([]), plt.yticks([]) 
    plt.show() 

while 1: 
    print("#------List------#") 
    print("1. Median Filter") 
    print("2. Gaussian Filter")
    print("3. Mean Filter")
    ch = int(input("Choose your opinion : "))
    if ch == 1: 
        median() 
    elif ch == 2: 
        gauss() 
    elif ch == 3: 
        mean() 
    else: 
        print("Invalid Option !!!")
'''
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Global Variable
img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
out = None

def twoDfilter(src, kernel):
    try:
        m, n = kernel.shape
    except ValueError:
        m = kernel.shape[0]
    d = int((m-1)/2)
    h, w = src.shape[0], src.shape[1]
    dst = np.zeros((h, w))
    for y in range(d, h - d):
        for x in range(d, w - d):
            dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)
    return dst

def edge_gradient():
    #Manual Gradient Edge Detector Calculation
    kernel_x = np.array([-1, 0, 1])
    kernel_y = np.array([[-1],[0],[1]])
    #Library Gradient Edge Detector Calculation
    lib_x = cv2.filter2D(gray,-1,kernel_x)
    lib_y = cv2.filter2D(gray,-1,kernel_y)
    return lib_x,lib_y,kernel_x,kernel_y

def edge_sobel():
    #Manual Sobel Edge Detector Calculation
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    #Library Sobel Edge Detector Calculation
    lib_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
    lib_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
    return lib_x,lib_y,kernel_x,kernel_y

def main():
    grad_lx, grad_ly, grad_kx, grad_ky = edge_gradient()
    sob_lx, sob_ly, sob_kx, sob_ky = edge_sobel()

    gray_grad_x = twoDfilter(gray, grad_kx)
    gray_grad_y = twoDfilter(gray, grad_ky)
    gray_sob_x = twoDfilter(gray, sob_kx)
    gray_sob_y = twoDfilter(gray, sob_ky)

    grad_out = np.sqrt(gray_grad_x ** 2 + gray_grad_y ** 2)
    sob_out = np.sqrt(gray_sob_x ** 2 + gray_sob_y ** 2)

    grad_outlib = grad_lx + grad_ly
    sob_outlib = np.sqrt(sob_lx ** 2 + sob_ly ** 2)

    plt.figure(figsize=(8*5, 8*5), constrained_layout=False)

    plt.subplot(241), plt.imshow(gray, cmap = 'gray'), 
    plt.title("Grey Image", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(242), plt.imshow(gray_grad_x, cmap = 'gray'), 
    plt.title("Horizontal No-lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(243), plt.imshow(gray_grad_y, cmap = 'gray'), 
    plt.title("Vertical No-lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(244), plt.imshow(grad_out, cmap = 'gray'), 
    plt.title("Gradient Edge No-lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(246), plt.imshow(grad_lx, cmap = 'gray'), 
    plt.title("Horizontal Lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(247), plt.imshow(grad_ly, cmap = 'gray'), 
    plt.title("Vertical Lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(248), plt.imshow(grad_outlib, cmap = 'gray'), 
    plt.title("Gradient Edge Lib", fontsize=9), plt.xticks([]), plt.yticks([])

    plt.figure(figsize=(8*5, 8*5), constrained_layout=False)

    plt.subplot(241), plt.imshow(gray, cmap = 'gray'), 
    plt.title("Grey Image", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(242), plt.imshow(gray_sob_x, cmap = 'gray'), 
    plt.title("Horizontal No-lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(243), plt.imshow(gray_sob_y, cmap = 'gray'), 
    plt.title("Vertical No-lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(244), plt.imshow(sob_out, cmap = 'gray'), 
    plt.title("Sobel Edge No-lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(246), plt.imshow(sob_lx, cmap = 'gray'), 
    plt.title("Horizontal Lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(247), plt.imshow(sob_ly, cmap = 'gray'), 
    plt.title("Vertical Lib", fontsize=9), plt.xticks([]), plt.yticks([])
    plt.subplot(248), plt.imshow(sob_outlib, cmap = 'gray'), 
    plt.title("Sobel Edge Lib", fontsize=9), plt.xticks([]), plt.yticks([])

    plt.show()

# Run
main()'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Fuction harris detection
def harris_det():
    filename = 'image.jpg'
    img = cv.imread(filename)
    original = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    
    # Find corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    plt.subplot(231), plt.imshow(original), plt.title("Original")
    plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(dst), plt.title("Corner")
    plt.xticks([]), plt.yticks([])

    # Harris Corners
    corner = cv.dilate(dst,None, iterations=3)
    img[corner>0.01*corner.max()]=[0,0,255]
    harris = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.subplot(233), plt.imshow(harris), plt.title("Harris Detection")
    plt.xticks([]), plt.yticks([])    
    plt.show()

# Run
harris_det()
