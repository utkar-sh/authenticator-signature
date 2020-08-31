import numpy as np
import cv2

# load the file to be checked
img  = cv2.imread('Test/sign (10).jpg')
img_final = cv2.imread('Test/sign (10).jpg')
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 110, 255, cv2.THRESH_BINARY)
image_final = cv2.bitwise_and(img2gray , img2gray , mask =  mask)
ret, new_img = cv2.threshold(image_final, 110 , 255, cv2.THRESH_BINARY_INV)  # for white text , cv.THRESH_BINARY
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3 , 3))
dilated = cv2.dilate(new_img,kernel,iterations = 9)

_,contours, hierarchy = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # get contours
index = 0
for contour in contours:

    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    #Don't plot small false positives that aren't text
    if w < 100 and h<100:
        continue

    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

    cropped = img_final[y :y +  h , x : x + w]
    s = 'final_' + 'crop_' + str(index) + '.jpg'
    cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(cropped, 140, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(cropped, cropped, mask=mask)
    ret, new_img = cv2.threshold(image_final, 140, 255, cv2.THRESH_BINARY)
    index = index + 1

# loading the trained data files
samples=np.loadtxt('generalsamples.data',np.float32)
responses=np.loadtxt('generalresponses.data',np.float32)
responses=responses.reshape((responses.size,1))

# applying K-Nearest Neighbours algorithm
model= cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)
roi = cropped
roismall = cv2.resize(roi, (25, 25))
roismall = roismall.reshape((1, 625))
roismall = np.float32(roismall)
retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
text = str(int((results[0][0])))

if text=='1':
    print('Signature is Authenticated Successfully.')
elif text=='0':
    print('Signature Authentication Failed.')