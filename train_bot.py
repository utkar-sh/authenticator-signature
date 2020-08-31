import numpy as np
import cv2
import os, os.path

# getting number of correct and incorrect images in the respective training folders
Correct_path = 'Correct'
Incorrect_path = 'Incorrect'
count_correct=len([name for name in os.listdir(Correct_path) if os.path.isfile(os.path.join(Correct_path, name))])
count_incorrect=len([name for name in os.listdir(Incorrect_path) if os.path.isfile(os.path.join(Incorrect_path, name))])

f1 = open('generalsamples.data', 'ab')
f2 = open('generalresponses.data', 'ab')

# training the bot with correct/genuine signatures
for i in range(1,count_correct+1):
    img = cv2.imread('Correct\sig ('+str(i)+').jpg')
    img_final = cv2.imread('Correct\sig ('+str(i)+').jpg')
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 110, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 110, 255, cv2.THRESH_BINARY_INV)  # for white text , cv.THRESH_BINARY
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    dilated = cv2.dilate(new_img, kernel, iterations=9)
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # get contours
    index = 0
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 100 and h < 100:
            continue

        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cropped = img_final[y:y + h, x: x + w]
        s = 'final_' + 'crop_' + str(index) + '.jpg'
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(cropped, 130, 255, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(cropped, cropped, mask=mask)
        ret, new_img = cv2.threshold(image_final, 130, 255, cv2.THRESH_BINARY)

        index = index + 1

    # setting up samples and corresponding responses
    responses = []
    samples = np.empty((0, 625))
    roismall = cv2.resize(cropped, (25, 25))

    responses.append(1)
    sample = roismall.reshape((1, 625))
    samples = np.append(samples, sample, 0)

    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))

    np.savetxt(f1, samples)
    np.savetxt(f2, responses)

# training the bot with incorrect signatures
for i in range(1, count_incorrect):
    img = cv2.imread('Incorrect\sig (' + str(i) + ').jpg')
    img_final = cv2.imread('Incorrect\sig (' + str(i) + ').jpg')
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 110, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 110, 255, cv2.THRESH_BINARY_INV)  # for white text , cv.THRESH_BINARY
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(new_img, kernel, iterations=9)
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # get contours
    index = 0
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 100 and h < 100:
            continue

        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cropped = img_final[y:y + h, x: x + w]
        s = 'final_' + 'crop_' + str(index) + '.jpg'
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(cropped, 130, 255, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(cropped, cropped, mask=mask)
        ret, new_img = cv2.threshold(image_final, 130, 255, cv2.THRESH_BINARY)

        index = index + 1

    # setting up samples and corresponding responses
    responses = []
    samples = np.empty((0, 625))
    roismall = cv2.resize(cropped, (25, 25))

    responses.append(0)
    sample = roismall.reshape((1, 625))
    samples = np.append(samples, sample, 0)

    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))

    np.savetxt(f1, samples)
    np.savetxt(f2, responses)

f1.close()
f2.close()