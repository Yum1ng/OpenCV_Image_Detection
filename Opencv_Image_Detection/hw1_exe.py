import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import colorsys

folder = "testset"



def sigmoid(t):
    return 1/(1+np.exp(-1*t))

w1 = 2.83041518
w2 = 7.28183566
w3 = -0.49499156
w4 = -6.02723031
W = np.array([w1, w2, w3, w4])
red = np.array([10,10,201])
W_distance = np.array([[-13.55766845], [-12.00187583], [8.21819279]])


def calculate_distance(w, h):
    return w*W_distance[0] + h*W_distance[1] + W_distance[2]


def calculate_barrelness(image, curr_cnt):
    #cv.imread(image)
    print("in calculation ")

    row = image.shape[0]
    col = image.shape[1]
    x, y, w, h = cv2.boundingRect(curr_cnt)
    score = 0
    if w < row/30 or h < col/20:
        score += 300
    if w/h > 2 or h/w > 2:
        score += 100
    box_color = image[x:x+w, y:y+h, :]
    box_color_mean = np.mean(np.mean(box_color, axis=0), axis=0)
    score += 2 * np.power(np.power((box_color_mean[0]-red[0]),2) + np.power((box_color_mean[1]-red[1]), 2) + np.power((box_color_mean[2]-red[2]),2),1/2)
    return score


def myAlgorithm(img):
    print("shape: ", img.shape)
    image_gray = np.array(np.zeros((img.shape[0], img.shape[1])))
    #image_gray = np.zeros((img.shape))
    for i in range(img.shape[0]):
        if i % 500 == 0:
            print("i: ", i)
        for j in range(img.shape[1]):
            X = np.array(img[i, j, :]/255)
            X = colorsys.rgb_to_hsv(X[0], X[1], X[2])
            xb = np.array(1)
            X = np.hstack((X, xb))
            g = np.dot(X, W.transpose())
            y_result = sigmoid(g)

            if y_result > 0.5:
                image_gray[i, j] = 1
    cv2.imshow('gray', image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    thresh = np.array(image_gray*255, dtype=np.uint8)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.asarray(contours)
    cnt_new = list()

    ratio = 40/57
    for i in range(len(cnt)):
        x, y, w, h = cv2.boundingRect(cnt[i])
        aspect_ratio = float(w)/h
        rect = cv2.minAreaRect(cnt[i]) # Draw rectangle
        if rect[1][1] > 20 and rect[1][0] > 10 and ((rect[1][1]/rect[1][0] < 1.2 * ratio and rect[1][1]/rect[1][0] > 0.8 * ratio) or rect[1][0]/rect[1][1] < 2 * ratio and rect[1][0]/rect[1][1] > 0.8 * ratio):
            print("i is: ", i)
            cnt_new.append(np.asarray(cnt[i]))
    min_score = 50000
    min_index = 0

    for k in range(len(cnt_new)):
        score = calculate_barrelness(img, cnt_new[k])
        print("score is :", score)
        if score < min_score:
            min_score = score
            min_index = k
    print("result min score (most red) index is : ", min_index)
    x, y, w, h = cv2.boundingRect(cnt_new[min_index])
    rect = cv2.minAreaRect(cnt_new[min_index])  # Draw rectangle
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    distance = calculate_distance(w/img.shape[0],h/img.shape[1])
    print("BottomLeftX = ", box[0][0], "BottomLeftY = ", box[0][1], "TopRightX = ", box[2][0], "TopRightY = ", box[2][1])
    print("Distance : ", distance)
    cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

    cv2.imshow("Image", img)

pic = 0
for filename in os.listdir(folder):
    # read one test image
    img = cv2.imread(os.path.join(folder,filename))
    # Your computations here!
    print("image NO.", pic, " :")
    myAlgorithm(img)
    pic += 1
    #blX, blY, trX, trY, d = myAlgorithm(img)
    # Display results:
    # (1) Segmented image
    # (2) Barrel bounding box
    # (3) Distance of barrel
    # You may also want to plot and display other diagnostic information
    cv2.waitKey(0)
    cv2.destroyAllWindows()

