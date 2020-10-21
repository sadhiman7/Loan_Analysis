import cv2
import numpy as np
import imutils

def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

image=cv2.imread("picture1.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(blurred,30,50)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)

cv2.drawContours(image, cnts, 0, (0, 0, 0), 2)

p = cv2.arcLength(cnts[0], True)
approx = cv2.approxPolyDP(cnts[0], 0.02 * p, True)

approx=mapp(approx)

pts=np.float32([[0,0],[800,0],[800,800],[0,800]])

op=cv2.getPerspectiveTransform(approx,pts)
dst=cv2.warpPerspective(image,op,(800,800))


cv2.imshow("Scanned",dst)
cv2.imshow("Image", image)
cv2.imshow("Gray", gray)
cv2.imshow("Blurred", blurred)
cv2.imshow("Edged", edged)

cv2.imwrite("output/Image.jpg", image)
cv2.imwrite("output/Gray.jpg", gray)
cv2.imwrite("output/Blurred.jpg", blurred)
cv2.imwrite("output/Edged.jpg", edged)
cv2.imwrite("output/Scanned.jpg", dst)

cv2.waitKey(0)


