import cv2
import numpy as np
import os

path = 'references'
inPath = 'input'
outPath = 'output'

orb = cv2.ORB_create(nfeatures=1000)

images = []
classNames = []

myList = os.listdir(path)

print('Total Classes', len(myList))

for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findDes(images):
    desList = []
    for image in images:
        kp, des = orb.detectAndCompute(image, None)
        desList.append(des)
    return desList

def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    for des in desList:
        matches = bf.knnMatch(des, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.90*n.distance:
                good.append([m])
        matchList.append(len(good))
    if max(matchList)>165:
        finalVal = matchList.index(max(matchList))
    print(matchList)
    return(finalVal)

desList = findDes(images)

myList = os.listdir(inPath)

for img in myList:
    checkImg = cv2.imread(f'{inPath}/{img}', 0)
    id = findID(checkImg, desList)
    if id!=-1:
        print(img, "is", classNames[id])
    else:
        print(img, "is document")
