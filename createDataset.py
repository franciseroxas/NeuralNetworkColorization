import os
import cv2 
import numpy as np

if(os.path.exists("dataset") == False):
    os.mkdir("dataset")

if(os.path.exists("dataset_bw") == False):
    os.mkdir("dataset_bw")

for i in range(1, 1005):
    for j in range(6, 15):
        chapterNumber = str(i)
        chapterNumber = chapterNumber.zfill(4)

        pageNumber = str(j)
        pageNumber = pageNumber.zfill(3)

        print(chapterNumber + "/" + str(i)+"-"+str(j)+".png")
        
        try:
            #save the original image so that the image can be cropped with PIL
            img = cv2.imread(chapterNumber + "/" + str(i)+"-"+str(j)+".png")
            img = cv2.bilateralFilter(img, 9, 75, 75)
            img = cv2.resize(img, dsize = (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(e)
            continue
        
        for k in range(10):
            #Select a 256 x 256 part of the image that is within the original image
            randHeight = np.random.randint(0, img.shape[0] - 256)
            randWidth = np.random.randint(0, img.shape[1] - 256)
            
            #Crop a random img of size 256 x 256 from the image 
            croppedImg = img[randHeight:randHeight+256, randWidth:randWidth+256, :]
            #Save the color image
            cv2.imwrite("dataset/"+chapterNumber+"-"+pageNumber+"-"+str(k)+"_clr.png", croppedImg)
            #Save the bw image
            cv2.imwrite("dataset_bw/"+chapterNumber+"-"+pageNumber+"-"+str(k)+"_bw.png", 
                cv2.cvtColor(croppedImg, cv2.COLOR_BGR2GRAY))