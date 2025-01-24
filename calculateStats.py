import numpy as np
import cv2
import os
import torch
from train import train
#Want to count how many red, blue, green and permutations of these colors are in the dataset. Want to keep an equal number of each of these numbers may need to impute or oversample from pictures that contain these colors

#255, 0, 0 Blue
#0, 255, 0 Green 
#0, 0, 255 Red
#255, 255, 0 Blue Green
#255, 0, 255 Blue Red
#0, 255, 255 Green Red
def countColors(pictureDir):
    colorDict = {}
    colorDict['Blue'] = []
    colorDict['Green'] = []
    colorDict['Red'] = []
    colorDict['BlueGreen'] = []
    colorDict['BlueRed'] = []
    colorDict['GreenRed'] = []

    fileList = os.listdir(pictureDir)
    for i in range(len(fileList)):
        img = cv2.imread(pictureDir + fileList[i])
        width = img.shape[1]
        height = img.shape[0]

        blueCount = np.sum(img[:, :, 0] > 128)
        greenCount = np.sum(img[:, :, 1] > 128)
        redCount = np.sum(img[:, :, 2] > 128)

        if(blueCount > (width * height / 2) and greenCount > (width * height / 2) and redCount < min(greenCount, blueCount)):
            colorDict.get('BlueGreen').append(fileList[i])
        elif(blueCount > (width * height / 2) and redCount > (width * height / 2) and greenCount < min(blueCount, redCount)):
            colorDict.get('BlueRed').append(fileList[i])
        elif(greenCount > (width * height / 2) and redCount > (width * height / 2) and blueCount < min(greenCount, redCount)):
            colorDict.get('GreenRed').append(fileList[i])
        elif(blueCount >= greenCount and blueCount >= redCount):
            colorDict.get('Blue').append(fileList[i])
        elif(greenCount >= blueCount and greenCount >= redCount):
            colorDict.get('Green').append(fileList[i])
        else:
            colorDict.get('Red').append(fileList[i])
        
        print(len(colorDict.get('Blue')), 
              len(colorDict.get('Green')), 
              len(colorDict.get('Red')), 
              len(colorDict.get('BlueGreen')),
              len(colorDict.get('BlueRed')),
              len(colorDict.get('GreenRed')),
              len(fileList), fileList[i])
        torch.save({'colorDict': colorDict}, "colorDict.pt")
    return colorDict

#colorDict = countColors('dataset/')
#torch.save({'colorDict': colorDict}, "colorDict.pt")
colorDict = torch.load('colorDict.pt')
colorDict = colorDict['colorDict']

print(len(colorDict.get('Blue')),
              len(colorDict.get('Green')),
              len(colorDict.get('Red')),
              len(colorDict.get('BlueGreen')),
              len(colorDict.get('BlueRed')),
              len(colorDict.get('GreenRed')))

blueList = colorDict.get('Blue')
greenList = colorDict.get('Green')
redList = colorDict.get('Red')
blueGreenList = colorDict.get('BlueGreen')
blueRedList = colorDict.get('BlueRed')
greenRedList = colorDict.get('GreenRed')


equalDataSetList = []

equalDataSetList = equalDataSetList + (blueGreenList[0:1000])
equalDataSetList = equalDataSetList + (blueRedList[0:1000])
equalDataSetList = equalDataSetList + (greenRedList[0:1000])
print(len(equalDataSetList))

for i in range(len(equalDataSetList)):
    print(equalDataSetList[i], i)
    img = cv2.imread('dataset/' + equalDataSetList[i])
    cv2.imwrite('dataset_equal/' + str(i) + "_clr.png", img)
    cv2.imwrite('dataset_equal_bw/'+ str(i) + "_bw.png", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

train()
