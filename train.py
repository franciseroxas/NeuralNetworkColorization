from dataset import BW2ClrImageDataset
from network import UNetColorizer

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import cv2
import os
import numpy as np
import argparse
import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(modelDict = None):
    if(os.path.exists('output') == False):
        os.mkdir('output')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    if(os.path.exists("./datasets.pt")):
        datasetsDict = torch.load("./datasets.pt")
        trainDataset = datasetsDict['trainDataset']
        valDataset = datasetsDict['valDataset']
        del datasetsDict
    else:
        myDataset = BW2ClrImageDataset(bw_img_dir = 'dataset_equal_bw/', clr_img_dir = 'dataset_equal/')
        trainDataset, valDataset = random_split(myDataset, [0.8, 0.2])
        torch.save({'trainDataset': trainDataset,
                    'valDataset': valDataset}, "./datasets.pt")

    batch_size = 1
    trainDataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=8)
 
    model = UNetColorizer()
    if torch.cuda.is_available():
        model.cuda()
        
    learningRate = 0.0001 / batch_size
    num_epochs = 100

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08)
    lossOverTime = []
    startEpoch = 0

    if(modelDict is not None):
        model.load_state_dict(modelDict['model_state_dict'])
        loss = modelDict['loss']
        startEpoch = modelDict['epoch'] + 1
        optimizer.load_state_dict(modelDict['optimizer_state_dict'])
    
    for epoch in range(startEpoch, num_epochs):
        currentLoss = 0
        startOfBatch = time.time()
        for batch_ndx, data in enumerate(trainDataloader):
            bwImgsTensor, clrImgsTensor = data
            bwImgsTensor, clrImgsTensor = bwImgsTensor.to(device), clrImgsTensor.to(device)
            
            colorOutput = model(bwImgsTensor)
            loss = criterion(clrImgsTensor, colorOutput) #loss is symmetric 
            batchLoss = loss.cpu().detach().numpy() / batch_size
            currentLoss = currentLoss + batchLoss #normalize by batch size for comparison
            
            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("Epoch", epoch, " | Batch", batch_ndx, " | Total", len(trainDataloader), " | LR:", np.round(learningRate, 7), " | Batch Loss:", np.round(batchLoss, 5), " | Time:", np.round(time.time() - startOfBatch, 5))

            #For visualization of an output
            cv2.imwrite("output/train" + str(batch_ndx % 50) + "_predicted.png", torch.transpose(colorOutput[0][:].unsqueeze(3), 0, 3).squeeze().cpu().detach().numpy() * 128 + 128)
            cv2.imwrite("output/train" + str(batch_ndx % 50) + "_color.png", torch.transpose(clrImgsTensor[0][:].unsqueeze(3), 0, 3).squeeze().cpu().detach().numpy() * 128 + 128)
            cv2.imwrite("output/train" + str(batch_ndx % 50) + "_bw.png", bwImgsTensor[0][0].cpu().detach().numpy() * 128 + 128)
            
            try:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'batch_ndx': batch_ndx,
                        'learningRate': learningRate,
                        'batch_size': batch_size,
                        'lossList': lossOverTime,
                        }, "./upscalingModel_inProgress.pt")
            except Exception as e:
                print(e)
                
            startOfBatch = time.time()

        time.sleep(1)
        print("----------------------------")
        print("Epoch", epoch, "Loss:", currentLoss)
        print("----------------------------")
            
        lossOverTime.append(currentLoss)
        plt.figure()
        plt.plot(np.array(lossOverTime))
        plt.title("Loss of neural network over time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("Loss of neural network.png")
        plt.close("all")

        try:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'batch_ndx': batch_ndx,
                    'learningRate': learningRate,
                    'batch_size': batch_size,
                    'lossList': lossOverTime,
                    }, "./upscalingModel_recentEpoch.pt")
        except Exception as e:
            print(e)

def main(args = None):   
    parser = argparse.ArgumentParser(description='Neural Network Colorization Project in Pytorch')
    args = parser.parse_args()
    
    modelDict = torch.load('upscalingModel_recentEpoch - 7 epochs.pt')
    train(modelDict = modelDict)

if __name__ == "__main__":
    main()
