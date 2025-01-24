import torch
import torch.nn as nn

class UNetConvBlock(nn.Module):
    def __init__(self):
        super(UNetConvBlock, self).__init__()
        #3x3 block 
        #3x3 
        self.block1 = nn.Sequential(nn.ConstantPad2d(1, -1),
                                    nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (3,3)),
                                    nn.BatchNorm2d(num_features = 64),
                                    nn.ReLU(inplace = True),                            
                                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
                                    nn.BatchNorm2d(num_features = 64))
        
        #3x3 
        #3x5
        self.block2 = nn.Sequential(nn.ConstantPad2d(1, -1),
                                    nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (3,3)),
                                    nn.BatchNorm2d(num_features = 64),
                                    nn.ReLU(inplace = True), 
                                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (5,5), padding = (2,2)),
                                    nn.BatchNorm2d(num_features = 64))
        
        #5x5 
        #3x3
        self.block3 = nn.Sequential(nn.ConstantPad2d(2, -1),
                                    nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (5,5)),
                                    nn.BatchNorm2d(num_features = 64),
                                    nn.ReLU(inplace = True), 
                                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
                                    nn.BatchNorm2d(num_features = 64))
        
        #5x5
        #5x5
        self.block4 = nn.Sequential(nn.ConstantPad2d(2, -1),
                                    nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (5,5)),
                                    nn.BatchNorm2d(num_features = 64),
                                    nn.ReLU(inplace = True), 
                                    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (5,5), padding = (2,2)),
                                    nn.BatchNorm2d(num_features = 64))
                                    
        self.convolutionalization1 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)),
                                                   nn.BatchNorm2d(num_features = 256),
                                                   nn.ReLU(inplace = True),
                                                   nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)),
                                                   nn.BatchNorm2d(num_features = 256),
                                                   nn.ReLU(inplace = True),
                                                   nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)))
                                                        
        self.convolutionalization2 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)),
                                                   nn.BatchNorm2d(num_features = 256),
                                                   nn.ReLU(inplace = True),
                                                   nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)),
                                                   nn.BatchNorm2d(num_features = 256),
                                                   nn.ReLU(inplace = True),
                                                   nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)))
        
    def forward(self, input):
        block1 = self.block1(input)
        block2 = self.block2(input)
        block3 = self.block3(input)
        block4 = self.block4(input)
        
        output = torch.zeros(block4.shape[0], block4.shape[1]*4, block4.shape[2], block4.shape[3], device = block4.device)
        output[:, 0::4, :, :] = block1
        output[:, 1::4, :, :] = block2
        output[:, 2::4, :, :] = block3
        output[:, 3::4, :, :] = block4
        
        output = self.convolutionalization1(output) + self.convolutionalization2(input)
        return output

#Change to a better upscaler after getting this one to work
class UNetColorizer(nn.Module):

    def __init__(self):
        super(UNetColorizer, self).__init__()
        
        #Encoder
        self.encode1 = nn.Sequential(nn.ConstantPad2d(1, -1),
                                     nn.Conv2d(in_channels = 1, out_channels = 256, kernel_size = (3,3)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 64))

                                             
        self.encode2 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 64))
                                                                          
        self.encode3 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 64))
                                     
        self.encode4 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock())
                                             
        #BottleNeck
        self.bottleNeck1 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock())
                                     
        self.bottleNeck2 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock())
        
        #Decoder
        self.decode1 = nn.Sequential(nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock())
                                     
        self.decode2 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock())                             
                                     
        self.decode3 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock(),
                                     nn.ReLU(inplace = True), 
                                     UNetConvBlock())
                                     
        self.decode4 = nn.Sequential(nn.ConstantPad2d(1, -1),
                                     nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3,3)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True), 
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ReLU(inplace = True),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.BatchNorm2d(num_features = 256))
                                                                          
        self.convolutionalizationLayers = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)),
                                                        nn.ReLU(inplace = True), 
                                                        nn.BatchNorm2d(num_features = 256),
                                                        nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)),
                                                        nn.ReLU(inplace = True), 
                                                        nn.BatchNorm2d(num_features = 256),
                                                        nn.Conv2d(in_channels = 256, out_channels = 3, kernel_size = (1,1)))
                                                        
    def forward(self, input):
        #Encoder
        encode1 = self.encode1(input)
        
        #256 x 256
        inputEncode2 = torch.zeros((encode1.shape[0], encode1.shape[1]*4, int(encode1.shape[2]/2), int(encode1.shape[3]/2)), device = encode1.device)
        inputEncode2[:, 0::4, :, :] = encode1[:, :, 0::2, 0::2]
        inputEncode2[:, 1::4, :, :] = encode1[:, :, 0::2, 1::2]
        inputEncode2[:, 2::4, :, :] = encode1[:, :, 1::2, 0::2]
        inputEncode2[:, 3::4, :, :] = encode1[:, :, 1::2, 1::2]
        
        encode2 = self.encode2(inputEncode2)
        
        #64 x 64
        inputEncode3 = torch.zeros((encode2.shape[0], encode2.shape[1]*4, int(encode2.shape[2]/2), int(encode2.shape[3]/2)), device = encode2.device)
        inputEncode3[:, 0::4, :, :] = encode2[:, :, 0::2, 0::2]
        inputEncode3[:, 1::4, :, :] = encode2[:, :, 0::2, 1::2]
        inputEncode3[:, 2::4, :, :] = encode2[:, :, 1::2, 0::2]
        inputEncode3[:, 3::4, :, :] = encode2[:, :, 1::2, 1::2]
               
        encode3 = self.encode3(inputEncode3)
        
        #32 x 32
        inputEncode4 = torch.zeros((encode3.shape[0], encode3.shape[1]*4, int(encode3.shape[2]/2), int(encode3.shape[3]/2)), device = encode3.device)
        inputEncode4[:, 0::4, :, :] = encode3[:, :, 0::2, 0::2]
        inputEncode4[:, 1::4, :, :] = encode3[:, :, 0::2, 1::2]
        inputEncode4[:, 2::4, :, :] = encode3[:, :, 1::2, 0::2]
        inputEncode4[:, 3::4, :, :] = encode3[:, :, 1::2, 1::2]
               
        encode4 = self.encode4(inputEncode4)
        
        #Bottle Neck
        bottleNeck1 = self.bottleNeck1(encode4)
        bottleNeck2 = self.bottleNeck2(bottleNeck1)

        #Decoder        
        #32 x 32
        decode1 = self.decode1(torch.concatenate((bottleNeck2, encode4), 1))

        #64 x 64
        inputDecode2 = torch.zeros((decode1.shape[0], int(decode1.shape[1]/4), int(decode1.shape[2]*2), int(decode1.shape[3]*2)), device = decode1.device)
        inputDecode2[:, :, 0::2, 0::2] = decode1[:, 0::4, :, :]
        inputDecode2[:, :, 0::2, 1::2] = decode1[:, 1::4, :, :]
        inputDecode2[:, :, 1::2, 0::2] = decode1[:, 2::4, :, :]
        inputDecode2[:, :, 1::2, 1::2] = decode1[:, 3::4, :, :]
        
        decode2 = self.decode2(torch.concatenate((inputDecode2, encode3), 1))
        
        #128 x 128
        inputDecode3 = torch.zeros((decode2.shape[0], int(decode2.shape[1]/4), int(decode2.shape[2]*2), int(decode2.shape[3]*2)), device = decode2.device)
        inputDecode3[:, :, 0::2, 0::2] = decode2[:, 0::4, :, :]
        inputDecode3[:, :, 0::2, 1::2] = decode2[:, 1::4, :, :]
        inputDecode3[:, :, 1::2, 0::2] = decode2[:, 2::4, :, :]
        inputDecode3[:, :, 1::2, 1::2] = decode2[:, 3::4, :, :]
        
        decode3 = self.decode3(torch.concatenate((inputDecode3, encode2), 1))
        
        #256 x 256
        inputDecode4 = torch.zeros((decode3.shape[0], int(decode3.shape[1]/4), int(decode3.shape[2]*2), int(decode3.shape[3]*2)), device = decode3.device)
        inputDecode4[:, :, 0::2, 0::2] = decode3[:, 0::4, :, :]
        inputDecode4[:, :, 0::2, 1::2] = decode3[:, 1::4, :, :]
        inputDecode4[:, :, 1::2, 0::2] = decode3[:, 2::4, :, :]
        inputDecode4[:, :, 1::2, 1::2] = decode3[:, 3::4, :, :]
        
        decode4 = self.decode3(torch.concatenate((inputDecode4, encode1), 1))
        output = self.convolutionalizationLayers(decode4)
        
        return output
