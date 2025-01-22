import torch
import torch.nn as nn

#Change to a better upscaler after getting this one to work
class UNetColorizer(nn.Module):

    def __init__(self):
        super(UNetColorizer, self).__init__()
        #Self feature transform for edges
        self.edgeTransform1 = nn.Sequential(nn.ConstantPad2d(1, -1),
                                           nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3,3)),
                                           nn.ReLU(), 
                                           nn.BatchNorm2d(num_features = 64),
                                           nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
                                           nn.ReLU(), 
                                           nn.BatchNorm2d(num_features = 64))
        
        self.edgeTransform2 = nn.Sequential(nn.ConstantPad2d(2, -1),
                                           nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (3,3), dilation = (2,2)),
                                           nn.ReLU(), 
                                           nn.BatchNorm2d(num_features = 64),
                                           nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
                                           nn.ReLU(), 
                                           nn.BatchNorm2d(num_features = 64))
        
        #Encoder
        self.encode1 = nn.Sequential(nn.Conv2d(in_channels = 129, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
                                             
        self.encode2 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 64))
                                                                          
        self.encode3 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256))
                                             
        #BottleNeck
        self.bottleNeck1 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256))
                                     
        self.bottleNeck2 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256))
        
        #Decoder
        self.decode1 = nn.Sequential(nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256))
                                     
        self.decode2 = nn.Sequential(nn.ConvTranspose2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256))
                                     
        self.decode3 = nn.Sequential(nn.ConvTranspose2d(in_channels = 128, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256),
                                     nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3,3), padding = (1,1)), 
                                     nn.ReLU(), 
                                     nn.BatchNorm2d(num_features = 256))
                                                                          
        self.convolutionalizationLayers = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)),
                                                        nn.ReLU(), 
                                                        nn.BatchNorm2d(num_features = 256),
                                                        nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,1)),
                                                        nn.ReLU(), 
                                                        nn.BatchNorm2d(num_features = 256),
                                                        nn.Conv2d(in_channels = 256, out_channels = 3, kernel_size = (1,1)))
                                                        
    def forward(self, input):
        #Make sure that the edge effects are dealt with
        encoderInput1 = self.edgeTransform1(input)
        encoderInput2 = self.edgeTransform2(input)
        
        #Encoder
        encode1 = self.encode1(torch.concatenate((encoderInput1, encoderInput2, input), 1))
        
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
        
        #Bottle Neck
        bottleNeck1 = self.bottleNeck1(encode3)
        bottleNeck2 = self.bottleNeck2(bottleNeck1)

        #Decoder        
        #64 x 64
        decode1 = self.decode1(torch.concatenate((bottleNeck2, encode3), 1))

        #128 x 128
        inputDecode2 = torch.zeros((decode1.shape[0], int(decode1.shape[1]/4), int(decode1.shape[2]*2), int(decode1.shape[3]*2)), device = decode1.device)
        inputDecode2[:, :, 0::2, 0::2] = decode1[:, 0::4, :, :]
        inputDecode2[:, :, 0::2, 1::2] = decode1[:, 1::4, :, :]
        inputDecode2[:, :, 1::2, 0::2] = decode1[:, 2::4, :, :]
        inputDecode2[:, :, 1::2, 1::2] = decode1[:, 3::4, :, :]
        
        decode2 = self.decode2(torch.concatenate((inputDecode2, encode2), 1))
        
        #256 x 256
        inputDecode3 = torch.zeros((decode2.shape[0], int(decode2.shape[1]/4), int(decode2.shape[2]*2), int(decode2.shape[3]*2)), device = decode2.device)
        inputDecode3[:, :, 0::2, 0::2] = decode2[:, 0::4, :, :]
        inputDecode3[:, :, 0::2, 1::2] = decode2[:, 1::4, :, :]
        inputDecode3[:, :, 1::2, 0::2] = decode2[:, 2::4, :, :]
        inputDecode3[:, :, 1::2, 1::2] = decode2[:, 3::4, :, :]
        
        decode3 = self.decode3(torch.concatenate((inputDecode3, encode1), 1))
        output = self.convolutionalizationLayers(decode3)
        
        return output
