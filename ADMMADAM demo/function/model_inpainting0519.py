import torch
from torch import nn

class generator(nn.Module):

    #generator model
    def __init__(self):
        super(generator,self).__init__()
        
        self.t1=nn.Sequential(
            nn.Conv2d(in_channels=172,out_channels=172,kernel_size=(7,7),stride=1,padding=2,dilation=1),
            nn.BatchNorm2d(172),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.t2=nn.Sequential(
            nn.Conv2d(in_channels=172,out_channels=256,kernel_size=(4,4),stride=2,padding=2,dilation=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.t3=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(4,4),stride=2,padding=1,dilation=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
        )
        
        
        # residual blocks
        self.t4=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512)
        ) 
        self.t5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512)
        ) 
        self.t6=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512)
        )         
        self.t7=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=2,dilation=2),
            nn.BatchNorm2d(512)

        ) 

        self.t8=nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(4,4),stride=2,padding=1,dilation=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
            )
        self.t9=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=(4,4),stride=2,padding=1,dilation=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
            )
        self.t10=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=172,kernel_size=(7,7),stride=1,padding=3,dilation=1),
            nn.LeakyReLU(0.2,inplace=True)
            )
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self,x):
    	x1=self.t1(x)
    	x2=self.t2(x1)
    	x3=self.t3(x2)      
        
    	identity1=x3
    	x4=self.t4(x3)
    	x4=self.lrelu(x4+identity1)
        
    	identity2=x4
    	x5=self.t5(x4)
    	x5=self.lrelu(x5+identity2)
        
    	identity3=x5
    	x6=self.t6(x5)
    	x6=self.lrelu(x6+identity3)
    	
    	identity4=x6
    	x7=self.t7(x6)
    	x7=self.lrelu(x7+identity4)
    	
    	x8=self.t8(x7)
    	x9=self.t9(x8)
    	x10=self.t10(x9)

    	return x10 #output of generator