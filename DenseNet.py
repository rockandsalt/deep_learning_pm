
## Credit to andreasVeit
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes).cuda()
        self.relu = nn.ReLU(inplace=True).cuda()
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False).cuda()
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers).cuda()
    def forward(self, x):
        return self.layer(x)

class DiscriminatorBlock(nn.Module):
    def __init__(self,in_channel,out_channel,k,s,p):
        super(DiscriminatorBlock,self).__init__()
        self.conv1 = nn.Conv3d(in_channel,out_channel,kernel_size = k, stride = s, padding = p).cuda()
        self.bn1 = nn.BatchNorm3d(out_channel).cuda()
        self.lrelu = nn.LeakyReLU().cuda()

    def forward(self,x):
        out = self.bn1(self.conv1(x))
        return self.lrelu(out)

class Generator(nn.Module):
    def __init__(self, u, num_block, growth_rate=12, dropRate=0.0):
        super(Generator, self).__init__()
        d_in_planes = 2*growth_rate
        c_in_planes = d_in_planes

        block = BasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv3d(1, d_in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False).cuda()
        
        self.list_dense_block = []
        self.list_compressor_block = []

        for i in range(num_block):
            if(i != 0):
                d_in_planes = 1

            d_block = DenseBlock(u, d_in_planes, growth_rate, block, dropRate)
            self.list_dense_block.append(d_block)

            if(i == num_block - 1):
                c_in_planes = int(c_in_planes+u*growth_rate)
            else:
                c_in_planes = int(c_in_planes+u*growth_rate+d_in_planes)
                compressor = nn.Conv3d(c_in_planes,1,kernel_size =1 , stride= 1, padding= 0, bias= False).cuda()
                self.list_compressor_block.append(compressor)
        
        self.recon = nn.Conv3d(c_in_planes+1, 1, kernel_size=1, stride=1, padding=0, bias=False).cuda()

    def forward(self, x):
        out = self.conv1(x)

        compressor_in = out
        for idx, cpr in enumerate(self.list_compressor_block):
            d_block = self.list_dense_block[idx]
            d_out = d_block(out)

            compressor_in = torch.cat([compressor_in, d_out], 1)
            out = cpr(compressor_in)
            
        last_block = self.list_dense_block[-1]
        d_out = last_block(out)

        return self.recon(torch.cat([compressor_in,d_out], 1))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv3d(1,64,kernel_size = 3, stride=1).cuda()
        self.lrelu1 = nn.LeakyReLU().cuda()

        self.block1 = DiscriminatorBlock(64,64,3,2,1)
        self.block2 = DiscriminatorBlock(64,128,3,1,1)
        self.block3 = DiscriminatorBlock(128,128,3,2,1)
        self.block4 = DiscriminatorBlock(128,256,3,1,1)
        self.block5 = DiscriminatorBlock(256,256,3,2,1)
        self.block6 = DiscriminatorBlock(256,512,3,1,1)
        self.block7 = DiscriminatorBlock(512,512,3,2,1)
        self.ada = nn.AdaptiveAvgPool3d(1).cuda()
        self.Dense1 = nn.Conv3d(512,1024, kernel_size  =1 ).cuda()
        self.lrelu2 = nn.LeakyReLU().cuda()
        self.Dense2 = nn.Conv3d(1024,1, kernel_size = 1).cuda()

    def forward(self, x):
        out = self.lrelu1(self.conv1(x))

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.ada(out)
        out = self.Dense1(out)
        out = self.lrelu2(out)
        out = self.Dense2(out)

        return F.sigmoid(out)
    
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.d_loss = nn.BCELoss()

    def forward(self, out_labels, target_labels, out_images, target_images):
        adversarial_loss = self.d_loss(out_labels,target_labels)
        image_loss = self.mse_loss(out_images, target_images)
        return image_loss - 0.001 * adversarial_loss