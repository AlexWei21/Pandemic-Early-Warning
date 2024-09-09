import torch.nn as nn
import torch
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, bn = True, ln = False, input_shape = None):
        super(Bottleneck, self).__init__()
        
        self.bn = bn

        self.conv1 = nn.Conv1d(in_channels, 
                               out_channels, 
                               kernel_size=1, 
                               stride=1, 
                               padding=0)
        
        if self.bn:
            self.batch_norm1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, 
                               out_channels, 
                               kernel_size=3, 
                               stride=stride, 
                               padding=1)
        
        if self.bn:
            self.batch_norm2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(out_channels, 
                               out_channels*self.expansion, 
                               kernel_size=1, 
                               stride=1, 
                               padding=0)
        if self.bn:
            self.batch_norm3 = nn.BatchNorm1d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        if self.bn:
            x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        if self.bn:
            x = self.batch_norm2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        if self.bn:
            x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, bn=True, ln=True, input_shape = None):
        super(Block, self).__init__()
        self.bn=bn
        self.input_shape=input_shape
        self.ln=ln

        self.conv1 = nn.Conv1d(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               padding=1, 
                               stride=stride, 
                               bias=False)
        
        self.input_shape = int((self.input_shape - 3 + 2*1)/stride + 1)

        if self.bn:
            self.batch_norm1 = nn.BatchNorm1d(out_channels)
        
        if self.ln:
            self.layer_norm1 = nn.LayerNorm([out_channels,self.input_shape])

        self.conv2 = nn.Conv1d(out_channels, 
                               out_channels, 
                               kernel_size=3, 
                               padding=1, 
                               stride=1, 
                               bias=False)
        
        self.input_shape = int((self.input_shape - 3 + 2*1 / 1) + 1)

        if self.bn:
            self.batch_norm2 = nn.BatchNorm1d(out_channels)
        if self.ln:
            self.layer_norm2 = nn.LayerNorm([out_channels, self.input_shape])

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.GELU()

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        if self.bn:
            x = self.batch_norm1(x)
        if self.ln:
            x = self.layer_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if self.bn:
            x = self.batch_norm2(x)
        if self.ln:
            x = self.layer_norm2(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, output_dim, input_shape=46, num_channels=3, bn=True, ln=True):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.bn = bn
        self.ln = ln
        
        self.input_shape = input_shape

        self.conv1 = nn.Conv1d(num_channels, 
                               64, 
                               kernel_size=7, 
                               stride=2, 
                               padding=3, 
                               bias=False)
        
        self.input_shape = int((self.input_shape - 7 + 2*3)/2 + 1)
        
        if self.bn:
            self.batch_norm1 = nn.BatchNorm1d(64)
        if self.ln:
            self.layer_norm1 = nn.LayerNorm([64,self.input_shape])

        self.relu = nn.LeakyReLU()

        self.max_pool = nn.MaxPool1d(kernel_size = 3, 
                                     stride=2, 
                                     padding=1)
        
        self.input_shape = int((self.input_shape - 3 + 2*1)/2 + 1)

        self.layer1, self.input_shape = self._make_layer(ResBlock, layer_list[0], planes=64, bn=bn, ln=self.ln, input_shape=self.input_shape)

        self.layer2, self.input_shape = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2, bn=bn,ln=self.ln, input_shape=self.input_shape)
        self.layer3, self.input_shape = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2, bn=bn,ln=self.ln, input_shape=self.input_shape)
        self.layer4, self.input_shape = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2, bn=bn,ln=self.ln, input_shape=self.input_shape)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.meta_data_layer1 = nn.Linear(40,64)
        self.meta_data_layer2 = nn.Linear(64,128)

        self.fc1 = nn.Linear(512*ResBlock.expansion + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64,output_dim)
        
    def forward(self, x, meta_data):

        x = self.conv1(x)
        if self.bn:
            x = self.batch_norm1(x)
        if self.ln:
            x = self.layer_norm1(x)
        x = self.relu(x)

        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        meta_data = self.meta_data_layer1(meta_data)
        meta_data = self.meta_data_layer2(meta_data)

        x = torch.cat([x,meta_data], dim = 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x
    
    def _make_layer(self, ResBlock, blocks, planes,input_shape, stride=1, bn=True, ln=True):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            if bn:
                ii_downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, 
                            planes*ResBlock.expansion, 
                            kernel_size=1, 
                            stride=stride),
                    nn.BatchNorm1d(planes*ResBlock.expansion)
                )
            elif ln:
                ii_downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, 
                            planes*ResBlock.expansion, 
                            kernel_size=1, 
                            stride=stride),
                    nn.LayerNorm([planes*ResBlock.expansion, int((input_shape - 1 + 2 * 0)/stride + 1)])
                )
            else:
                ii_downsample = nn.Sequential(
                    nn.Conv1d(self.in_channels, 
                            planes*ResBlock.expansion, 
                            kernel_size=1, 
                            stride=stride)
                )
   
        layers.append(ResBlock(self.in_channels, 
                                planes, 
                                i_downsample=ii_downsample, 
                                stride=stride,
                                input_shape = input_shape,
                                bn=bn,
                                ln=ln,))
        
        self.in_channels = planes*ResBlock.expansion

        input_shape = int((input_shape - 3 + 2) / stride + 1)
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes, bn=bn, ln=ln, input_shape=input_shape))
            input_shape = int((input_shape - 3 + 2) / 1 + 1)
            
        return nn.Sequential(*layers), input_shape
    
def ResNet18(output_dim, channels=1, batch_norm=True, layer_norm=False):
    return ResNet(ResBlock=Block, 
                  output_dim=output_dim, 
                  num_channels=channels, 
                  bn=batch_norm, 
                  ln=layer_norm,
                  input_shape=46, 
                  layer_list=[2,2,2,2])

def ResNet34(output_dim, channels=1, batch_norm=False, layer_norm=False):
    return ResNet(ResBlock=Block, 
                  output_dim=output_dim, 
                  num_channels=channels, 
                  bn=batch_norm, 
                  ln=layer_norm,
                  input_shape=46, 
                  layer_list=[3,4,6,3])

def ResNet50(output_dim, channels=1, batch_norm=False, layer_norm=False):
    return ResNet(ResBlock = Bottleneck, 
                  output_dim = output_dim, 
                  num_channels = channels, 
                  bn=batch_norm,
                  ln=layer_norm,
                  # input_shape=46,
                  layer_list = [3,4,6,3], )
    
def ResNet101(output_dim, channels=1, batch_norm=True, layer_norm=False):
    return ResNet(ResBlock = Bottleneck, 
                  output_dim = output_dim,
                  num_channels = channels,
                  bn=batch_norm,
                  ln=layer_norm,
                  layer_list = [3,4,23,3], )

def ResNet152(output_dim, channels=3, batch_norm=True):
    return ResNet(Bottleneck, [3,8,36,3], output_dim, channels, bn=batch_norm)

if __name__ == '__main__':

    model = ResNet50(output_dim=12,
                      channels=2,
                      batch_norm=False,
                      layer_norm=False,)
    
    print(model)

    data = torch.randn((64,2,46))

    output = model(data)

    print(output.shape)