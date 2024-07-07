import torch.nn as nn
import torch

class ResNet18(nn.Module):
    def __init__(self, dropout_percentage, input_dim, output_dim):
        super(ResNet18, self).__init__()
        
        self.dropout_percentage = dropout_percentage
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.GELU()
        
        # BLOCK-1 (starting block) input=(224x224) output=(56x56)
        self.conv1 = nn.Conv1d(in_channels = self.input_dim,
                               out_channels = 64, 
                               kernel_size=7, 
                               stride=2, 
                               padding=3)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, 
                                     stride=2, 
                                     padding=1)
        
        # BLOCK-2 (1) input=(56x56) output = (56x56)
        self.conv2_1_1 = nn.Conv1d(in_channels=64, 
                                   out_channels=64, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm2_1_1 = nn.BatchNorm1d(64)
        self.conv2_1_2 = nn.Conv1d(in_channels=64, 
                                   out_channels=64, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm2_1_2 = nn.BatchNorm1d(64)
        self.dropout2_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-2 (2)
        self.conv2_2_1 = nn.Conv1d(in_channels=64, 
                                   out_channels=64, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm2_2_1 = nn.BatchNorm1d(64)
        self.conv2_2_2 = nn.Conv1d(in_channels=64, 
                                   out_channels=64, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm2_2_2 = nn.BatchNorm1d(64)
        self.dropout2_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-3 (1) input=(56x56) output = (28x28)
        self.conv3_1_1 = nn.Conv1d(in_channels=64, 
                                   out_channels=128, 
                                   kernel_size=3, 
                                   stride=2, 
                                   padding=1)
        self.batchnorm3_1_1 = nn.BatchNorm1d(128)
        self.conv3_1_2 = nn.Conv1d(in_channels=128, 
                                   out_channels=128, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm3_1_2 = nn.BatchNorm1d(128)
        self.concat_adjust_3 = nn.Conv1d(in_channels=64, 
                                         out_channels=128, 
                                         kernel_size=1, 
                                         stride=2, 
                                         padding=0)
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-3 (2)
        self.conv3_2_1 = nn.Conv1d(in_channels=128, 
                                   out_channels=128, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm3_2_1 = nn.BatchNorm1d(128)
        self.conv3_2_2 = nn.Conv1d(in_channels=128, 
                                   out_channels=128, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm3_2_2 = nn.BatchNorm1d(128)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-4 (1) input=(28x28) output = (14x14)
        self.conv4_1_1 = nn.Conv1d(in_channels=128, 
                                   out_channels=256, 
                                   kernel_size=3, 
                                   stride=2, 
                                   padding=1)
        self.batchnorm4_1_1 = nn.BatchNorm1d(256)
        self.conv4_1_2 = nn.Conv1d(in_channels=256, 
                                   out_channels=256, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm4_1_2 = nn.BatchNorm1d(256)
        self.concat_adjust_4 = nn.Conv1d(in_channels=128, 
                                         out_channels=256, 
                                         kernel_size=1, 
                                         stride=2, 
                                         padding=0)
        self.dropout4_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-4 (2)
        self.conv4_2_1 = nn.Conv1d(in_channels=256, 
                                   out_channels=256, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm4_2_1 = nn.BatchNorm1d(256)
        self.conv4_2_2 = nn.Conv1d(in_channels=256, 
                                   out_channels=256, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm4_2_2 = nn.BatchNorm1d(256)
        self.dropout4_2 = nn.Dropout(p=self.dropout_percentage)
        
        # BLOCK-5 (1) input=(14x14) output = (7x7)
        self.conv5_1_1 = nn.Conv1d(in_channels=256, 
                                   out_channels=512, 
                                   kernel_size=3, 
                                   stride=2, 
                                   padding=1)
        self.batchnorm5_1_1 = nn.BatchNorm1d(512)
        self.conv5_1_2 = nn.Conv1d(in_channels=512, 
                                   out_channels=512, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm5_1_2 = nn.BatchNorm1d(512)
        self.concat_adjust_5 = nn.Conv1d(in_channels=256, 
                                         out_channels=512, 
                                         kernel_size=1, 
                                         stride=2, 
                                         padding=0)
        self.dropout5_1 = nn.Dropout(p=self.dropout_percentage)
        # BLOCK-5 (2)
        self.conv5_2_1 = nn.Conv1d(in_channels=512, 
                                   out_channels=512, 
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm5_2_1 = nn.BatchNorm1d(512)
        self.conv5_2_2 = nn.Conv1d(in_channels=512, 
                                   out_channels=512,
                                   kernel_size=3, 
                                   stride=1, 
                                   padding=1)
        self.batchnorm5_2_2 = nn.BatchNorm1d(512)
        self.dropout5_2 = nn.Dropout(p=self.dropout_percentage)
        
        # Final Block input=(7x7) 
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=512, out_features=1024)
        self.out = nn.Linear(in_features=1024, out_features=output_dim)
        # END
    
    def forward(self, x):
        
        # block 1 --> Starting block
        x = self.conv1(x)
        
        # print(x.shape)
        # ln = nn.LayerNorm([64,23]).cuda()
        # x = ln(x)
        # print(x.shape)

        # x = self.batchnorm1(x)
        x = self.relu(x)
        op1 = self.maxpool1(x)
           
        # block2 - 1
        x = self.conv2_1_1(op1)
        # x = self.batchnorm2_1_1(x)
        x = self.relu(x)    # conv2_1 

        x = self.conv2_1_2(x)
        # x = self.batchnorm2_1_2(x)                 
        x = self.dropout2_1(x)

        # block2 - Adjust - No adjust in this layer as dimensions are already same
        # block2 - Concatenate 1
        op2_1 = self.relu(x + op1)

        # block2 - 2
        x = self.conv2_2_1(op2_1)
        # x = self.batchnorm2_2_1(x)
        x = self.relu(x)  # conv2_2 

        x = self.conv2_2_2(x)
        # x = self.batchnorm2_2_2(x)                 # conv2_2
        x = self.dropout2_2(x)
        # op - block2
        op2 = self.relu(x + op2_1)
        
        # block3 - 1[Convolution block]
        x = self.conv3_1_1(op2)
        x = self.batchnorm3_1_1(x)
        x = self.relu(x)    # conv3_1
        
        x = self.conv3_1_2(x)
        x = self.batchnorm3_1_2(x)                 # conv3_1
        x = self.dropout3_1(x)

        # block3 - Adjust
        op2 = self.concat_adjust_3(op2) # SKIP CONNECTION
        # block3 - Concatenate 1
        op3_1 = self.relu(x + op2)
        # block3 - 2[Identity Block]
        x = self.conv3_2_1(op3_1)
        # x = self.batchnorm3_2_1(x)
        x = self.relu(x)  # conv3_2
        # x = self.batchnorm3_2_2(self.conv3_2_2(x))                 # conv3_2 
        x = self.dropout3_2(x)
        # op - block3
        op3 = self.relu(x + op3_1)
        
        
        # block4 - 1[Convolition block]
        x = self.conv4_1_1(op3)
        # x = self.batchnorm4_1_1(x)
        x = self.relu(x)    # conv4_1

        x = self.conv4_1_2(x)
        # x = self.batchnorm4_1_2(x)                 # conv4_1
        x = self.dropout4_1(x)

        # block4 - Adjust
        op3 = self.concat_adjust_4(op3) # SKIP CONNECTION
        # block4 - Concatenate 1
        op4_1 = self.relu(x + op3)
        # block4 - 2[Identity Block]
        x = self.conv4_2_1(op4_1)
        # x = self.batchnorm4_2_1(x)
        x = self.relu(x)  # conv4_2

        x = self.conv4_2_2(x)
        # x = self.batchnorm4_2_2(x)                 # conv4_2
        x = self.dropout4_2(x)
        # op - block4
        op4 = self.relu(x + op4_1)

        
        # block5 - 1[Convolution Block]
        x = self.conv5_1_1(op4)
        # x = self.batchnorm5_1_1(x)
        x = self.relu(x)    # conv5_1

        x = self.conv5_1_2(x)
        # x = self.batchnorm5_1_2(x)                 # conv5_1
        x = self.dropout5_1(x)
        # block5 - Adjust
        op4 = self.concat_adjust_5(op4) # SKIP CONNECTION
        # block5 - Concatenate 1
        op5_1 = self.relu(x + op4)
        # block5 - 2[Identity Block]
        x = self.conv5_2_1(op5_1)
        # x = self.batchnorm5_2_1(x)
        x = self.relu(x)  # conv5_2

        x = self.conv5_2_1(x)
        # x = self.batchnorm5_2_1(x)                 # conv5_2
        x = self.dropout5_2(x)
        # op - block5
        op5 = self.relu(x + op5_1)


        # FINAL BLOCK - classifier 
        x = self.avgpool(op5)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc(x))
        x = self.out(x)

        return x
    
if __name__ == '__main__':
    model = ResNet18(input_dim=1,
                    output_dim=12,
                    dropout_percentage=0.5,)

    data = torch.randn((64,1,30))

    print(model)

    output = model(data)
    print(output.shape)