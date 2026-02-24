import torch
import torch.nn as nn
import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, img_height, num_classes, map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        # CNN Feature Extractor
        # Very simple VGG-style extractor.
        # Takes (1, img_height, W) -> (C, 1, W/4) roughly
        assert img_height % 16 == 0, "Image height must be a multiple of 16"
        
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16 -> 32x8

        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8 -> 64x4

        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4 -> 128x2

        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2 -> 256x1

        convRelu(6, True)  # 512x1 -> 512x1

        self.cnn = cnn
        
        # Map to sequence
        self.map_to_seq = nn.Linear(512 * (img_height // 16), map_to_seq_hidden) 
        # Actually with the pooling logic above:
        # H=32 -> Pool1 -> 16 -> Pool2 -> 8 -> Pool3(2,1) -> 4 -> Pool4(2,1) -> 2 -> Conv6 -> 1?
        # Wait, the pooling kernel (2,2) reduces H by half.
        # 32 -> 16 -> 8 -> 4 -> 2 -> 1.
        # So output height is 1. Number of channels is 512.
        
        self.rnn = nn.Sequential(
            nn.LSTM(512, rnn_hidden, bidirectional=True, batch_first=True),
            # nn.LSTM(rnn_hidden * 2, rnn_hidden, bidirectional=True, batch_first=True) # Stacked
        )

        self.embedding = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W)
        conv = self.cnn(x) # (B, 512, 1, W_new)
        
        # Squeeze H dimension (should be 1)
        # Check H
        # print("Conv shape:", conv.shape)
        
        b, c, h, w = conv.size()
        assert h == 1, "The input height must be reduced to 1 by the CNN"
        
        # Permute to (B, W, C) for RNN
        conv = conv.squeeze(2) # (B, 512, W)
        conv = conv.permute(0, 2, 1) # (B, W, 512)
        
        # RNN
        # self.rnn returns (output, (h_n, c_n))
        # output: (B, W, 2*hidden)
        recurrent, _ = self.rnn(conv)
        
        # FC
        # (B, W, num_classes)
        output = self.embedding(recurrent)
        
        return output
