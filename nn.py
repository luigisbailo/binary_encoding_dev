import torch
import torch.nn as nn

class Classifier_cnn (nn.Module):
   
    def __init__(self, pen_lin_nodes=64, backbone_nodes = [1024]):
        super(Classifier_cnn, self).__init__()
        self.pen_lin_nodes = pen_lin_nodes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.hidden_layers = nn.ModuleList()      
        self.hidden_layers.append(nn.Linear(in_features=576, out_features=backbone_nodes[0]))

        for l in range (1, len(backbone_nodes)):
            self.hidden_layers.append(nn.Linear(in_features=backbone_nodes[l-1], out_features=backbone_nodes[l]))

        if pen_lin_nodes:
            self.pen_layer = nn.Linear(in_features=backbone_nodes[-1], out_features=pen_lin_nodes)
            self.output_layer = nn.Linear(in_features=pen_lin_nodes, out_features=10)
        else:
            self.output_layer = nn.Linear(in_features=backbone_nodes[-1], out_features=10)
        
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = self.activation(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)
        x = self.activation(self.bn3(self.conv3(x)))
        x = nn.functional.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # Flatten
        
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        if self.pen_lin_nodes:
            x_pen = self.pen_layer(x)
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)

        else:
            x_pen = x
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        
        return y.reshape(-1,10), x_pen