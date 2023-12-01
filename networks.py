import torch
import torch.nn as nn
import importlib
from typing import cast

class Classifier(nn.Module):

    def __init__ (self, model, architecture, in_channels, num_classes):
    
        super(Classifier, self).__init__()

        torch_module= importlib.import_module("torch.nn")

        self.architecture = architecture
        self.architecture_hypers = architecture['hypers']
        self.backbone_dense_nodes= architecture['hypers']['backbone_dense_nodes']
        self.activation= architecture['hypers']['activation']
        self.dropout = architecture['hypers']['dropout']
        pen_nodes = architecture['hypers']['pen_nodes']

        if model == 'bin_enc' or model == 'lin_pen':
            self.pen_lin_nodes = pen_nodes
            self.pen_nonlin_nodes = None
        elif model == 'no_pen':
            self.pen_lin_nodes = None
            self.pen_nonlin_nodes = None
        elif model == 'nonlin_pen':
            self.pen_lin_nodes = None
            self.pen_nonlin_nodes = pen_nodes

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.activation = getattr(torch_module, self.activation)

    def make_dense_classifier (self, input_dims):

        l_layers = nn.ModuleList ()
        l_nodes = [input_dims] + self.backbone_dense_nodes
 
        for i in range(len(l_nodes)-1):
            l_layers.append(nn.Linear(l_nodes[i],l_nodes[i+1]))
            l_layers.append(self.activation())
            if self.architecture_hypers['dense_bn']:
                l_layers.append(nn.BatchNorm1d(l_nodes[i+1]))
            if self.dropout>0:
                l_layers.append(nn.Dropout(p=self.dropout))


        self.dense_classifier = nn.Sequential(*l_layers)


    def make_penultimate (self, input_dims):

        if self.pen_lin_nodes:
            pen_layer = nn.Linear(in_features=input_dims, out_features=self.pen_lin_nodes)
            output_layer = nn.Linear(in_features=self.pen_lin_nodes, out_features=self.num_classes)
        elif self.pen_nonlin_nodes:
            pen_layer = nn.Sequential(
                nn.Linear(in_features=input_dims, out_features=self.pen_nonlin_nodes),
                self.activation()           
            )
            output_layer = nn.Linear(in_features=self.pen_nonlin_nodes, out_features=self.num_classes)
        else:
            pen_layer = None
            output_layer = nn.Linear(in_features=input_dims, out_features=self.num_classes)    

        return pen_layer, output_layer


    def classifier_forward (self, x):
        
        if self.backbone_dense_nodes:
            x = self.dense_classifier(x)

        if self.pen_lin_nodes:
            x_pen = self.pen_layer(x)
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        elif self.pen_nonlin_nodes:
            x_pen = self.pen_layer(x)
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        else:
            x_pen = x
            x_output = self.output_layer(x_pen)
            y = torch.softmax(x_output, dim=-1)
        
        return y.reshape(-1,self.num_classes), x_pen
    

class MLPvanilla (Classifier):
   
    def __init__ (self,  model, architecture, in_channels, num_classes):
         
        super().__init__( model, architecture, in_channels, num_classes)
        
        self.make_dense_classifier (input_dims=784)
        self.pen_layer, self.output_layer = self.make_penultimate(self.backbone_dense_nodes[-1])
        
    def forward(self, x):
        
        x = torch.flatten(x, start_dim=1)

        return self.classifier_forward(x)



class MLPconvs(Classifier):

    def __init__ (self,  model, architecture, in_channels, num_classes):

        super().__init__( model, architecture, in_channels, num_classes)

        self.backbone_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.activation(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            self.activation(),
        )
        if self.backbone_dense_nodes:
            self.make_dense_classifier (input_dims=512*3*3)
            self.pen_layer, self.output_layer = self.make_penultimate(self.backbone_dense_nodes[-1])
    
        else:
            self.pen_layer, self.output_layer = self.make_penultimate(input_dims=512*3*3)
            
        
    def forward(self, x):

        x = self.backbone_dense_nodes(x)    
        x = torch.flatten(x, start_dim=1)

        return self.classifier_forward(x)


VGG_cfgs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(Classifier):

    def __init__ (self, model, architecture, in_channels, num_classes):

        super().__init__( model, architecture, in_channels, num_classes)

        self.make_backbone_layers()
        if self.backbone_dense_nodes:
            self.make_dense_classifier (input_dims=512*1*1)
            self.pen_layer, self.output_layer = self.make_penultimate(self.backbone_dense_nodes[-1])
        else:
            self.pen_layer, self.output_layer = self.make_penultimate(input_dims=512*1*1)
        
    def make_backbone_layers(self):
    
        cfg = VGG_cfgs[str(self.architecture['backbone_model'])]
        in_channels = self.in_channels
        l_layers = nn.ModuleList ()
        for v in cfg:
            if v == 'M':
                l_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                l_layers.append(conv2d)
                if self.architecture_hypers['conv_bn']:
                    l_layers.append(nn.BatchNorm2d(v))
                l_layers.append(self.activation())
                in_channels = v

        self.backbone_layers =  nn.Sequential(*l_layers)
  
        
    def forward(self, x):

        x = self.backbone_layers(x)    
        x = torch.flatten(x, start_dim=1)

        return self.classifier_forward(x)




# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):

    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                        stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels):

    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                        stride=1, padding=0, bias=False)

# class BasicBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, activation, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
        
#         self.expansion = 1        
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.activation = activation(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample

    
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.activation(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.activation(out)
        # return out


class Bottleneck(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation, downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.expansion = 4        
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = conv1x1(out_channels, self.expansion*out_channels)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.activation(out)
        return out

    
# ResNet
class ResNet50(Classifier):

    def __init__ (self, model, architecture, in_channels, num_classes):

        super().__init__( model, architecture, in_channels, num_classes)
        
        self.expansion = 4
        self.in_channels = 64

        layers = [3, 4, 6, 3]
        self.make_backbone_layers(Bottleneck, layers)
        
        if self.backbone_dense_nodes:
            self.make_dense_classifier (input_dims=512*self.expansion)
            self.pen_layer, self.output_layer = self.make_penultimate(self.backbone_dense_nodes[-1])
        else:
            self.pen_layer, self.output_layer = self.make_penultimate(input_dims=512*self.expansion)


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        layers = []
                
        if stride != 1 or self.in_channels != out_channels*self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
            
        layers.append(block(self.in_channels, out_channels, self.activation, downsample=downsample, stride=stride))
        self.in_channels = out_channels*self.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, self.activation))
        return nn.Sequential(*layers)


    def make_backbone_layers(self, block, layers):
            
        self.in_channels = 64
        self.conv = conv3x3(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        l_layers = nn.ModuleList ()
        l_layers.append(self.conv)
        l_layers.append(self.bn)
        l_layers.append(self.activation())
        # l_layers.append(self.max_pool)
        l_layers.append(self.layer1)
        l_layers.append(self.layer2)
        l_layers.append(self.layer3)
        l_layers.append(self.layer4)
        l_layers.append(self.avg_pool)

        self.backbone_layers =  nn.Sequential(*l_layers)

    def forward(self, x):

        x = self.backbone_layers(x)    
        x = torch.flatten(x, start_dim=1)

        return self.classifier_forward(x)

    

