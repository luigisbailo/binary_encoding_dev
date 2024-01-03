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
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def make_dense_classifier (self, input_dims):

        l_layers = nn.ModuleList ()
        l_nodes = [input_dims] + self.backbone_dense_nodes
 
        for i in range(len(l_nodes)-1):
            if self.dropout:
                if len(self.dropout) != len (self.backbone_dense_nodes):
                    print('ERROR: length of dropout values different from lenght head layers')
                    sys.exit(1)
                l_layers.append(nn.Dropout(p=self.dropout[i]))

            l_layers.append(nn.Linear(l_nodes[i],l_nodes[i+1]))
            l_layers.append(self.activation())
            if self.architecture_hypers['dense_bn']:
                l_layers.append(nn.BatchNorm1d(l_nodes[i+1]))


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
            # y = torch.softmax(x_output, dim=-1)
        elif self.pen_nonlin_nodes:
            x_pen = self.pen_layer(x)
            x_output = self.output_layer(x_pen)
            # y = torch.softmax(x_output, dim=-1)
        else:
            x_pen = x
            x_output = self.output_layer(x_pen)
            # y = torch.softmax(x_output, dim=-1)
            
        return x_output.reshape(-1,self.num_classes), x_pen
    

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



class Bottleneck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, activation, stride=1):
        super(Bottleneck, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),     
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion*out_channels)
        )
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )
        self.activation = activation
                
    def forward(self, x):

        return self.activation(inplace=True)(self.residual_function(x) + self.shortcut(x))


    
# ResNet
class ResNet50(Classifier):

    def __init__ (self, model, architecture, in_channels, num_classes):

        super().__init__( model, architecture, in_channels, num_classes)
        
        self.in_channels = 64

        blocks = [3, 4, 6, 3]
        self.make_backbone_layers(Bottleneck, blocks)

        if self.backbone_dense_nodes:
            self.make_dense_classifier (input_dims=512*Bottleneck.expansion)
            self.pen_layer, self.output_layer = self.make_penultimate(self.backbone_dense_nodes[-1])
        else:
            self.pen_layer, self.output_layer = self.make_penultimate(input_dims=512*Bottleneck.expansion)


    def make_layer(self, block, out_channels, num_blocks, stride=1):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, self.activation, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)                            



    def make_backbone_layers(self, block, blocks):
            
        self.in_channels = 64
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.activation(inplace=True),
        )
        self.layer1 = self.make_layer(block, 64, blocks[0], 1)
        self.layer2 = self.make_layer(block, 128, blocks[1], 2)
        self.layer3 = self.make_layer(block, 256, blocks[2], 2)
        self.layer4 = self.make_layer(block, 512, blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        

    def forward(self, x):
        output = self.layer0(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)   
        output = torch.flatten(output, start_dim=1)

        return self.classifier_forward(output)

    

