import torch
import torch.nn as nn
import importlib

class Classifier(nn.Module):

    def __init__ (self, model, architecture, dropout=0.5):
    
        super(Classifier, self).__init__()

        torch_module= importlib.import_module("torch.nn")

        self.backbone_dense_nodes= architecture['hypers']['backbone_dense_nodes']
        self.activation= architecture['hypers']['activation']
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

        self.in_channels = 1
        self.num_classes = 10
        self.dropout = dropout
        self.activation = getattr(torch_module, self.activation)

    def make_backbone_dense_layers (self, input_dims):

        l_layers = nn.ModuleList ()
        l_nodes = [input_dims] + self.backbone_dense_nodes
 
        for i in range(len(l_nodes)-1):
            l_layers.append(nn.Linear(l_nodes[i],l_nodes[i+1]))
            l_layers.append(self.activation())
            if (i<len(l_nodes)-2):
                l_layers.append(nn.Dropout(p=self.dropout))

        self.backbone_dense_layers = nn.Sequential(*l_layers)


    def get_penultimate (self, input_dims):

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


    def from_conv_forward (self, x):
        
        if self.backbone_dense_nodes:
            x = self.backbone_dense_layers(x)

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
   
    def __init__ (self,  model, architecture, dropout=0.5):
         
        super().__init__( model, architecture,  dropout)
        
        self.make_backbone_dense_layers (input_dims=784)
        self.pen_layer, self.output_layer = self.get_penultimate(self.backbone_dense_nodes[-1])
        
    def forward(self, x):
        
        x = torch.flatten(x, start_dim=1)

        return self.from_conv_forward(x)



class MLPconvs(Classifier):

    def __init__ (self,  model, architecture,  dropout=0.5):

        super().__init__( model, architecture,  dropout)

        self.backbone_conv_layers = nn.Sequential(
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
            self.make_backbone_dense_layers (input_dims=512*3*3)
            self.pen_layer, self.output_layer = self.get_penultimate(self.backbone_dense_nodes[-1])
    
        else:
            self.pen_layer, self.output_layer = self.get_penultimate(input_dims=512*3*3)
            
        
    def forward(self, x):

        x = self.backbone_dense_nodes(x)    
        x = torch.flatten(x, start_dim=1)

        return self.from_conv_forward(x)


class VGG11(Classifier):

    def __init__ (self, model, architecture, dropout=0.5):

        super().__init__( model, architecture,  dropout)

        self.backbone_conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            self.activation(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            self.activation(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            self.activation(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if self.backbone_dense_nodes:
            self.make_backbone_dense_layers (input_dims=512*1*1)
            self.pen_layer, self.output_layer = self.get_penultimate(self.backbone_dense_nodes[-1])
        else:
            self.pen_layer, self.output_layer = self.get_penultimate(input_dims=512*1*1)
            
        
    def forward(self, x):

        x = self.backbone_conv_layers(x)    
        x = torch.flatten(x, start_dim=1)

        return self.from_conv_forward(x)


class VGG13(Classifier):

    def __init__ (self,  model, architecture, dropout=0.5):

        super().__init__( model, architecture, dropout)

        self.backbone_conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            self.activation(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            self.activation(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            self.activation(),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            self.activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if self.backbone_dense_nodes:
        
            self.make_backbone_dense_layers (input_dims=512*1*1)
            self.pen_layer, self.output_layer = self.get_penultimate(self.backbone_dense_nodes[-1])
    
        else:
            self.pen_layer, self.output_layer = self.get_penultimate(input_dims=512*1*1)
            
        
    def forward(self, x):

        x = self.backbone_conv_layers(x)    
        x = torch.flatten(x, start_dim=1)

        return self.from_conv_forward(x)


