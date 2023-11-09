import torch
import torch.nn as nn

class Classifier(nn.Module):

    def __init__ (self, pen_lin_nodes = 128, pen_nonlin_nodes = None, dropout=0.5, backbone_dense_nodes=1024):
    
        super(Classifier, self).__init__()

        self.in_channels = 1
        self.num_classes = 10
        self.dropout = dropout
        self.backbone_dense_nodes = backbone_dense_nodes
        self.pen_lin_nodes = pen_lin_nodes
        self.pen_nonlin_nodes = pen_nonlin_nodes
               
    
    def make_backbone_dense_layers (self, input_dims):

        self.backbone_dense_layers = nn.Sequential(
            nn.Linear(input_dims, self.backbone_dense_nodes),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.backbone_dense_nodes, self.backbone_dense_nodes),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.backbone_dense_nodes, self.backbone_dense_nodes),
            nn.ReLU(),
            )     


    def get_penultimate (self):

        if self.pen_lin_nodes:
            pen_layer = nn.Linear(in_features=self.backbone_dense_nodes, out_features=self.pen_lin_nodes)
            output_layer = nn.Linear(in_features=self.pen_lin_nodes, out_features=self.num_classes)
        elif self.pen_nonlin_nodes:
            pen_layer = nn.Sequential(
                nn.Linear(in_features=self.backbone_dense_nodes, out_features=self.pen_nonlin_nodes),
                nn.ReLU()           
            )
            output_layer = nn.Linear(in_features=self.pen_nonlin_nodes, out_features=self.num_classes)
        else:
            pen_layer = None
            output_layer = nn.Linear(in_features=self.backbone_dense_nodes, out_features=self.num_classes)    

        return pen_layer, output_layer


    def from_dense_forward (self, x):

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
   
    def __init__ (self, pen_lin_nodes = 128, pen_nonlin_nodes = None, dropout=0.5, backbone_dense_nodes=1024):
         
        super().__init__( pen_lin_nodes, pen_nonlin_nodes, dropout, backbone_dense_nodes)
        
        self.make_backbone_dense_layers (input_dims=784)
        self.pen_layer, self.output_layer = self.get_penultimate()
        
    def forward(self, x):
        
        x = torch.flatten(x, start_dim=1)

        return self.from_dense_forward(x)



class MLPconvs(Classifier):

    def __init__ (self, pen_lin_nodes = 128, pen_nonlin_nodes = None, dropout=0.5, backbone_dense_nodes=1024):

        super().__init__( pen_lin_nodes, pen_nonlin_nodes, dropout, backbone_dense_nodes)

        self.backbone_conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.make_backbone_dense_layers (input_dims=512*3*3)
        self.pen_layer, self.output_layer = self.get_penultimate()
            
        
    def forward(self, x):

        x = self.backbone_conv_layers(x)    
        x = torch.flatten(x, start_dim=1)

        return self.from_dense_forward(x)


class VGG11(Classifier):

    def __init__ (self, pen_lin_nodes = 128, pen_nonlin_nodes = None, dropout=0.5, backbone_dense_nodes=1024):

        super().__init__( pen_lin_nodes, pen_nonlin_nodes, dropout, backbone_dense_nodes)

        self.backbone_conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.make_backbone_dense_layers (input_dims=512*1*1)
        self.pen_layer, self.output_layer = self.get_penultimate()
            
        
    def forward(self, x):

        x = self.backbone_conv_layers(x)    
        x = torch.flatten(x, start_dim=1)

        return self.from_dense_forward(x)

