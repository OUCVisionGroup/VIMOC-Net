from torch import nn, Tensor
import torch
from torch.nn import functional as F
from torchvision import models

class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)

    def __make_weight(self,feature,scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])]+ [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)

class DAFG(nn.Module):
    def __init__(self, load_weights=False):
        super(DAFG, self).__init__()
        self.seen = 0
        self.context = ContextualModule(512, 512)
        self.weight_gen_net = WeightGenerationNetwork(2, 3)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 1024,batch_norm=True, dilation = True)
        self.output_layer = nn.Conv2d(64, 10, kernel_size=1)
              
        self.weight_level = nn.Sequential(
            nn.Conv2d(1024, 64,  kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,  kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.depth_layer1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
        )

        
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.reg_layer1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.reg_layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.reg_layer3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),   
        )

        self.out_layer1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),        
        )

        self.depth = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  
        )
        
        self.relu = nn.ReLU()

        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            # address the mismatch in key names for python 3
            pretrained_dict = {k[9:]: v for k, v in mod.state_dict().items() if k[9:] in self.frontend.state_dict()}
            self.frontend.load_state_dict(pretrained_dict)

    def forward(self,prev_img,img,diff):
        
        prev_img = self.frontend(prev_img)
        img = self.frontend(img)
        
        prev_img = self.context(prev_img)   
        img = self.context(img)

        x = torch.cat((prev_img,img),1) 

        depth1=self.depth_layer1(x)
        depth=self.depth_head(depth1)


        depth_x = self.depth(depth1)
        x = self.out_layer1(x)

        M = F.softmax(self.weight_level(torch.cat((x,depth_x),1)), dim=1)
        x_fuse = x * M[:,0:1,:,:] + depth_x * M[:,1:2,:,:]
    
            
        x1 = self.reg_layer1(x_fuse)
        x2 = self.reg_layer2(x_fuse)
        x3 = self.reg_layer3(x_fuse)

        weight = self.weight_gen_net(diff)
        x = x1 * weight[:,0:1,:,:]+ x2 * weight[:,1:2,:,:]+ x3 * weight[:,2:,:,:] 

     
        x = self.output_layer(x)
        x = self.relu(x)

        return  x, depth
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class WeightGenerationNetwork(nn.Module):
    def __init__(self, input_channels, num_scales):
        super(WeightGenerationNetwork, self).__init__()
        self.input_channels = input_channels
        self.num_scales = num_scales
        
        self.reg_layer1 = nn.Sequential(
             nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 128, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(128, 1, kernel_size=3, padding=1),
             nn.ReLU(inplace=True),
        )

        self.reg_layer2 = nn.Sequential(
             nn.Conv2d(input_channels, 64, kernel_size=3, padding=2, dilation=2),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
             nn.ReLU(inplace=True),
             nn.Conv2d(128, 1, kernel_size=3, padding=2, dilation =2),
             nn.ReLU(inplace=True),
        )

        self.reg_layer3 = nn.Sequential(
             nn.Conv2d(input_channels, 64, kernel_size=3, padding=3, dilation =3),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 128, kernel_size=3, padding=3, dilation=3),
             nn.ReLU(inplace=True),
             nn.Conv2d(128, 1, kernel_size=3, padding=3, dilation =3),
             nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((45, 80))
        self.fc = nn.Linear(128, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.gap(x)

        x1 = self.reg_layer1(x)
        x2 = self.reg_layer1(x)
        x3 = self.reg_layer1(x)
       
        x = torch.cat((x1,x2,x3),1) 
        weights = F.softmax(x, dim=1)
       
        return weights