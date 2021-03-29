
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class Net(nn.Module):
    def __init__(self,model_name='efficientnet-b3',num_classes=1000,pretrained=True):
        super().__init__()
        MODEL_TYPE={'resnet-50','resnet-101','resnet-152',\
                    'efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3'}

        assert model_name in MODEL_TYPE, f"Available : {MODEL_TYPE}"
        baseline,version=model_name.split('-')
        if baseline == 'resnet':
            if version == '50':
                self.backbone=models.resnet50(pretrained=pretrained)
            elif version == '101':
                self.backbone=models.resnet101(pretrained=pretrained)
            elif version == '152':
                self.backbone=models.resnet152(pretrained=pretrained)

            in_features=self.backbone.fc.in_features
            self.backbone.fc=nn.Linear(in_features,num_classes)
        
        elif baseline == 'efficientnet':
            if pretrained :
                self.backbone=EfficientNet.from_pretrained(model_name,num_classes=num_classes)
            else:
                self.backbone=EfficientNet.from_name(model_name,num_classes=num_classes)

    def forward(self,x):
        return self.backbone(x)