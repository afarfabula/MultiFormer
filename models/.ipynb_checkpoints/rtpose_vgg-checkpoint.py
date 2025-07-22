"""CPM Pytorch Implementation"""
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
from torch.nn import init
from models.cbam import CBAM
def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)


def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class rtpose_model(nn.Module):
    def __init__(self, model_dict):
        super(rtpose_model, self).__init__()
        #self.tf1 = ChannelTransformer_OPP(vis=False, img_size=[36, 36], channel_num=185, num_layers=1, num_heads=3)
        #self.tf2 = ChannelTransformer_OPP(vis=False, img_size=[36, 36], channel_num=185, num_layers=1, num_heads=3)
        #self.tf3 = ChannelTransformer_OPP(vis=False, img_size=[36, 36], channel_num=185, num_layers=1, num_heads=3)
        #self.model0 = model_dict['block0']
        self.cbam1 = CBAM(gate_channels=57)
        self.cbam2 = CBAM(gate_channels=57)
        #elf.cbam3 = CBAM(gate_channels=23)
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        #self.model4_1 = model_dict['block4_1']
        # self.model5_1 = model_dict['block5_1']
        # self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        #self.model4_2 = model_dict['block4_2']
        # self.model5_2 = model_dict['block5_2']
        # self.model6_2 = model_dict['block6_2']

        self._initialize_weights_norm()

    def forward(self, x , st1 , st2):

        #saved_for_loss = []
        #out1 = self.model0(x)
        out1=x
        #print('特征提取器输出',out1.shape)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_2, out1_1], 1)
        #print('注意力前',out2.shape)
        
        #微调
        #out1_weight = self.cbam1(out2)
        #预训练
        out1_weight = self.cbam1( out2)
        
        #print('权重维度',out1_weight.shape)
        out2 =out1_weight*out1 
        #print('注意力后',out2.shape)
        
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_2, out2_1], 1)
        
        #微调
        #out3_weight = self.cbam2(out3)
        #预训练
        out3_weight = self.cbam2( out3)
        
        
        out3 = out3_weight*out2
        

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        #out4 = torch.cat([out3_1, out3_2, out1], 1)
        #out4 = self.tf3(out4)
        

        #out4_1 = self.model4_1(out4)
        #out4_2 = self.model4_2(out4)
        # out5 = torch.cat([out4_1, out4_2, out1], 1)
        

        # out5_1 = self.model5_1(out5)
        # out5_2 = self.model5_2(out5)
        # out6 = torch.cat([out5_1, out5_2, out1], 1)
        # saved_for_loss.append(out5_1)
        # saved_for_loss.append(out5_2)
        #
        # out6_1 = self.model6_1(out6)
        # out6_2 = self.model6_2(out6)
        # saved_for_loss.append(out6_1)
        # saved_for_loss.append(out6_2)

        return out3_1, out3_2, out3_1, out3_2, out2_1, out2_2, out1_1, out1_2

    def _initialize_weights_norm(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    init.constant_(m.bias, 0.0)

        # last layer of these block don't have Relu
        init.normal_(self.model1_1[6].weight, std=0.01)
        init.normal_(self.model1_2[6].weight, std=0.01)

        init.normal_(self.model2_1[8].weight, std=0.01)
        init.normal_(self.model3_1[8].weight, std=0.01)
        #init.normal_(self.model4_1[8].weight, std=0.01)
        # init.normal_(self.model5_1[12].weight, std=0.01)
        # init.normal_(self.model6_1[12].weight, std=0.01)

        init.normal_(self.model2_2[8].weight, std=0.01)
        init.normal_(self.model3_2[8].weight, std=0.01)
        #init.normal_(self.model4_2[8].weight, std=0.01)
        # init.normal_(self.model5_2[12].weight, std=0.01)
        # init.normal_(self.model6_2[12].weight, std=0.01)


def get_model():
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet'
    Returns: Module, the defined model
    """
    blocks = {}
    models = {}
     # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [256, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          # {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [256, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          # {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    blocks['block2_1' ] = [
            {'Mconv1_stage2_L1' : [256, 128, 7, 1, 3]},
            {'Mconv2_stage2_L1' : [128, 128, 7, 1, 3]},
            {'Mconv3_stage2_L1' : [128, 128, 7, 1, 3]},
            #{'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            #{'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage2_L1' : [128, 128, 1, 1, 0]},
            {'Mconv7_stage2_L1' : [128, 38, 1, 1, 0]} ]

    blocks['block2_2'] = [
            {'Mconv1_stage2_L2' : [256, 128, 7, 1, 3]},
            {'Mconv2_stage2_L2' : [128, 128, 7, 1, 3]},
            {'Mconv3_stage2_L2' : [128, 128, 7, 1, 3]},
            #{'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            #{'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage2_L2' : [128, 128, 1, 1, 0]},
            {'Mconv7_stage2_L2' : [128, 19, 1, 1, 0]} ]
    
        # Stages 2 - 6
    blocks['block3_1' ] = [
            {'Mconv1_stage3_L1' : [256, 128, 7, 1, 3]},
            {'Mconv2_stage3_L1' : [128, 128, 7, 1, 3]},
            {'Mconv3_stage3_L1' : [128, 128, 7, 1, 3]},
            #{'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            #{'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage3_L1' : [128, 128, 1, 1, 0]},
            {'Mconv7_stage3_L1' : [128, 38, 1, 1, 0]} ]

    blocks['block3_2'] = [
            {'Mconv1_stage3_L2' : [256, 128, 7, 1, 3]},
            {'Mconv2_stage3_L2' : [128, 128, 7, 1, 3]},
            {'Mconv3_stage3_L2' : [128, 128, 7, 1, 3]},
            #{'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            #{'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage3_L2' : [128, 128, 1, 1, 0]},
            {'Mconv7_stage3_L2' : [128, 19, 1, 1, 0]} ]



    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    model = rtpose_model(models)
    # 冻结所有参数
    #for param in model.parameters():
        #param.requires_grad = False
    return model


