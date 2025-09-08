

import torch.nn as nn

class Adapter(nn.Module):

    def __init__(self,
                 clip_model,
                 clip_model_cfg,
                 target,
                 layers_out_config,
                 ):

        super(Adapter, self).__init__()

        input_sizes = [clip_model_cfg['vision_cfg']['width']] * len(layers_out_config)

        for i,input_size in enumerate(input_sizes):
            self.add_module("{}_adapter".format(i), nn.Sequential(nn.Conv2d(input_size, target, 1, 1)))


    def forward(self, tokens):
        vision_features=[]
        for i,token in enumerate(tokens):
            vision_feature=getattr(self,'{}_adapter'.format(i))(token).contiguous().permute(0, 2, 3, 1)
            vision_feature = vision_feature / vision_feature.norm(dim=-1, keepdim=True)
            vision_features.append(vision_feature)
        return vision_features
