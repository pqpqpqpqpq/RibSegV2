import importlib

import torch
import torch.nn as nn

from .building_blocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv

def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, f_maps=16, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        self.encoders = Encoders(f_maps, in_channels, layer_order, num_groups)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # encoder part
        x, encoders_features = self.encoders(x)

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        # x = self.dropout(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric

        return x

class Encoders(nn.Module):
    def __init__(self, f_maps, in_channels, layer_order, num_groups):
        super().__init__()
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoder.trainable = False
            encoders.append(encoder)

        # with torch.no_grad():
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        # pool_fea = self.avg_pool(encoders_features[0]).squeeze(0).squeeze(1).squeeze(1).squeeze(1)
        encoders_features = encoders_features[1:]

        return x, encoders_features

if __name__ == "__main__":
    net = UNet3D(in_channels=1, out_channels=2, final_sigmoid=False)
    print(net)