# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn
import random

from PIL import ImageFilter


class MoCoV3(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=192, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCoV3, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        # self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_q = base_encoder
        # self.encoder_k = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder


        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        # self.predictor = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim, dim),
        # )

        self.projector = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=1024, out_channels=128, kernel_size=1),
        )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    def forward(self, x1, x2):
        """
        Input:
            x1: a batch of query images
            x2: a batch of key images
        Output:
            q1,q2,k1,k2
        """
        x1 = x1.type(torch.cuda.FloatTensor)
        x2 = x2.type(torch.cuda.FloatTensor)
        # compute query featurs
        q = self.encoder_q(x1)
        q = self.projector(q)
        # q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.projector(self.encoder_k(x2))  # keys: NxC
            # k = nn.functional.normalize(k, dim=1)

        return q, k