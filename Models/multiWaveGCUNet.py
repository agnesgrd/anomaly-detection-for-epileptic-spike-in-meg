# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 17:10
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : MultWaveGC-UNet.py
# @Software: PyCharm
from torch import nn
import torch
from .net import WaveDecompose, IDWTLayers, EmbeddingGraph, MultiLevelWaveGCN
from utils import adjacency_matrix

class MultiWaveGCUNet(nn.Module):
    """Using Multi-scale DWT decomposition, different frequency components are
    input into GCN as features to aggregate node features, and finally IDWT is
    used to restore the aggregated features layer by layer.

    During the restoration process, the losses are calculated
    separately, which will be used in the final training.
    """
    def __init__(self, input_channel, embedding_dim=64, top_k=30, input_node_dim=2, graph_alpha=3, device=torch.device('cuda:1'), gc_depth=1, batch_size=128, filters = [32, 64, 128]):
        super(MultiWaveGCUNet, self).__init__()

        #Multilevel DWT
        self.wave_decompose = WaveDecompose(input_channel)

        # Dynamic graph network
        graph_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(adjacency_matrix()))
        self.graph1 = graph_embedding
        self.graph2 = graph_embedding
        self.graph3 = graph_embedding
        self.input_channel = input_channel
        wave_conv_out_dim = filters

        # GCN
        self.conv1_gcn_input1 = nn.Sequential(
            nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[0], kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )
        self.conv1_gcn_input2 = nn.Sequential(
            nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[1], kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )

        self.conv1_gcn_input3 = nn.Sequential(
            nn.Conv2d(in_channels=input_node_dim, out_channels=wave_conv_out_dim[2], kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )

        self.wave_gcn_layer1 = MultiLevelWaveGCN(input_channel=wave_conv_out_dim[0], gcn_depth=gc_depth)

        self.wave_gcn_layer2 = MultiLevelWaveGCN(input_channel=wave_conv_out_dim[1], gcn_depth=gc_depth)

        self.wave_gcn_layer3 = MultiLevelWaveGCN(input_channel=wave_conv_out_dim[2], gcn_depth=gc_depth)

        self.wave_generate_high_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[0], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.wave_generate_high_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[1], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.wave_generate_high_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[2], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.wave_generate_low_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=wave_conv_out_dim[2], out_channels=1, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU()
        )

        # IDWT
        self.idwt_layer_1 = IDWTLayers()
        self.idwt_layer_2 = IDWTLayers()
        self.idwt_layer_3 = IDWTLayers()

    def forward(self, x, idx, device, latent=False):
        adj1 = self.graph1(idx)
        adj2 = self.graph2(idx)
        adj3 = self.graph3(idx)

        # ablation
        wave_feature = self.wave_decompose(x)

        # reconstruction loss 0：x

        # reconstruction loss 1：wave_high1
        wave_low1 = wave_feature[0][0]
        wave_high1 = wave_feature[0][1]

        wave_feature1 = torch.cat((torch.unsqueeze(wave_feature[0][0], dim=1), torch.unsqueeze(wave_feature[0][1], dim=1)), dim=1)
        wave_feature1 = wave_feature1.transpose(2, 3)
        wave_feature1 = self.conv1_gcn_input1(wave_feature1)
        # gcn input feature：wave_feature1

        wave_latent1 = self.wave_gcn_layer1(wave_feature1, adj1)

        # latent_representation1：wave_latent1
        generated_high1 = self.wave_generate_high_layer1(wave_latent1).squeeze(dim=1)

        # reconstruction loss 2：wave_high2
        wave_low2 = wave_feature[1][0]
        wave_high2 = wave_feature[1][1]

        wave_feature2 = torch.cat((torch.unsqueeze(wave_feature[1][0], dim=1), torch.unsqueeze(wave_feature[1][1], dim=1)), dim=1)
        wave_feature2 = wave_feature2.transpose(2, 3)
        wave_feature2 = self.conv1_gcn_input2(wave_feature2)
        # gcn input feature：wave_feature2

        wave_latent2 = self.wave_gcn_layer2(wave_feature2, adj2)

        # latent_representation2：wave_latent2
        generated_high2 = self.wave_generate_high_layer2(wave_latent2).squeeze(dim=1)

        # reconstruction loss 3：wave_low3，wave_high3
        wave_low3 = wave_feature[2][0]
        wave_high3 = wave_feature[2][1]

        wave_feature3 = torch.cat((torch.unsqueeze(wave_feature[2][0], dim=1), torch.unsqueeze(wave_feature[2][1], dim=1)), dim=1)
        wave_feature3 = wave_feature3.transpose(2, 3)
        wave_feature3 = self.conv1_gcn_input3(wave_feature3)
        # gcn input feature：wave_feature3

        wave_latent3 = self.wave_gcn_layer3(wave_feature3, adj3)

        if latent:
            return [wave_latent1, wave_latent2, wave_latent3]

        # latent_representation3：wave_latent3
        generated_low3 = self.wave_generate_low_layer3(wave_latent3).squeeze(dim=1)
        generated_high3 = self.wave_generate_high_layer3(wave_latent3).squeeze(dim=1)

        generated_low2 = self.idwt_layer_3(generated_low3, [generated_high3])
        generated_low1 = self.idwt_layer_2(generated_low2, [generated_high2])
        generated_recons = self.idwt_layer_1(generated_low1, [generated_high1])

        generated_loss_1 = (wave_low3, generated_low3.transpose(1, 2))
        generated_loss_2 = (wave_high3, generated_high3.transpose(1, 2))
        generated_loss_3 = (wave_high2, generated_high2.transpose(1, 2))
        generated_loss_4 = (wave_high1, generated_high1.transpose(1, 2))
        generated_loss_5 = (x, generated_recons.transpose(1, 2))

        return [generated_loss_1, generated_loss_2, generated_loss_3, generated_loss_4, generated_loss_5]
