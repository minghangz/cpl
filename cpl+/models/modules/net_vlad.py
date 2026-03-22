"""NetVLAD implementation
"""
# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    def __init__(self, cluster_size: object, feature_size: object, add_batch_norm: object = True) -> object:
        super().__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        init_sc = (1 / math.sqrt(feature_size))
        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, cluster_size))
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(cluster_size)
        self.out_dim = cluster_size * feature_size

    def reset_parameters(self):
        # init_sc = (1 / math.sqrt(self.feature_size))
        # The `clusters` weights are the `(w,b)` in the paper
        # self.clusters = nn.Parameter(init_sc * th.randn(self.feature_size, self.cluster_size))
        # # The `clusters2` weights are the visual words `c_k` in the paper
        # self.clusters2 = nn.Parameter(init_sc * th.randn(1, self.feature_size, self.cluster_size))
        self.batch_norm.reset_parameters()

    def forward(self, x, x_mask=None, flatten=True):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D
            x_mask: B x N
            flatten: boolean
        Returns:
            (th.Tensor): B x DK
        """
        # self.sanity_checks(x)
        max_sample = x.size()[1]
        # print(x.size(), self.feature_size)
        x = x.contiguous().view(-1, self.feature_size)  # B x N x D -> BN x D
        # if x.device != self.clusters.device:
        #      import ipdb; ipdb.set_trace()
        assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x K) -> BN x K

        if self.add_batch_norm:
            assignment = self.batch_norm(assignment)

        # if x_mask is not None:
        #     assignment = assignment.masked_fill(x_mask.unsqueeze(-1) == 0, -1e30)
        assignment = F.softmax(assignment, dim=1)  # BN x K -> BN x K
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        if x_mask is not None:
            assignment = assignment.masked_fill(x_mask.unsqueeze(-1) == 0, 0)
        # assert not th.isnan(assignment).any()
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2  # B x D x K

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        if flatten:
            vlad = vlad.view(-1, self.cluster_size * self.feature_size)  # -> B x DK
            vlad = F.normalize(vlad)
        else:
            # vlad = vlad.transpose(-1, -2) / x_mask.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).float()
            vlad = vlad.transpose(-1, -2)
        return vlad  # B x DK or B x K x D

    # def sanity_checks(self, x):
    #     """Catch any nans in the inputs/clusters"""
    #     if th.isnan(th.sum(x)):
    #         print("nan inputs")
    #         ipdb.set_trace()
    #     if th.isnan(self.clusters[0][0]):
    #         print("nan clusters")
    #         ipdb.set_trace()
