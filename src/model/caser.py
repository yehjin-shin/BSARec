import torch
import torch.nn as nn
from torch.nn import functional as F
from model._abstract_model import SequentialRecModel

"""
[Paper]
Author: Jiaxi Tang et al.
Title: "Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding"
Conference: WSDM 2018

[Code Reference]
https://github.com/graytowne/caser_pytorch
https://github.com/RUCAIBox/RecBole
"""

class CaserModel(SequentialRecModel):
    def __init__(self, args):
        super(CaserModel, self).__init__(args)

        self.args = args
        self.user_embeddings = nn.Embedding(args.num_users, args.hidden_size, padding_idx=0)

        # load parameters info
        self.embedding_size = args.hidden_size
        self.n_h = args.nh
        self.n_v = args.nv
        self.reg_weight = args.reg_weight
        self.dropout_prob = args.hidden_dropout_prob
        self.max_seq_length = args.max_seq_length

        # load dataset info
        self.n_users = args.num_users

        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_h = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=self.n_h,
                    kernel_size=(i, self.embedding_size),
                )
                for i in lengths
            ]
        )

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(
            self.embedding_size + self.embedding_size, self.embedding_size
        )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()

        self.apply(self.init_weights)

    def reg_loss(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss

    def reg_loss_conv_h(self):
        r"""
        L2 loss on conv_h
        """
        loss_conv_h = 0
        for name, parm in self.conv_h.named_parameters():
            if name.endswith("weight"):
                loss_conv_h = loss_conv_h + parm.norm(2)
        return self.reg_weight * loss_conv_h


    def forward(self, input_ids, user_ids, all_sequence_output=False):
        # Embedding Look-up
        # use unsqueeze() to get a 4-D input for convolution layers. (batch_size * 1 * max_length * embedding_size)
        item_seq_emb = self.item_embeddings(input_ids).unsqueeze(1)
        user_emb = self.user_embeddings(user_ids).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_seq_emb)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = torch.cat([z, user_emb], 1)
        seq_output = self.ac_fc(self.fc2(x))

        # the hidden_state of the predicted item, size:(batch_size * hidden_size)
        return seq_output.unsqueeze(1)


    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):

        seq_out = self.forward(input_ids, user_ids)
        seq_out = seq_out[:, -1, :]
        pos_ids, neg_ids = answers, neg_answers

        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)

        # [batch hidden_size]
        seq_emb = seq_out # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)

        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=seq_out.device), torch.zeros(neg_logits.shape, device=seq_out.device)
        indices = (pos_ids != 0).nonzero().reshape(-1)
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])

        reg_loss = self.reg_loss(
            [
                self.user_embeddings.weight,
                self.item_embeddings.weight,
                self.conv_v.weight,
                self.fc1.weight,
                self.fc2.weight,
            ]
        )
        loss = loss + self.reg_weight * reg_loss + self.reg_loss_conv_h()

        return loss

