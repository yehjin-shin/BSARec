import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, TransformerEncoder

"""
[Paper]
Author: Fei Sun et al.
Title: "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
Conference: CIKM 2019

[Code Reference]
https://github.com/FeiSun/BERT4Rec
https://github.com/RUCAIBox/RecBole
"""

class BERT4RecModel(SequentialRecModel):
    def __init__(self, args):
        super(BERT4RecModel, self).__init__(args)

        # load parameters info
        self.mask_ratio = args.mask_ratio
        self.max_seq_length = args.max_seq_length
        self.item_embeddings = nn.Embedding(args.item_size+1, args.hidden_size, padding_idx=0)

        # load dataset info
        self.mask_token = args.item_size
        self.n_items = args.item_size
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_encoder = TransformerEncoder(args)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # parameters initialization
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_bi_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
                                                
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(
            masked_index.size(0), max_length, device=masked_index.device
        )
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot
    
    # input_ids 256, 50
    # answers 256
    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):

        masked_index = torch.rand(input_ids.size(0)).to(input_ids.device)
        mask_valid = (input_ids > 0).sum(axis=1)
        
        mask_num =  int(input_ids.size(1) * self.mask_ratio)
        masked_index = torch.zeros(input_ids.size(0), mask_num).to(input_ids.device)

        for i in range(input_ids.size(0)):
            masked_index[i, :] = torch.multinomial(torch.ones(input_ids.size(1)), mask_num, replacement=False)
        
        masked_index = masked_index.long()

        pos_items = input_ids.gather(dim=1, index=masked_index)
        for i in range(input_ids.size(0)):
            input_ids[i, masked_index[i]] = self.mask_token

        seq_output = self.forward(input_ids)
        pred_index_map = self.multi_hot_embed(masked_index, input_ids.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        test_item_emb = self.item_embeddings.weight[: self.n_items]  # [item_num H]
        logits = torch.matmul(
            seq_output, test_item_emb.transpose(0, 1)
        )  # [B mask_len item_num]
        targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

        loss = torch.sum(
            loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1))
            * targets
        ) / torch.sum(targets)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss


    def predict(self, input_ids, user_ids, all_sequence_output=False):
        item_seq = self.reconstruct_test_data(input_ids)
        seq_output = self.forward(item_seq)

        return seq_output

    def reconstruct_test_data(self, item_seq):
        """
        Add mask token at the last position according to the lengths of item_seq
        """

        padding = self.mask_token * torch.ones(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        item_seq = item_seq[:, 1:]

        return item_seq