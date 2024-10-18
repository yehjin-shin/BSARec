import math
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward

"""
[Paper]
Author: Xinyu Du et al.
Title: "Frequency Enhanced Hybrid Attention Network for Sequential Recommendation."
Conference: SIGIR 2023

[Code Reference]
https://github.com/sudaada/FEARec
"""

class FEARecModel(SequentialRecModel):
    def __init__(self, args):
        super(FEARecModel, self).__init__(args)
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = FEARecEncoder(args)
        self.batch_size = args.batch_size
        self.gamma = 1e-10

        # arguments for FEARec
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.tau = args.tau
        self.fredom = eval(args.fredom)
        self.fredom_type = args.fredom_type
        self.ssl = args.ssl
        self.sim = args.sim
        self.lmd_sem = args.lmd_sem
        self.lmd = args.lmd

        self.apply(self.init_weights)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        z = z[:, -1, :]
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
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

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        seq_output = self.forward(input_ids)
        seq_output = seq_output[:, -1, :]

        # cross-entropy loss
        test_item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        # Unsupervised NCE: original vs dropout
        if self.ssl in ['us', 'un']:
            aug_seq_output = self.forward(input_ids)
            nce_logits, nce_labels = self.info_nce(seq_output, aug_seq_output, temp=self.tau,
                                                batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        # Supervised NCE: original vs semantic augmentation
        if self.ssl in ['us', 'su']:
            sem_aug = same_target
            sem_aug_seq_output = self.forward(sem_aug)
            sem_nce_logits, sem_nce_labels = self.info_nce(seq_output, sem_aug_seq_output, temp=self.tau,
                                                        batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
        
        # Unsupervised + Supervised NCE: dropout vs semantic augmentation
        if self.ssl == 'us_x':
            # unsupervised
            aug_seq_output = self.forward(input_ids)
            # supervised
            sem_aug = same_target
            sem_aug_seq_output = self.forward(sem_aug)

            sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
                                                        batch_size=input_ids.shape[0], sim=self.sim)

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)
            
        if self.fredom:
            seq_output_f = torch.fft.rfft(seq_output, dim=1, norm='ortho')
            aug_seq_output_f = torch.fft.rfft(aug_seq_output, dim=1, norm='ortho')
            sem_aug_seq_output_f = torch.fft.rfft(sem_aug_seq_output, dim=1, norm='ortho')

            if self.fredom_type in ['us', 'un']:
                loss += 0.1 * abs(seq_output_f - aug_seq_output_f).flatten().mean()

            if self.fredom_type in ['us', 'su']:
                loss += 0.1 * abs(seq_output_f - sem_aug_seq_output_f).flatten().mean()
                
            if self.fredom_type == 'us_x':
                loss += 0.1 * abs(aug_seq_output_f - sem_aug_seq_output_f).flatten().mean()

        return loss

class FEARecEncoder(nn.Module):
    def __init__(self, args):
        super(FEARecEncoder, self).__init__()
        self.args = args

        self.blocks = []
        for i in range(args.num_hidden_layers):
            self.blocks.append(FEARecBlock(args, layer_num=i))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):

        all_encoder_layers = [ hidden_states ]

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])

        return all_encoder_layers

class FEARecBlock(nn.Module):
    def __init__(self, args, layer_num):
        super(FEARecBlock, self).__init__()
        self.layer = FEARecLayer(args, layer_num)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class FEARecLayer(nn.Module):
    def __init__(self, args, fea_layer=0):
        super(FEARecLayer, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12) # layernorm implemented in fmlp
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.max_item_list_length = args.max_seq_length
        self.dual_domain = True

        self.global_ratio = args.global_ratio
        self.n_layers = args.num_hidden_layers

        self.scale = None
        self.mask_flag = True
        self.output_attention = False

        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.factor = 10 # config['topk_factor']

        self.filter_mixer = None
        if self.global_ratio > (1 / self.n_layers):
            print("{}>{}:{}".format(self.global_ratio, 1 / self.n_layers, self.global_ratio > (1 / self.n_layers)))
            self.filter_mixer = 'G'
        else:
            print("{}>{}:{}".format(self.global_ratio, 1 / self.n_layers, self.global_ratio > (1 / self.n_layers)))
            self.filter_mixer = 'L'
        self.slide_step = ((self.max_item_list_length // 2 + 1) * (1 - self.global_ratio)) // (self.n_layers - 1)
        self.local_ratio = 1 / self.n_layers
        self.filter_size = self.local_ratio * (self.max_item_list_length // 2 + 1)

        if self.filter_mixer == 'G':
            self.w = self.global_ratio
            self.s = self.slide_step

        if self.filter_mixer == 'L':
            self.w = self.local_ratio
            self.s = self.filter_size

        i = fea_layer
        self.left = int(((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (i * self.s))
        self.right = int((self.max_item_list_length // 2 + 1) - i * self.s)

        self.q_index = list(range(self.left, self.right))
        self.k_index = list(range(self.left, self.right))
        self.v_index = list(range(self.left, self.right))
        # if sample in time domain
        self.std = True # config['std']
        if self.std:
            self.time_q_index = self.q_index
            self.time_k_index = self.k_index
            self.time_v_index = self.v_index
        else:
            self.time_q_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_k_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_v_index = list(range(self.max_item_list_length // 2 + 1))

        print('modes_q={}, index_q={}'.format(len(self.q_index), self.q_index))
        print('modes_k={}, index_k={}'.format(len(self.k_index), self.k_index))
        print('modes_v={}, index_v={}'.format(len(self.v_index), self.v_index))

        self.spatial_ratio = args.spatial_ratio

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0) \
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [256, 50, 2, 32]
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        queries = self.transpose_for_scores(mixed_query_layer)
        keys = self.transpose_for_scores(mixed_key_layer)
        values = self.transpose_for_scores(mixed_value_layer)

        # B, H, L, E = query_layer.shape
        # AutoFormer
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)

        # Put it in an empty box.
        q_fft_box = torch.zeros(B, H, E, len(self.q_index), device=q_fft.device, dtype=torch.cfloat)

        for i, j in enumerate(self.q_index):
            q_fft_box[:, :, :, i] = q_fft[:, :, :, j]

        k_fft_box = torch.zeros(B, H, E, len(self.k_index), device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.q_index):
            k_fft_box[:, :, :, i] = k_fft[:, :, :, j]

        res = q_fft_box * torch.conj(k_fft_box)
        box_res = torch.zeros(B, H, E, L // 2 + 1,  device=q_fft.device, dtype=torch.cfloat)
        for i, j in enumerate(self.q_index):
            box_res[:, :, :, j] = res[:, :, :, i]

        corr = torch.fft.irfft(box_res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        new_context_layer_shape = V.size()[:-2] + (self.all_head_size,)
        context_layer = V.view(*new_context_layer_shape)

        if self.dual_domain:
            # Put it in an empty box.
            # q
            q_fft_box = torch.zeros(B, H, E, len(self.time_q_index), device=q_fft.device, dtype=torch.cfloat)

            for i, j in enumerate(self.time_q_index):
                q_fft_box[:, :, :, i] = q_fft[:, :, :, j]
            spatial_q = torch.zeros(B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat)
            for i, j in enumerate(self.time_q_index):
                spatial_q[:, :, :, j] = q_fft_box[:, :, :, i]

            # k
            k_fft_box = torch.zeros(B, H, E, len(self.time_k_index), device=q_fft.device, dtype=torch.cfloat)
            for i, j in enumerate(self.time_k_index):
                k_fft_box[:, :, :, i] = k_fft[:, :, :, j]
            spatial_k = torch.zeros(B, H, E, L // 2 + 1, device=k_fft.device, dtype=torch.cfloat)
            for i, j in enumerate(self.time_k_index):
                spatial_k[:, :, :, j] = k_fft_box[:, :, :, i]

            # v
            v_fft = torch.fft.rfft(values.permute(0, 2, 3, 1).contiguous(), dim=-1)
            # Put it in an empty box.
            v_fft_box = torch.zeros(B, H, E, len(self.time_v_index), device=v_fft.device, dtype=torch.cfloat)

            for i, j in enumerate(self.time_v_index):
                v_fft_box[:, :, :, i] = v_fft[:, :, :, j]
            spatial_v = torch.zeros(B, H, E, L // 2 + 1, device=v_fft.device, dtype=torch.cfloat)
            for i, j in enumerate(self.time_v_index):
                spatial_v[:, :, :, j] = v_fft_box[:, :, :, i]

            queries = torch.fft.irfft(spatial_q, dim=-1)
            keys = torch.fft.irfft(spatial_k, dim=-1)
            values = torch.fft.irfft(spatial_v, dim=-1)

            queries = queries.permute(0, 1, 3, 2)
            keys = keys.permute(0, 1, 3, 2)
            values = values.permute(0, 1, 3, 2)

            attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)
            qkv = torch.matmul(attention_probs, values)  # [256, 2, index, 32]
            context_layer_spatial = qkv.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer_spatial.size()[:-2] + (self.all_head_size,)
            context_layer_spatial = context_layer_spatial.view(*new_context_layer_shape)
            context_layer = (1 - self.spatial_ratio) * context_layer + self.spatial_ratio * context_layer_spatial

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states