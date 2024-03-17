import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()

        hidden_size = args.hidden_size
        inner_size = 4 * args.hidden_size

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(args.hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


#######################
## Basic Transformer ##
#######################

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.args = args
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12) # TODO
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super(TransformerBlock, self).__init__()
        self.layer = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        block = TransformerBlock(args) # self attention

        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):

        all_encoder_layers = [ hidden_states ]

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])

        return all_encoder_layers


#######################
######  BSARec  #######
#######################

class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.c = args.c // 2 + 1
        self.beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states
    
class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


#######################
######  FMLP-Rec ######
#######################

class FMLPRecLayer(nn.Module):
    def __init__(self, args):
        super(FMLPRecLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor

        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

class FMLPRecBlock(nn.Module):
    def __init__(self, args):
        super(FMLPRecBlock, self).__init__()
        self.layer = FMLPRecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states):
        layer_output = self.layer(hidden_states)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


#######################
######  FEARec  #######
#######################

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

class FEARecBlock(nn.Module):
    def __init__(self, args, layer_num):
        super(FEARecBlock, self).__init__()
        self.layer = FEARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output