import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = F.gelu

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.layernorm1(attention_output + hidden_states)

        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(layer_output + attention_output)
        
        return layer_output

class LunarConfig:
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        initializer_range=0.02,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range

class LunarV2Lite(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
    def get_position_embeddings(self, position_ids):
        seq_length = position_ids.size(1)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids)
        position_embeddings = self.get_position_embeddings(position_ids)
        
        hidden_states = embedding_output + position_embeddings
        hidden_states = self.dropout(hidden_states)

        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)

        logits = self.head(hidden_states)
        return logits

    def generate(self, input_ids, max_length, temperature=1.0, top_k=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0)
                
                outputs = self(input_ids, position_ids=position_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[0, torch.multinomial(probs[0], 1)]
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
        return input_ids
