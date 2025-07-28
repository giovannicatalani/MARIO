import torch
import torch.nn as nn


def position(position, features, layers, activation):
    """Use the position as input (i.e. no conditioning)
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = position

    for i, l in enumerate(layers):
        h = activation(l(h))

    return h


def feature(position, features, layers, activation):
    """Use the features as input.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = features

    for i, l in enumerate(layers):
        h = activation(l(h))

    return h


def concat(position, features, layers, activation):
    """Concatenates the input onto the features, and then feeds into the input of the neural network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """

    h = torch.cat([position, features], dim=-1)

    for i, l in enumerate(layers):
        h = activation(l(h))
    return h



def shift_modulation(position, features, layers, activation, with_batch=True):
    """Applies film conditioning (add only) on the network.
    Args:
        position   : [N, ..., d] tensor of coordinates
        features   : [N, ..., f] tensor of features
        layers     : nn.ModuleList of layers
        activation : activation function
    """
    feature_shape = features.shape[0]  # features.shape[:-1]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)
    # Maybe add assertion here... but if it errors, your feature_dim size is wrong

    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position

    for i, l in enumerate(layers):
        res = l(h)
        # Maybe also add another assertion here
        h = res * features[..., i, :] + features[..., i, :] + res
        h = activation(h)
    return h

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModulation(nn.Module):
    def __init__(self, d_query, d_key, d_value, embedding_size=8, n_heads=1, dropout=0.1):
        """
        Cross-attention modulation layer.
        Args:
            d_query : Dimension of query vectors (usually the position embedding size)
            d_key   : Dimension of key vectors (feature embedding size)
            d_value : Dimension of value vectors (same as key dimension)
            embedding_size : The size of the embeddings used for attention
            n_heads : Number of attention heads
            dropout : Dropout rate
        """
        super(CrossAttentionModulation, self).__init__()
        self.n_heads = n_heads
        self.embedding_size = embedding_size  # Set the embedding size
        self.seq_len_query = d_query // self.embedding_size  # Sequence length for queries
        self.seq_len_features = d_value // self.embedding_size  # Sequence length for features

        # Define projections for queries, keys, and values
        self.query_projection = nn.Linear(self.embedding_size, self.embedding_size)
        self.key_projection = nn.Linear(self.embedding_size, self.embedding_size)
        self.value_projection = nn.Linear(self.embedding_size, self.embedding_size)

        # Multi-head attention layer, working on embed_dim = embedding_size
        self.attention = nn.MultiheadAttention(embed_dim=self.embedding_size, num_heads=n_heads, dropout=dropout)

        # Output projection back to original d_query dimensions
        self.out_projection = nn.Linear(d_value, d_query)

    def forward(self, position, features):
        """
        Applies cross-attention modulation to the position tensor using features as keys and values.
        Args:
            position : [batch_size, 1, d_query] tensor of positions (queries)
            features : [batch_size, 1, d_key] tensor of features (keys and values)
        Returns:
            modulated_position : [batch_size, 1, d_query] tensor of position modulated by cross-attention
        """
        batch_size = position.shape[0]

        # Reshape position and features to have seq_len and embedding_size before projection
        queries = position.view(batch_size, self.seq_len_query, self.embedding_size)  # [batch_size, seq_len_query, embedding_size]
        keys = features.view(batch_size, self.seq_len_features, self.embedding_size)  # [batch_size, seq_len_features, embedding_size]
        values = features.view(batch_size, self.seq_len_features, self.embedding_size)  # [batch_size, seq_len_features, embedding_size]

        # Permute to [seq_len, batch_size, embedding_size] for multi-head attention
        queries = queries.permute(1, 0, 2)  # [seq_len_query, batch_size, embedding_size]
        keys = keys.permute(1, 0, 2)        # [seq_len_features, batch_size, embedding_size]
        values = values.permute(1, 0, 2)    # [seq_len_features, batch_size, embedding_size]

        # Apply the projections on the embeddings
        queries = self.query_projection(queries)  # [seq_len_query, batch_size, embedding_size]
        keys = self.key_projection(keys)          # [seq_len_features, batch_size, embedding_size]
        values = self.value_projection(values)    # [seq_len_features, batch_size, embedding_size]

        # Apply cross-attention
        attn_output, _ = self.attention(queries, keys, values)  # [seq_len_query, batch_size, embedding_size]

        # Reshape the attention output back to [batch_size, 1, d_query]
        attn_output = attn_output.permute(1, 0, 2).reshape(batch_size, 1, -1)  # [batch_size, 1, d_query]

        # Apply final linear transformation to match the output dimension with d_query
        modulated_position = self.out_projection(attn_output)  # [batch_size, 1, d_query]

        return modulated_position


def cross_attention_conditioning(position, features, layers, activation, attention_layer, with_batch=True):
    """
    Applies cross-attention modulation on the network.
    Args:
        position        : [N, ..., d] tensor of coordinates (queries)
        features        : [N, ..., f] tensor of features (keys and values)
        layers          : nn.ModuleList of layers
        activation      : activation function
        attention_layer : cross-attention modulation layer
    """
    feature_shape = features.shape[0]
    feature_dim = features.shape[-1]
    num_hidden = len(layers)

    # Reshape the features for layer-wise usage, similar to shift_modulation
    if with_batch:
        features = features.reshape(feature_shape, 1, num_hidden, feature_dim // num_hidden)
    else:
        features = features.reshape(feature_shape, num_hidden, feature_dim // num_hidden)

    h = position

    for i, layer in enumerate(layers):
        # Extract the relevant slice of the features for the current layer
        layer_features = features[..., i, :]  # Features specific to layer i

        # Apply cross-attention using the current features slice as keys and values
        h = attention_layer(layer(h), layer_features)

    return h

