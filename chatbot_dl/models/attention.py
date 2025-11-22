"""
Bahdanau Attention Mechanism for Seq2Seq Models

This module implements the Bahdanau (additive) attention mechanism, which allows
the decoder to focus on different parts of the input sequence at each decoding step.

Deep Learning Concepts:
- Attention scores: Learnable alignment between encoder outputs and decoder state
- Softmax normalization: Converts raw scores to probability distribution
- Context vector: Weighted sum of encoder outputs based on attention weights
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BahdanauAttention(layers.Layer):
    """
    Bahdanau Attention Layer
    
    This implements additive attention (Bahdanau et al., 2015) where:
    1. Query (decoder hidden state) and keys (encoder outputs) are combined
    2. Attention scores are computed via a feedforward network
    3. Softmax creates attention distribution (which encoder positions to focus on)
    4. Context vector is created as weighted sum of encoder outputs
    
    Forward Propagation Flow:
    query (decoder_hidden) -> [Dense layer] -> combined with keys (encoder_output)
    -> [tanh activation] -> [Dense layer] -> attention scores
    -> [softmax] -> attention weights -> [weighted sum] -> context vector
    """
    
    def __init__(self, units):
        """
        Initialize attention layer with learnable parameters
        
        Args:
            units: Dimensionality of the attention mechanism's hidden representation
                   (controls the capacity of the alignment model)
        
        Deep Learning Note:
        - W1 and W2 are learnable weight matrices updated via backpropagation
        - These weights learn to align query and key vectors in a shared space
        """
        super(BahdanauAttention, self).__init__()
        self.units = units
        
        # Dense layer to transform encoder outputs (keys/values)
        # Parameters updated via gradient descent during backpropagation
        self.W1 = layers.Dense(units)
        
        # Dense layer to transform decoder hidden state (query)
        # Creates query representation in same space as transformed keys
        self.W2 = layers.Dense(units)
        
        # Final dense layer to compute scalar attention scores
        # Output dimension = 1 (single score per encoder timestep)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        """
        Forward propagation through attention mechanism
        
        Args:
            query: Decoder hidden state at current timestep, shape (batch_size, hidden_dim)
                   This is the "question" - what should we focus on?
            values: Encoder outputs for all timesteps, shape (batch_size, max_length, hidden_dim)
                    These are the "keys and values" - what to attend to
        
        Returns:
            context_vector: Weighted sum of encoder outputs, shape (batch_size, hidden_dim)
            attention_weights: Probability distribution over input positions, 
                             shape (batch_size, max_length, 1)
        
        Deep Learning Process:
        1. Transform query and values into alignment space (learned transformation)
        2. Compute alignment scores using tanh nonlinearity (additive attention)
        3. Apply softmax to get probability distribution (attention weights)
        4. Compute weighted sum (context vector) via matrix multiplication
        """
        
        # Expand query dimensions for broadcasting: (batch_size, 1, units)
        # This allows element-wise addition with encoder outputs at each timestep
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Additive attention score computation:
        # score = V * tanh(W1(values) + W2(query))
        # 
        # Forward propagation steps:
        # 1. W1(values): Transform encoder outputs -> (batch_size, max_length, units)
        # 2. W2(query): Transform decoder state -> (batch_size, 1, units)
        # 3. Add transformed representations (broadcast addition)
        # 4. Apply tanh nonlinearity (introduces non-linear decision boundary)
        # 5. V projects to scalar scores -> (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)
        ))
        
        # Softmax normalization over encoder timesteps (axis=1)
        # Converts raw scores to probability distribution that sums to 1
        # This ensures stable gradients during backpropagation
        # Output shape: (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Context vector: Weighted sum of encoder outputs
        # Matrix multiplication: attention_weights^T * values
        # This aggregates information from input sequence based on learned alignment
        # Shape: (batch_size, hidden_dim)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
