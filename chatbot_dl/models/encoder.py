"""
LSTM Encoder for Sequence-to-Sequence Models

This module implements the encoder component that processes input sequences
and creates a latent representation (hidden state) for the decoder.

Deep Learning Concepts:
- LSTM (Long Short-Term Memory): Recurrent architecture that maintains long-term dependencies
- Hidden state: Learned representation of the input sequence in latent space
- Embedding layer: Maps discrete tokens to continuous vector space
- Forward propagation: Sequential processing of input tokens
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Encoder(keras.Model):
    """
    LSTM-based Encoder
    
    Architecture:
    Input tokens -> Embedding layer -> LSTM -> (outputs, hidden_state, cell_state)
    
    The encoder performs forward propagation to:
    1. Convert input tokens to dense embeddings (continuous representation)
    2. Process sequence through LSTM (captures sequential dependencies)
    3. Output hidden state encoding of entire input sequence
    
    This hidden state serves as the "thought" or latent representation that
    the decoder will use to generate the response.
    """
    
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        """
        Initialize encoder architecture with learnable parameters
        
        Args:
            vocab_size: Size of input vocabulary (number of unique tokens)
            embedding_dim: Dimension of embedding vectors (word representation size)
            enc_units: Number of LSTM units (hidden state dimension)
            batch_sz: Batch size for training (affects initialization)
        
        Deep Learning Note:
        - Embedding weights are learned during backpropagation
        - LSTM has 4 gates (input, forget, output, cell) with separate weights
        - Total parameters: vocab_size * embedding_dim + 4 * (enc_units * (enc_units + embedding_dim))
        """
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        
        # Embedding layer: Maps token IDs to dense vectors
        # This is a learnable lookup table updated via gradient descent
        # Shape: (vocab_size, embedding_dim)
        # Each row is a vector representation of a word in latent space
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer: Processes sequences while maintaining internal memory
        # return_sequences=True: Output hidden state at each timestep
        # return_state=True: Also return final hidden and cell states
        # 
        # LSTM equations (forward propagation at each timestep t):
        # f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)  # Forget gate
        # i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)  # Input gate
        # C_tilde = tanh(W_C * [h_{t-1}, x_t] + b_C) # Candidate cell state
        # C_t = f_t * C_{t-1} + i_t * C_tilde        # Cell state update
        # o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)  # Output gate
        # h_t = o_t * tanh(C_t)                      # Hidden state update
        self.lstm = layers.LSTM(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
    
    def call(self, x, hidden):
        """
        Forward propagation through encoder
        
        Args:
            x: Input token sequences, shape (batch_size, sequence_length)
               These are integer indices into the vocabulary
            hidden: Initial hidden state (typically zeros), shape (batch_size, enc_units)
        
        Returns:
            output: LSTM outputs at all timesteps, shape (batch_size, sequence_length, enc_units)
                    These are the "keys" and "values" for attention mechanism
            state_h: Final hidden state, shape (batch_size, enc_units)
                     This encodes the entire input sequence in latent space
            state_c: Final cell state, shape (batch_size, enc_units)
                     LSTM's long-term memory component
        
        Deep Learning Process:
        1. Embedding lookup: Convert discrete tokens to continuous vectors
        2. LSTM forward pass: Sequential processing with recurrent connections
        3. Return all outputs (for attention) and final states (for decoder initialization)
        """
        
        # Embedding lookup: x (batch, seq_len) -> (batch, seq_len, embedding_dim)
        # This is a differentiable operation - gradients flow back to update embeddings
        x = self.embedding(x)
        
        # LSTM forward propagation:
        # Processes sequence step-by-step, maintaining hidden state h and cell state c
        # Gradients during backpropagation flow through time (BPTT)
        # output: All hidden states (for attention to select from)
        # state_h: Final hidden state (initialization for decoder)
        # state_c: Final cell state (LSTM memory, also passed to decoder)
        output, state_h, state_c = self.lstm(x, initial_state=[hidden, hidden])
        
        return output, state_h, state_c
    
    def initialize_hidden_state(self):
        """
        Initialize hidden state with zeros
        
        Returns:
            Zero tensor of shape (batch_size, enc_units)
        
        Deep Learning Note:
        Starting with zeros is common practice. The network learns appropriate
        representations through gradient descent during training.
        """
        return tf.zeros((self.batch_sz, self.enc_units))
