"""
LSTM Decoder with Bahdanau Attention

This module implements the decoder component that generates output sequences
using attention mechanism to focus on relevant parts of the input.

Deep Learning Concepts:
- Attention-based decoding: Uses context vector from encoder outputs
- Teacher forcing: Training technique using ground truth as input
- Autoregressive generation: Each prediction depends on previous outputs
- Backpropagation through time: Gradients flow through sequential decisions
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.attention import BahdanauAttention


class Decoder(keras.Model):
    """
    LSTM-based Decoder with Attention
    
    Architecture:
    Input token -> Embedding -> Concatenate with context vector 
    -> LSTM -> Dense layer -> Output probability distribution
    
    At each decoding step:
    1. Attention mechanism computes context vector from encoder outputs
    2. Current token embedding is concatenated with context
    3. LSTM processes this combined input
    4. Dense layer projects to vocabulary space
    5. Softmax gives probability distribution over next token
    
    During backpropagation:
    - Gradients flow from cross-entropy loss back through all layers
    - LSTM gradients propagate through time (BPTT)
    - Attention weights are updated to improve alignment
    """
    
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        """
        Initialize decoder architecture with learnable parameters
        
        Args:
            vocab_size: Size of output vocabulary
            embedding_dim: Dimension of embedding vectors
            dec_units: Number of LSTM units (should match encoder for state transfer)
            batch_sz: Batch size for training
        
        Deep Learning Note:
        - All weight matrices are randomly initialized and learned via gradient descent
        - Adam optimizer adapts learning rates per parameter during training
        """
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        
        # Embedding layer: Maps output tokens to dense vectors
        # Separate from encoder embeddings (different vocabularies possible)
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer: Processes embedding + context at each timestep
        # return_sequences=True: Output at every step (for multi-step decoding)
        # return_state=True: Track hidden state across decoding steps
        self.lstm = layers.LSTM(
            dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        
        # Output projection layer: Maps LSTM output to vocabulary space
        # Shape: (dec_units) -> (vocab_size)
        # Final layer before softmax - produces logits for each token
        self.fc = layers.Dense(vocab_size)
        
        # Attention mechanism: Computes context vector from encoder outputs
        # This is where the "alignment" between input and output is learned
        self.attention = BahdanauAttention(self.dec_units)
    
    def call(self, x, hidden, cell, enc_output):
        """
        Forward propagation through decoder for one timestep
        
        Args:
            x: Current input token, shape (batch_size, 1)
               During training: ground truth token (teacher forcing)
               During inference: previously generated token
            hidden: Previous hidden state, shape (batch_size, dec_units)
            cell: Previous cell state, shape (batch_size, dec_units)
            enc_output: All encoder outputs, shape (batch_size, max_length, enc_units)
                       Used by attention to create context vector
        
        Returns:
            predictions: Logits over vocabulary, shape (batch_size, vocab_size)
                        Apply softmax to get probability distribution
            state_h: Updated hidden state
            state_c: Updated cell state
            attention_weights: Where the model focused, shape (batch_size, max_length, 1)
        
        Deep Learning Process:
        1. Attention: Compute context vector (weighted sum of encoder outputs)
        2. Embedding: Convert token ID to dense vector
        3. Concatenation: Combine embedding with context (information fusion)
        4. LSTM: Process combined input (sequential modeling)
        5. Dense: Project to vocabulary space (output generation)
        
        Gradient Flow (Backpropagation):
        Loss -> fc -> LSTM -> [embedding, attention] -> encoder
        """
        
        # Attention mechanism forward pass:
        # Uses current hidden state (query) and encoder outputs (keys/values)
        # to compute context vector (what to focus on from input)
        # context_vector shape: (batch_size, enc_units)
        # attention_weights shape: (batch_size, max_length, 1) - alignment scores
        context_vector, attention_weights = self.attention(hidden, enc_output)
        
        # Embedding lookup: Convert token ID to dense representation
        # x shape: (batch_size, 1) -> (batch_size, 1, embedding_dim)
        # Gradients update embeddings during backpropagation
        x = self.embedding(x)
        
        # Concatenate embedding with context vector
        # This allows LSTM to use both:
        # - Current token information (embedding)
        # - Relevant input information (context from attention)
        # Shape: (batch_size, 1, embedding_dim + enc_units)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # LSTM forward propagation:
        # Processes concatenated input while maintaining internal state
        # output: LSTM hidden state at this timestep
        # state_h, state_c: Updated hidden and cell states (carry forward to next step)
        output, state_h, state_c = self.lstm(x, initial_state=[hidden, cell])
        
        # Reshape output: (batch_size, 1, dec_units) -> (batch_size, dec_units)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # Dense layer: Project to vocabulary space
        # Output shape: (batch_size, vocab_size)
        # These are logits (pre-softmax scores) for each possible next token
        # During training, cross-entropy loss is computed against ground truth
        # Gradients from loss flow back through this layer
        predictions = self.fc(output)
        
        return predictions, state_h, state_c, attention_weights
