import tensorflow as tf
import pickle
from utils.preprocessing import clean_text
from models.encoder import Encoder
from models.decoder import Decoder

EMBEDDING_DIM = 256
UNITS = 512
MAX_LEN = 20
CHECKPOINT_DIR = "checkpoints"

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
encoder = Encoder(vocab_size, EMBEDDING_DIM, UNITS)
decoder = Decoder(vocab_size, EMBEDDING_DIM, UNITS)
optimizer = tf.keras.optimizers.Adam()

checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

if checkpoint_manager.latest_checkpoint:
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
    status.expect_partial()
   
while running chat.py bot responing nothing 