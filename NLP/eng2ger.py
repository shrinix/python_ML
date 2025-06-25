import io
import json
import numpy as np
import pandas as pd
import random
import re
import tensorflow as tf
import unicodedata

from google.colab import files
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download the training set.
#!wget https://raw.githubusercontent.com/futuremojo/nlp-demystified/main/datasets/hun_eng_pairs/hun_eng_pairs_train.txt

with open('hun_eng_pairs_train.txt') as file:
  train = [line.rstrip() for line in file]

train[:3]

print(len(train))

# Separate the input (Hungarian) and target (English) sentences into separate lists.
SEPARATOR = '<sep>'
train_input, train_target = map(list, zip(*[pair.split(SEPARATOR) for pair in train]))

print(train_input[:3])
print(train_target[:3])

print("\u00E1", "\u0061\u0301")

# Unicode normalization
def normalize_unicode(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(s):
  s = normalize_unicode(s)
  s = re.sub(r"([?.!,¿])", r" \1 ", s)
  s = re.sub(r'[" "]+', " ", s)
  s = s.strip()
  return s

# Preprocess both the source and target sentences.
train_preprocessed_input = [preprocess_sentence(s) for s in train_input]
train_preprocessed_target = [preprocess_sentence(s) for s in train_target]

train_preprocessed_input[:3]

def tag_target_sentences(sentences):
  tagged_sentences = map(lambda s: (' ').join(['<sos>', s, '<eos>']), sentences)
  return list(tagged_sentences)

train_tagged_preprocessed_target = tag_target_sentences(train_preprocessed_target)

train_tagged_preprocessed_target[:3]

# Tokenizer for the Hungarian input sentences. Note how we're not filtering punctuation.
source_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>', filters='"#$%&()*+-/:;=@[\\]^_`{|}~\t\n')
source_tokenizer.fit_on_texts(train_preprocessed_input)
source_tokenizer.get_config()

source_vocab_size = len(source_tokenizer.word_index) + 1
print(source_vocab_size)

# Tokenizer for the English target sentences.
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>', filters='"#$%&()*+-/:;=@[\\]^_`{|}~\t\n')
target_tokenizer.fit_on_texts(train_tagged_preprocessed_target)
target_tokenizer.get_config()

target_vocab_size = len(target_tokenizer.word_index) + 1
print(target_vocab_size)

train_encoder_inputs = source_tokenizer.texts_to_sequences(train_preprocessed_input)

print(train_encoder_inputs[:3])
print(source_tokenizer.sequences_to_texts(train_encoder_inputs[:3]))

def generate_decoder_inputs_targets(sentences, tokenizer):
  seqs = tokenizer.texts_to_sequences(sentences)
  decoder_inputs = [s[:-1] for s in seqs] # Drop the last token in the sentence.
  decoder_targets = [s[1:] for s in seqs] # Drop the first token in the sentence.

  return decoder_inputs, decoder_targets

train_decoder_inputs, train_decoder_targets = generate_decoder_inputs_targets(train_tagged_preprocessed_target, 
                                                                              target_tokenizer)

print(train_decoder_inputs[0], train_decoder_targets[0])

print(target_tokenizer.sequences_to_texts(train_decoder_inputs[:1]), 
      target_tokenizer.sequences_to_texts(train_decoder_targets[:1]))

max_encoding_len = len(max(train_encoder_inputs, key=len))
max_encoding_len

max_decoding_len = len(max(train_decoder_inputs, key=len))
max_decoding_len

padded_train_encoder_inputs = pad_sequences(train_encoder_inputs, max_encoding_len, padding='post', truncating='post')
padded_train_decoder_inputs = pad_sequences(train_decoder_inputs, max_decoding_len, padding='post', truncating='post')
padded_train_decoder_targets = pad_sequences(train_decoder_targets, max_decoding_len, padding='post', truncating='post')

print(padded_train_encoder_inputs[0])
print(padded_train_decoder_inputs[0])
print(padded_train_decoder_targets[0])

target_tokenizer.sequences_to_texts([padded_train_decoder_inputs[0]])

# Download validation set pairs.
#!wget https://raw.githubusercontent.com/futuremojo/nlp-demystified/main/datasets/hun_eng_pairs/hun_eng_pairs_val.txt

with open('hun_eng_pairs_val.txt') as file:
  val = [line.rstrip() for line in file]

def process_dataset(dataset):

  # Split the Hungarian and English sentences into separate lists.
  input, output = map(list, zip(*[pair.split(SEPARATOR) for pair in dataset]))

  # Unicode normalization and inserting spaces around punctuation.
  preprocessed_input = [preprocess_sentence(s) for s in input]
  preprocessed_output = [preprocess_sentence(s) for s in output]

  # Tag target sentences with <sos> and <eos> tokens.
  tagged_preprocessed_output = tag_target_sentences(preprocessed_output)

  # Vectorize encoder source sentences.
  encoder_inputs = source_tokenizer.texts_to_sequences(preprocessed_input)

  # Vectorize and create decoder input and target sentences.
  decoder_inputs, decoder_targets = generate_decoder_inputs_targets(tagged_preprocessed_output, 
                                                                    target_tokenizer)
  
  # Pad all collections.
  padded_encoder_inputs = pad_sequences(encoder_inputs, max_encoding_len, padding='post', truncating='post')
  padded_decoder_inputs = pad_sequences(decoder_inputs, max_decoding_len, padding='post', truncating='post')
  padded_decoder_targets = pad_sequences(decoder_targets, max_decoding_len, padding='post', truncating='post')

  return padded_encoder_inputs, padded_decoder_inputs, padded_decoder_targets

# Process validation dataset
padded_val_encoder_inputs, padded_val_decoder_inputs, padded_val_decoder_targets = process_dataset(val)

embedding_dim = 128
hidden_dim = 256
default_dropout=0.2
batch_size = 32
epochs = 30

# The initial encoder input layer which will take in padded sequences. We're specifying
# a shape of None here but you can specify it upfront if you want since we
# know what the max encoding length is.
encoder_inputs = layers.Input(shape=[None], name='encoder_inputs')

# The embedding layer. Similar to what we did in the RNN demo.
encoder_embeddings = layers.Embedding(source_vocab_size, 
                                      embedding_dim,
                                      mask_zero=True,
                                      name='encoder_embeddings')

# Passing the input layer output to the embedding layer creates a link between the
# two. Input sequences will now flow into the embedding layer which will output
# a sequence of embeddings.
encoder_embedding_output = encoder_embeddings(encoder_inputs)


# We're not using any kind of attention mechanism in this model, so setting only
# return_state to True is enough. return_sequences remains False.
encoder_lstm = layers.LSTM(hidden_dim, 
                           return_state=True, 
                           dropout=default_dropout, 
                           name='encoder_lstm')

# Passing the embedding layer output to the LSTM layer creates another link.
# IMPORTANT: The LSTM always returns three values. When return_sequences is
# False, encoder_outputs and state_h are the SAME. When return_sequences is
# True, encoder_outputs contains the encoder hidden states from each time step.
#
# Side note: we won't be using encoder_outputs here so that variable can be 
# replaced with a _ if preferred.
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding_output)

# The final hidden and cell/context states from the encoder will be the the
# initial states for the decoder.
encoder_states = (state_h, state_c)

decoder_inputs = layers.Input(shape=[None], name='decoder_inputs')


decoder_embeddings = layers.Embedding(target_vocab_size, 
                                      embedding_dim, 
                                      mask_zero=True,
                                      name='decoder_embeddings')


decoder_embedding_output = decoder_embeddings(decoder_inputs)

# Return sequences set to True.
decoder_lstm = layers.LSTM(hidden_dim,
                           return_sequences=True,
                           return_state=True,
                           dropout=default_dropout,
                           name='decoder_lstm')


# Set the decoder's initial state to the encoder's final output states. Since
# return_sequences is set to True, decoder_outputs is going to be a collection of
# the decoder's hidden state at each timestep. Also note that since we don't need
# the decoder's final hidden output and cell states, those are just set to _.
decoder_outputs, _, _ = decoder_lstm(decoder_embedding_output, initial_state=encoder_states)

# Have a softmax layer in the end to create a probability distribution for the output word.
decoder_dense = layers.Dense(target_vocab_size, activation='softmax', name='decoder_dense')

# The probability distribution for the output word.
y_proba = decoder_dense(decoder_outputs)

# Note how the model is taking two inputs in an array.
model = tf.keras.Model([encoder_inputs, decoder_inputs], y_proba, name='hun_eng_seq2seq_nmt_no_attention')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  metrics='sparse_categorical_accuracy')
model.summary()

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='hun_eng_seq2seq_nmt_no_attention.png', show_shapes=True, show_layer_names=True)

print('encoder_inputs layer\n input dimension {}\n output dimension: {}'.format((batch_size, max_encoding_len), (batch_size, max_encoding_len)))
print()
print('encoder_embeddings layer\n input dimension {}\n output dimension: {}'.format((batch_size, max_encoding_len), (batch_size, max_encoding_len, embedding_dim)))
print()
print('encoder_lstm layer\n input dimension {}\n output dimension: {}'.format((batch_size, max_encoding_len, embedding_dim), [(batch_size, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)]))
print()
print()
print('decoder_inputs layer\n input dimension {}\n output dimension: {}'.format((batch_size, max_decoding_len), (batch_size, max_decoding_len)))
print()
print('decoder_embeddings layer\n input dimension {}\n output dimension: {}'.format((batch_size, max_decoding_len), (batch_size, max_decoding_len, embedding_dim)))
print()
print('decoder_lstm layer\n input dimension {}\n output dimension: {}'.format([(batch_size, max_decoding_len, embedding_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)], [(batch_size, max_decoding_len, hidden_dim), (batch_size, hidden_dim), (batch_size, hidden_dim)]))
print()
print('decoder_dense layer(softmax)\n input dimension {}\n output dimension: {}'.format((batch_size, max_decoding_len, hidden_dim), (batch_size, max_decoding_len, target_vocab_size)))

# Saving this to a folder on my local machine.
filepath="./HunEngNMTNoAttention/training1/cp.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                 save_weights_only=True,
                                                 verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# history = model.fit([padded_train_encoder_inputs, padded_train_decoder_inputs], padded_train_decoder_targets,
#                      batch_size=batch_size,
#                      epochs=epochs,
#                      validation_data=([padded_val_encoder_inputs, padded_val_decoder_inputs], padded_val_decoder_targets),
#                      callbacks=[cp_callback, es_callback])

###### Save the model.
# model.save('hun_eng_s2s_nmt_no_attention')


###### Zip and download the model.
# !zip -r ./hun_eng_s2s_nmt_no_attention.zip ./hun_eng_s2s_nmt_no_attention
# files.download("./hun_eng_s2s_nmt_no_attention.zip")


###### Save the tokenizers as JSON files. The resulting files can be downloaded by left-clicking on them.
# source_tokenizer_json = source_tokenizer.to_json()
# with io.open('source_tokenizer.json', 'w', encoding='utf-8') as f:
#   f.write(json.dumps(source_tokenizer_json, ensure_ascii=False))

# target_tokenizer_json = target_tokenizer.to_json()
# with io.open('target_tokenizer.json', 'w', encoding='utf-8') as f:
#   f.write(json.dumps(target_tokenizer_json, ensure_ascii=False))

# Retrieve the tokenizers.
#!wget https://github.com/futuremojo/nlp-demystified/raw/main/models/nmt_no_attention/hun_eng_s2s_nmt_no_attention_tokenizers.zip

#unzip -o hun_eng_s2s_nmt_no_attention_tokenizers.zip

with open('source_tokenizer.json') as f:
    data = json.load(f)
    source_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

with open('target_tokenizer.json') as f:
    data = json.load(f)
    target_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

# Load the model.
model = tf.keras.models.load_model('hun_eng_s2s_nmt_no_attention')

# Retrieve the test dataset.
#!wget https://raw.githubusercontent.com/futuremojo/nlp-demystified/main/datasets/hun_eng_pairs/hun_eng_pairs_test.txt

with open('hun_eng_pairs_test.txt') as file:
  test = [line.rstrip() for line in file]

test[:3]

# Preprocess test dataset
padded_test_encoder_inputs, padded_test_decoder_inputs, padded_test_decoder_targets = process_dataset(test)

# Evaluate the model on the test set.
model.evaluate([padded_test_encoder_inputs, padded_test_decoder_inputs], padded_test_decoder_targets)

# These are the layers of our trained model.
[layer.name for layer in model.layers]

encoder_inputs = model.get_layer('encoder_inputs').input

encoder_embedding_layer = model.get_layer('encoder_embeddings')
encoder_embeddings = encoder_embedding_layer(encoder_inputs)

encoder_lstm = model.get_layer('encoder_lstm')

_, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embeddings)

encoder_states = [encoder_state_h, encoder_state_c]

# Our stand-alone encoder model. encoder_inputs is the input to the encoder,
# and encoder_states is the expected output.
encoder_model_no_attention = tf.keras.Model(encoder_inputs, encoder_states)

plot_model(encoder_model_no_attention, to_file='encoder_model_no_attention_plot.png', show_shapes=True, show_layer_names=True)

decoder_inputs = model.get_layer('decoder_inputs').input

decoder_embedding_layer = model.get_layer('decoder_embeddings')
decoder_embeddings = decoder_embedding_layer(decoder_inputs)

# Inputs to represent the decoder's LSTM hidden and cell states. We'll populate 
# these manually using the encoder's output for the initial state.
decoder_input_state_h = tf.keras.Input(shape=(hidden_dim,), name='decoder_input_state_h')
decoder_input_state_c = tf.keras.Input(shape=(hidden_dim,), name='decoder_input_state_c')
decoder_input_states = [decoder_input_state_h, decoder_input_state_c]

decoder_lstm = model.get_layer('decoder_lstm')

decoder_sequence_outputs, decoder_output_state_h, decoder_output_state_c = decoder_lstm(
    decoder_embeddings, initial_state=decoder_input_states
)

# Update hidden and cell states for the next time step.
decoder_output_states = [decoder_output_state_h, decoder_output_state_c]

decoder_dense = model.get_layer('decoder_dense')
y_proba = decoder_dense(decoder_sequence_outputs)

decoder_model_no_attention = tf.keras.Model(
    [decoder_inputs] + decoder_input_states, 
    [y_proba] + decoder_output_states
)

plot_model(decoder_model_no_attention, to_file='decoder_model_no_attention_plot.png', show_shapes=True, show_layer_names=True)

def translate_without_attention(sentence: str, 
                                source_tokenizer, encoder,
                                target_tokenizer, decoder,
                                max_translated_len = 30):

  # Vectorize the source sentence and run it through the encoder.    
  input_seq = source_tokenizer.texts_to_sequences([sentence])

  # Get the tokenized sentence to see if there are any unknown tokens.
  tokenized_sentence = source_tokenizer.sequences_to_texts(input_seq)

  states = encoder.predict(input_seq)  

  current_word = '<sos>'
  decoded_sentence = []

  while len(decoded_sentence) < max_translated_len:
    
    # Set the next input word for the decoder.
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = target_tokenizer.word_index[current_word]
    
    # Determine the next word.
    target_y_proba, h, c = decoder.predict([target_seq] + states)
    target_token_index = np.argmax(target_y_proba[0, -1, :])
    current_word = target_tokenizer.index_word[target_token_index]

    if (current_word == '<eos>'):
      break

    decoded_sentence.append(current_word)
    states = [h, c]
  
  return tokenized_sentence[0], ' '.join(decoded_sentence)

# random.seed is just here to re-create results.
random.seed(1)
sentences = random.sample(test, 15)
sentences

def translate_sentences(sentences, translation_func, source_tokenizer, encoder,
                        target_tokenizer, decoder):
  translations = {'Tokenized Original': [], 'Reference': [], 'Translation': []}

  for s in sentences:
    source, target = s.split(SEPARATOR)
    source = preprocess_sentence(source)
    tokenized_sentence, translated = translation_func(source, source_tokenizer, encoder,
                                                      target_tokenizer, decoder)

    translations['Tokenized Original'].append(tokenized_sentence)
    translations['Reference'].append(target)
    translations['Translation'].append(translated)
  
  return translations

translations_no_attention = pd.DataFrame(translate_sentences(sentences, translate_without_attention,
                                                             source_tokenizer, encoder_model_no_attention,
                                                             target_tokenizer, decoder_model_no_attention))
translations_no_attention



