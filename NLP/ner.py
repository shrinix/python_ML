import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
import plotly.express as px
from keras.layers import Dense,LSTM,Embedding,InputLayer,SpatialDropout1D,Bidirectional
from keras.models import Sequential,Model
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback

def load_data():
    data = pd.read_csv("./data/ner_datasetreference.csv", encoding="latin1")

    data = data.fillna(method="ffill")
    print(data.head(20))

    print("Unique words in corpus:", data['Word'].nunique())
    print("Unique tags in corpus:", data['Tag'].nunique())

    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    num_words = len(words)

    tags = list(set(data["Tag"].values))
    num_tags = len(tags)

    # fig = px.histogram(data[~data.Tag.str.contains("O")], x="Tag",color="Tag")
    # fig.show()

    return data, words, tags

def sentence_integrate(data):
  agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
  return data.groupby('Sentence #').apply(agg_func).tolist()

def build_vocab(words, tags):
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    print(tag2idx)
    return word2idx, tag2idx

def create_NN_model(max_len, num_words):

    model = keras.Sequential()
    #model.add(InputLayer(max_len,))
    model.add(keras.Input(shape=(max_len,), dtype='int32'))
    model.add(Embedding(input_dim=num_words, output_dim=max_len, input_length=max_len))
    model.add(SpatialDropout1D(0.1))
    model.add( Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))

    model.summary()

    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=False,
        show_layer_names=True, rankdir='LR', expand_nested=True, dpi=300,
    )

    model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return model

def evaluate_model(model, x_test, y_test):
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss: {} ".format(results[0]))
    print("test accuracy: {} ".format(results[1]))

def train_model(model, x_train, y_train, x_test, y_test):
    logdir="./log/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    chkpt = ModelCheckpoint("model_weights.h5", monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=1, verbose=0, mode='max', baseline=None, restore_best_weights=False)
    callbacks = [PlotLossesCallback(), chkpt, early_stopping,tensorboard_callback]

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test,y_test),
        batch_size=32, 
        epochs=1,
        #callbacks=callbacks, //Need to fix runtime error with options
        verbose=1
    )

    return model

def main():
    np.random.seed(0)
    plt.style.use("ggplot")
    max_len = 50

    # print('Tensorflow version:', tf.__version__)
    # print('GPU detected:', tf.config.list_physical_devices('GPU'))
    data, words, tags = load_data()
    num_words = len(words)
    sentences=sentence_integrate(data)
    import plotly.express as px

    # fig = px.histogram(pd.DataFrame([len(s) for s in sentences],columns=['length']),x="length",marginal='box')
    # fig.show()

    print (sentences[0])

    word2idx, tag2idx = build_vocab(words, tags)

    from keras_preprocessing.sequence import pad_sequences

    X = [[word2idx[w[0]] for w in s] for s in sentences]

    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)

    y = [[tag2idx[w[2]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = create_NN_model(max_len, num_words)
    model = train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)

    i = np.random.randint(0, x_test.shape[0])
    print("This is sentence:",i)
    p = model.predict(np.array([x_test[i]]))
    p = np.argmax(p, axis=-1)

    print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
    print("-" *30)
    for w, true, pred in zip(x_test[i], y_test[i], p[0]):
        print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

if __name__ == "__main__":

    main()