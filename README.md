# Import google drive
from google.colab import drive
drive.mount('/content/drive')

import os
import joblib
import pandas as pd
import numpy as np
import time, torch
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, f1_score, auc, accuracy_score, classification_report
from tensorflow.keras.models import clone_model
from tensorflow.keras import optimizers, mixed_precision
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Activation, Bidirectional, Dropout, TimeDistributed, LayerNormalization, MultiHeadAttention, Embedding,LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence


np.random.seed(seed=17)

# Import simulation results
results = pd.read_csv('/content/drive/MyDrive/Transformer_Data/results_summary.csv', keep_default_na=False)
results.head()


# 7000 flag 0 files + 7000 flag 1 files

# Set flag files path
flag_0_path = "/content/drive/MyDrive/Pickle_Data_Files/flag_0"
flag_1_path = "/content/drive/MyDrive/Pickle_Data_Files/flag_1"

files_flag_0 = [f for f in os.listdir(flag_0_path) if f.endswith('.pkl')]
files_flag_1 = [f for f in os.listdir(flag_1_path) if f.endswith('.pkl')]

print(f"🔹 Flag 0 files: {len(files_flag_0)}, Flag 1 files: {len(files_flag_1)}")

# Random sampling 
files_flag_0 = np.random.choice(files_flag_0, size=7000, replace=False)
files_flag_1 = np.random.choice(files_flag_1, size=7000, replace=False)

paths_fail, paths_non_fail = [], []
labels_fail, labels_non_fail = [], []

for file in files_flag_1:
    paths_fail.append(os.path.join(flag_1_path, file))
    labels_fail.append(1)

for file in files_flag_0:
    paths_non_fail.append(os.path.join(flag_0_path, file))
    labels_non_fail.append(0)

# Check the number of data
print(f"✅ Fail data count: {len(paths_fail)}")
print(f"✅ Non-fail data count: {len(paths_non_fail)}")

# Generate file and label list 
filepaths, labels = [None] * 14000, [None] * 14000
filepaths[::2], labels[::2] = paths_fail, labels_fail
filepaths[1::2], labels[1::2] = paths_non_fail, labels_non_fail

# Dataset slpit
train_paths, train_labels = filepaths[:12000], labels[:12000]          # 80% Train Data (12000)
val_paths, val_labels = filepaths[12000:13000], labels[12000:13000]    # 10% Validation Data(1000)
test_paths, test_labels = filepaths[13000:], labels[13000:]            # 10% Test Data (1000)

# Check the data size
print(f"Train data: {len(train_paths)}, Validation data: {len(val_paths)}, Test data: {len(test_paths)}")


# Data generator
def data_generator_prev(batch_size, filepaths, labels, length):
    def gen():
        while True:
            count = 0
            X_encoder, X_decoder, y = [], [], []

            for file, label in zip(filepaths, labels):
                arr = joblib.load(os.path.join(DATA_PATH, file))

                y.append(int(label))
                X_encoder.append(arr[:length, 1:])  # Encoder input
                X_decoder.append(arr[:length, 1:])  # Decoder input

                count += 1
                if count >= batch_size:
                    yield (np.array(X_encoder).astype(np.float32),
                           np.array(X_decoder).astype(np.float32)), np.array(y).astype(np.float32)
                    X_encoder, X_decoder, y = [], [], []
                    count = 0

    output_signature = (
        (tf.TensorSpec(shape=(None, length, 178), dtype=tf.float32),
         tf.TensorSpec(shape=(None, length, 178), dtype=tf.float32)),  
        tf.TensorSpec(shape=(None,), dtype=tf.float32)  
    )

    return tf.data.Dataset.from_generator(gen, output_signature=output_signature).repeat()

num = 0
for (X_encoder, X_decoder), y in data_generator_prev(10, train_paths, train_labels, 100):
    print(f"Encoder Input Shape: {X_encoder.shape}, Decoder Input Shape: {X_decoder.shape}, Labels Shape: {y.shape}")

    num += 1
    if num > 4:
        break


K.clear_session()

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
import numpy as np

# Positional Encoding
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    pos_enc = pos * angle_rates

    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  

    return tf.cast(pos_enc, dtype=tf.float32)

# Transformer Encoder 
def transformer_encoder(inputs, d_model, num_heads, ff_dim, dropout=0.2):
    attn_layer = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)

    # MultiHeadAttention return values
    attn_output, attn_scores = attn_layer(inputs, inputs, return_attention_scores=True)  

    attn_output = Dropout(dropout)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = Dense(ff_dim, activation="relu")(attn_output)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout)(ffn)
    ffn_output = LayerNormalization(epsilon=1e-6)(attn_output + ffn)

    return ffn_output, attn_scores  

# Transformer Decoder 
def transformer_decoder(inputs, encoder_output, d_model, num_heads, ff_dim, dropout=0.1):
    attn_layer = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    attn_output, self_attn_scores = attn_layer(inputs, inputs, return_attention_scores=True)  

    attn_output = Dropout(dropout)(attn_output)
    attn_output = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Encoder-Decoder Cross Attention 
    cross_attn_layer = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    cross_attn_output, cross_attn_scores = cross_attn_layer(attn_output, encoder_output, return_attention_scores=True)  
    cross_attn_output = Dropout(dropout)(cross_attn_output)
    cross_attn_output = LayerNormalization(epsilon=1e-6)(attn_output + cross_attn_output)

    ffn = Dense(ff_dim, activation="relu")(cross_attn_output)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout)(ffn)
    ffn_output = LayerNormalization(epsilon=1e-6)(cross_attn_output + ffn)

    return ffn_output, self_attn_scores, cross_attn_scores  

# Dimension settings
window_size = 30
n_feats = 178
d_model = 128  
num_heads = 8  
ff_dim = 256   

# Encoder input
encoder_inputs = Input(shape=(window_size, n_feats))

# Positional Encoding
pos_encoding = positional_encoding(window_size, d_model)
x = Dense(d_model)(encoder_inputs)  
x += pos_encoding  

# Transformer Encoder
encoder_output, encoder_attn_scores_1 = transformer_encoder(x, d_model, num_heads, ff_dim)
encoder_output, encoder_attn_scores_2 = transformer_encoder(encoder_output, d_model, num_heads, ff_dim)

# Decoder input
decoder_inputs = Input(shape=(window_size, n_feats))  

decoder_x = Dense(d_model)(decoder_inputs)  
decoder_x += positional_encoding(window_size, d_model)

# Transformer Decoder 
decoder_x, self_attn_scores_1, cross_attn_scores_1 = transformer_decoder(decoder_x, encoder_output, d_model, num_heads, ff_dim)
decoder_x, self_attn_scores_2, cross_attn_scores_2 = transformer_decoder(decoder_x, encoder_output, d_model, num_heads, ff_dim)

# Final output
decoder_x = GlobalAveragePooling1D()(decoder_x)  
outputs = Dense(1, activation="sigmoid")(decoder_x)  

# Generate model
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[outputs,
                                                                encoder_attn_scores_1, encoder_attn_scores_2,
                                                                self_attn_scores_1, self_attn_scores_2,
                                                                cross_attn_scores_1, cross_attn_scores_2])

# learning rate scheduling and optimiser
learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = Adam(learning_rate=lr_schedule)

losses = ["binary_crossentropy", None, None, None, None, None, None]


metrics = [
    ["accuracy", Precision(name="precision"), Recall(name="recall")],
    None, None, None, None, None, None
]

# Compile model
model.compile(loss=losses, optimizer=optimizer, metrics=metrics)


# Model output summary
model.summary()




# Train Settings
EPOCH_SIZE = 30

early_stop = EarlyStopping(monitor="val_loss", patience=3)

checkpoint = ModelCheckpoint(
    filepath="checkpoint.keras",
    save_weights_only=False,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1  
)


# Model Train
history = model.fit(  
    data_generator_prev(64, train_paths, train_labels, window_size),                
    validation_data=data_generator_prev(64, val_paths, val_labels, window_size),    
    validation_steps=15,                                                    
    epochs=EPOCH_SIZE,
    steps_per_epoch=187,                                                       
    callbacks=[early_stop, checkpoint] 
)



# Save model
model_save_path = "/content/drive/MyDrive/Transformer.keras"
model.save(model_save_path)

print(f"✅ Saved Successfully : {model_save_path}")




# Loss graph
plt.figure(figsize=(8, 6))
plt.title("Learning Curves", fontsize=16, fontweight='bold', pad=10)
plt.xlabel("Epoch", fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel("Cross Entropy", fontsize=14, fontweight='bold', labelpad=10)

plt.plot(history.history['loss'], label='Train', linewidth=2.5, marker='o', markersize=6, linestyle='--', color='blue')
plt.plot(history.history['val_loss'], label='Validation', linewidth=2.5, marker='s', markersize=6, linestyle='-', color='red')

plt.legend(fontsize=12, loc='upper right', frameon=True)
plt.grid(True, linestyle='--', alpha=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()




# Accuracy graph
plt.figure(figsize=(8, 6))
plt.title("Accuracy Curves", fontsize=16, fontweight='bold', pad=10)
plt.xlabel("Epoch", fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel("Accuracy", fontsize=14, fontweight='bold', labelpad=10)

plt.plot(history.history['dense_10_accuracy'], label='Train', linewidth=2.5, marker='o', markersize=6, linestyle='--', color='blue')
plt.plot(history.history['val_dense_10_accuracy'], label='Validation', linewidth=2.5, marker='s', markersize=6, linestyle='-', color='red')

plt.legend(fontsize=12, loc='lower right', frameon=True)
plt.grid(True, linestyle='--', alpha=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()




# Recall & Precision graph
batch_size = 64 
num_batches = 10  
window_size = 100

x_test_list = []
y_test_list = []

# Convert data generator into an iterator
data_gen = data_generator_prev(batch_size, filepaths, labels, window_size)
data_gen_iter = iter(data_gen)


for _ in range(num_batches):
    (x_encoder_batch, x_decoder_batch), y_batch = next(data_gen_iter)  
    x_test_list.append((x_encoder_batch, x_decoder_batch))  
    y_test_list.append(y_batch)

# Convert lists into numpy arrays
X_encoder_test = np.concatenate([x[0] for x in x_test_list], axis=0)  # Extract encoder inputs
X_decoder_test = np.concatenate([x[1] for x in x_test_list], axis=0)  # Extract decoder inputs
y_test = np.concatenate(y_test_list, axis=0)  

print(f"X_encoder_test shape: {X_encoder_test.shape}, X_decoder_test shape: {X_decoder_test.shape}, y_test shape: {y_test.shape}")

#cGet model predictions
y_pred_prob = model.predict((X_encoder_test, X_decoder_test))  


y_test = y_test.flatten()


y_pred_prob = y_pred_prob[0]  
y_pred_prob = y_pred_prob.flatten()


# Compute Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='o', linestyle='-', color="blue", label="Precision-Recall Curve")
plt.xlabel("Recall", fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel("Precision", fontsize=14, fontweight='bold', labelpad=10)
plt.title("Precision-Recall Curve", fontsize=16, fontweight='bold', pad=10)
plt.legend(fontsize=12, loc='lower left', frameon=True)
plt.grid(True, linestyle='--', alpha=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()




# PR-AUC & recall metrics
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")


y_pred_labels = (y_pred_prob > 0.5).astype(int)  
f1 = f1_score(y_test, y_pred_labels)
print(f"F1-Score: {f1:.4f}")




# RMSE graph
train_rmse = np.sqrt(history.history['loss'])
val_rmse = np.sqrt(history.history['val_loss'])

plt.figure(figsize=(8, 6))
plt.title("RMSE Curves", fontsize=16, fontweight='bold', pad=10)
plt.xlabel("Epoch", fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel("RMSE", fontsize=14, fontweight='bold', labelpad=10)

plt.plot(train_rmse, label='Train RMSE', linewidth=2.5, marker='o', markersize=6, linestyle='--', color='blue')
plt.plot(val_rmse, label='Validation RMSE', linewidth=2.5, marker='s', markersize=6, linestyle='-', color='red')

plt.legend(fontsize=12, loc='upper right', frameon=True)
plt.grid(True, linestyle='--', alpha=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()




# Test model
test_y, pred_y = [], []
count=0

for X, y in data_generator_prev(64, test_paths, test_labels, 100):
    test_y.append(y)

  
    pred_y.append(model.predict(X)[0])

    count += 1
    if count > 15:
        break

test_y = np.concatenate(test_y)
pred_y = np.concatenate(pred_y)
pred_y[pred_y >= 0.5] = 1
pred_y[pred_y < 0.5] = 0



# Accuracy, precision, recall, F1 score
acc = accuracy_score(test_y, pred_y)

print('Accuracy on test set: {:.3f}'.format(acc))
print(classification_report(test_y, pred_y))
