
import pandas as pd
from numpy.random import seed
from tensorflow import set_random_seed
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -------------------- CONFIGURATION --------------------
BATCH_SIZE = 64
NUM_EPOCHS = 1
SEED = 409

# ------------------------------------------------------------

seed(SEED)
set_random_seed(SEED)

# Load data and explore contracts size/duration
data = pd.read_parquet('./data/processed/full_final_df.parquet.gzip')

# Contract size evaluatuion to define max_length size for series size
data.contract_size = data.payments_history.apply(lambda x: len(x))
print(data.contract_size.describe())

interest_quantiles = [0.8, 0.85, 0.9, 0.95]

for q in interest_quantiles:
    print(f'Quantile {round(q*100)}% of contract size is {data.contract_size.quantile(q)}')


# Data preparation for modeling
X = pad_sequences(data.payments_history.tolist(), value = -1) # First approach consider max length

y = data.next_period_status.tolist()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y, num_classes=label_encoder.classes_.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(94, dropout = 0.1, recurrent_dropout = 0.1))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(4, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[CategoricalAccuracy()])
history = model.fit(X_train, y_train, validation_split = 0.1,
                    epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Evaluate training
plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate fit quality
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
