import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers, optimizers
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import os
from sklearn.utils import class_weight

os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

files = [r"C:\masters\machine learning\january.csv", r"C:\masters\machine learning\Provan List\02081996.csv", r"C:\masters\machine learning\Provan List\03011996.csv",
         r"C:\masters\machine learning\Provan List\03041996.csv",r"C:\masters\machine learning\Provan List\03081996.csv",r"C:\masters\machine learning\Provan List\04041996.csv",
         r"C:\masters\machine learning\Provan List\04081996.csv", r"C:\masters\machine learning\Provan List\05041996.csv", r"C:\masters\machine learning\Provan List\06081996.csv",
         r"C:\masters\machine learning\Provan List\07041996.csv", r"C:\masters\machine learning\Provan List\08021996.csv", r"C:\masters\machine learning\Provan List\08041996.csv",
         r"C:\masters\machine learning\Provan List\11081995.csv", r"C:\masters\machine learning\Provan List\11091995.csv", r"C:\masters\machine learning\Provan List\12051996.csv", 
         r"C:\masters\machine learning\Provan List\13071995.csv", r"C:\masters\machine learning\Provan List\13101995.csv", r"C:\masters\machine learning\Provan List\14051996.csv",
         r"C:\masters\machine learning\Provan List\14071995.csv", r"C:\masters\machine learning\Provan List\14081995.csv", r"C:\masters\machine learning\Provan List\14091996.csv",
         r"C:\masters\machine learning\Provan List\14101995.csv", r"C:\masters\machine learning\Provan List\15081995.csv", r"C:\masters\machine learning\Provan List\18011996.csv", 
         r"C:\masters\machine learning\Provan List\19011996.csv", r"C:\masters\machine learning\Provan List\19031996.csv", r"C:\masters\machine learning\Provan List\24091995.csv",
         r"C:\masters\machine learning\Provan List\24091996.csv", r"C:\masters\machine learning\Provan List\25011996.csv", r"C:\masters\machine learning\Provan List\25091996.csv",
         r"C:\masters\machine learning\Provan List\26011996.csv", r"C:\masters\machine learning\Provan List\27011996.csv"]

df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
np.int64(df['time'])
df['time'] = df['time'].astype(np.int64)

df['p_l'] = pd.to_numeric(df['p_l'], errors='coerce')
df['v'] = pd.to_numeric(df['v'], errors='coerce')

def min_max_scaling(df, column_name, a, b):
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    df[column_name] = a * (df[column_name] - min_value) / (max_value - min_value) - b

min_max_scaling(df, 'p_l', 1, 0)
min_max_scaling(df, 'v', 2, 1)
min_max_scaling(df, 'time', 1, 0)

df.dropna(inplace=True)
#=========================================== DO NOT CHANGE THIS PART ====================================================
data = df.values  
batch_size = 101

batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]  # List of arrays

np.random.shuffle(batches)

shuffled_data = np.concatenate(batches, axis=0)  # Merge batches back into a single array
df_shuffled = pd.DataFrame(shuffled_data, columns=df.columns)
print(df_shuffled)

train, test = tts(df_shuffled, test_size=0.1, shuffle=False)  

train_y = train.pop('event')
test_y = test.pop('event')

train_y = train_y.values.reshape(-1, 1)
test_y = test_y.values.reshape(-1, 1)

tf.convert_to_tensor(train.values)
tf.convert_to_tensor(test.values)

train = tf.expand_dims(train.values, axis=-1)
test = tf.expand_dims(test.values, axis=-1)

#=========================================== END OF DO NOT CHANGE ====================================================

train_ds = tf.data.Dataset.from_tensor_slices((train, train_y)).batch(16)
test_ds = tf.data.Dataset.from_tensor_slices((test, test_y)).batch(16)

#for feature, event in train_ds.take(1):
 #   print('Features: {}, Event: {}'.format(feature, event))
print("Unique labels:", np.unique(train_y))

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv1D(64, 5, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(128, 5, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv1D(256, 5, activation='relu', padding='same')  
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.drop1 = layers.Dropout(0.3) 
        self.d2 = layers.Dense(64, activation='relu')
        self.drop2 = layers.Dropout(0.3)  
        self.final = layers.Dense(1, activation="sigmoid")  

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.drop1(x)
        x = self.d2(x)
        x = self.drop2(x)
        return self.final(x)

model = MyModel()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_y.flatten()),
    y=train_y.flatten()
)

class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class Weights:", class_weights_dict)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)

model.fit(train, train_y, validation_data=(test, test_y), epochs=10, batch_size=16, class_weight=class_weights_dict)
from sklearn.metrics import classification_report

# At the end of testing
y_true = []
y_pred = []

for x_batch_test, y_batch_test in test_ds:
    logits = model(x_batch_test, training=False)
    preds = (logits.numpy() > 0.5).astype(int)
    y_true.extend(y_batch_test.numpy().flatten())
    y_pred.extend(preds.flatten())

print(classification_report(y_true, y_pred))
auc = tf.keras.metrics.AUC(name='auc')
...
auc.update_state(y_batch_test, logits)
print("Test AUC:", auc.result().numpy())
auc.reset_state()
