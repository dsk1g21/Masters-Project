import numpy as np
import tensorflow as tf
from keras import layers, callbacks
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import os
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):  
        super().__init__(**kwargs)
        self.conv1 = layers.Conv1D(64, 5, activation='relu', padding='same')
        self.pool = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(128, 5, activation='relu', padding='same')
        self.conv3 = layers.Conv1D(256, 5, activation='relu', padding='same')  
        self.globalpool = layers.GlobalAveragePooling1D()
        self.d1 = layers.Dense(64, activation='relu')
        self.drop = layers.Dropout(0.5)
        self.d2 = layers.Dense(32, activation='relu') 
        self.final = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.globalpool(x)
        x = self.d1(x)
        x = self.drop(x)
        x = self.d2(x)
        x = self.drop(x)
        return self.final(x)
if __name__ == "__main__":

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
            r"C:\masters\machine learning\Provan List\26011996.csv", r"C:\masters\machine learning\Provan List\27011996.csv", r"C:\masters\machine learning\february.csv",
            r'C:\masters\machine learning\march.csv', r'C:\masters\machine learning\validate.csv']

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
    #=========================================== AUGMENTATION ====================================================
    augment = df[df['event'] == 1].copy()
    def Augmentation(df, column_name):
        if column_name == 'v':
            df[column_name] = df[column_name].apply(lambda x: x - np.random.uniform(0, 0.1))
        elif column_name == 'p_l':
            df[column_name] = df[column_name].apply(lambda x: x + np.random.uniform(0, 0.5))
        return df
    new_df, new_again, goddamn = Augmentation(augment, 'v'), Augmentation(augment, 'v'), Augmentation(augment, 'v')
    new_df, new_again, goddamn = Augmentation(new_df, 'p_l'), Augmentation(new_again, 'p_l'), Augmentation(goddamn, 'p_l')
    #=========================================== END AUGMENTATION =========================================================
    df = pd.concat([df, new_df, new_again, goddamn], ignore_index=True)
    #=========================================== DO NOT CHANGE THIS PART ====================================================
    data = df.values
    batch_size = 5008

    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)] 

    np.random.shuffle(batches)

    shuffled_data = np.concatenate(batches, axis=0)  
    df_shuffled = pd.DataFrame(shuffled_data, columns=df.columns)
    df_new = df_shuffled[['p_l', 'v', 'event']]

    train, test = tts(df_new, test_size=0.1, shuffle=False)  

    train_y = train.pop('event')
    test_y = test.pop('event')

    train_y = train_y.values.reshape(-1, 1)
    test_y = test_y.values.reshape(-1, 1)

    tf.convert_to_tensor(train.values)
    tf.convert_to_tensor(test.values)

    train = tf.expand_dims(train.values, axis=-1)
    test = tf.expand_dims(test.values, axis=-1)

    #=========================================== END OF DO NOT CHANGE ====================================================
    bathch_size = 16
    train_ds = tf.data.Dataset.from_tensor_slices((train, train_y)).batch(bathch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test, test_y)).batch(bathch_size)
        
    print("Unique labels:", np.unique(train_y))

    model = MyModel()

    train_labels = train_y.flatten()
    classes = np.unique(train_labels)

    unique, counts = np.unique(train_y, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',       
        patience=5,              
        restore_best_weights=True)

    checkpoint_cb = callbacks.ModelCheckpoint(
        'best_model.keras', 
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,  # change to True if you only want weights
        mode='min',
        verbose=1)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels)

    class_weights_dict = dict(zip(classes, class_weights))
    print("Class weights:", class_weights_dict)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

    model.fit(train, train_y, validation_data=(test, test_y), epochs=500,
            batch_size=bathch_size, class_weight=class_weights_dict,
            callbacks=[early_stop, checkpoint_cb])

    y_true = []
    y_pred = []

    for x_batch_test, y_batch_test in test_ds:
        logits = model(x_batch_test, training=False)
        preds = (logits.numpy() > 0.5).astype(int)
        y_true.extend(y_batch_test.numpy().flatten())
        y_pred.extend(preds.flatten())

    print(classification_report(y_true, y_pred))
    from sklearn.metrics import roc_auc_score

    # After predicting all test set
    y_pred_probs = []  # store actual sigmoid probabilities here

    for x_batch_test, y_batch_test in test_ds:
        logits = model(x_batch_test, training=False)
        y_pred_probs.extend(logits.numpy().flatten())

    print("Test AUC:", roc_auc_score(y_true, y_pred_probs))
