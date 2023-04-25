import numpy as np
import tensorflow as tf
from mne.io import read_raw_edf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from mne import set_eeg_reference, pick_types, Epochs, make_fixed_length_epochs, concatenate_epochs, \
    events_from_annotations

if __name__=="__main__":
    # Import necessary libraries


    # Load the PhysioNet EEG Motor Movement/Imagery Dataset
    data_path = '/5.felev/PycharmProjects/EEGArtifactRemovalProject/files/'
    subjects = range(1, 2)
    runs=range(1,2)

    epochs = []

    # Define the reference electrode
    ref_channel = ['Cz']

    for subject in subjects:
            for run in runs:
            # Load the EEG data
                raw = read_raw_edf(data_path + 'S{:03d}/S{:03d}R{:02d}.edf'.format(subject, subject,run), preload=True)
                events = raw.annotations
                for event in events:
                    raw.rename_channels(lambda x: x.strip('.'))
                 #raw.set_montage('standard_1005')
                    raw.filter(1, 30, fir_design='firwin')
                    raw.notch_filter(60, fir_design='firwin')

            # Apply the reference electrode
                    set_eeg_reference(raw, ref_channels=ref_channel)

            # Epoch the data
                    events, _ = events_from_annotations(raw, chunk_duration=4.0)
                    epochs.append(Epochs(raw, events, tmin=0, tmax=4.0, baseline=(0, 0), preload=True, detrend=1))

    # Concatenate the epochs and extract the data and labels
    epochs = concatenate_epochs(epochs)
    data = epochs.get_data()
    labels = epochs.events[:, 2]

    # Split the data into training and testing sets
    train_idx = np.arange(0, len(data), 3)
    test_idx = np.arange(1, len(data), 3)
    x_train = data[train_idx]
    y_train = labels[train_idx]
    y_train=np.asarray(y_train).reshape(y_train.shape[0],1)
    x_test = data[test_idx]
    y_test = labels[test_idx]
    y_test = np.asarray(y_test).reshape(y_test.shape[0],1)

    arr = np.array([1, 2, 3, 4, 5])

    # Define the deep learning model architecture
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(labels))
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.25)(x)
    x = LSTM(64)(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model and define the training parameters
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the training set
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    history = model.fit(x_train, to_categorical(y_train), batch_size=32, epochs=100,
                        validation_data=(x_test, to_categorical(y_test)), callbacks=[early_stopping])
