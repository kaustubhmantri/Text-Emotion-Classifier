from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

def build_model(input_dim, max_length, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, xtrain, ytrain, xtest, ytest, epochs=10, batch_size=32):
    history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_data=(xtest, ytest))
    return history
