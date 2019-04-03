from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adadelta, Adam

def train(x_train,y_train,x_val,y_val,epochs,activation):

    # using relu allows for overfitting
    model = Sequential()
    model.add(Dense(1000, input_dim=x_train.shape[1], activation=activation))
    model.add(Dense(1000, activation=activation))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=64)

    return model