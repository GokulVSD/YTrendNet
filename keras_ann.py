from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adadelta, Adam

def test(x_train,y_train,x_val,y_val):

    model = Sequential()
    model.add(Dense(3000, input_dim=x_train.shape[1], activation='sigmoid'))
    model.add(Dense(3000, activation='sigmoid'))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.summary()

    model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=64)

    scores = model.evaluate(x_val, y_val)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))