from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
# load train and test dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#reshape
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
#hot encode - the output variable is an integer from 0 to 9.
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
# convert from integers to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize to range 0-1 since RGB value can range from 0 to 255
X_train = X_train / 255.0
X_test = X_test / 255.0
# define cnn model
def define_model():
    model = Sequential()
    #input layer 784 nodes and layer-1 512 nodes
    model.add(Dense(512, input_shape=(784,), activation='relu'))
    model.add(Dropout(0.2))
    #hidden layer-2
    model.add(Dense(512, activation='relu',))
    model.add(Dropout(0.2))
    #output layer
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# Train model
model=define_model()
model.fit(X_train, Y_train , batch_size=128, epochs=20,verbose=2, validation_data=(X_test, Y_test))
# obtained accuracy for 20 epochs -
#Epoch 1/20
#469/469 - 8s - loss: 0.2426 - accuracy: 0.9268 - val_loss: 0.0980 - val_accuracy: 0.9681
#Epoch 2/20
# 469/469 - 9s - loss: 0.0993 - accuracy: 0.9695 - val_loss: 0.0850 - val_accuracy: 0.9733
#Epoch 3/20
#469/469 - 6s - loss: 0.0732 - accuracy: 0.9768 - val_loss: 0.0798 - val_accuracy: 0.9759
#Epoch 4/20
#469/469 - 6s - loss: 0.0548 - accuracy: 0.9825 - val_loss: 0.0677 - val_accuracy: 0.9795
#Epoch 5/20
#469/469 - 7s - loss: 0.0461 - accuracy: 0.9852 - val_loss: 0.0571 - val_accuracy: 0.9824
#Epoch 6/20
#469/469 - 6s - loss: 0.0365 - accuracy: 0.9877 - val_loss: 0.0619 - val_accuracy: 0.9816
#Epoch 7/20
#469/469 - 6s - loss: 0.0350 - accuracy: 0.9886 - val_loss: 0.0618 - val_accuracy: 0.9840
#Epoch 8/20
#469/469 - 6s - loss: 0.0300 - accuracy: 0.9907 - val_loss: 0.0675 - val_accuracy: 0.9823
#Epoch 9/20
#469/469 - 6s - loss: 0.0263 - accuracy: 0.9909 - val_loss: 0.0783 - val_accuracy: 0.9810
#Epoch 10/20
#469/469 - 7s - loss: 0.0238 - accuracy: 0.9920 - val_loss: 0.0740 - val_accuracy: 0.9815
#Epoch 11/20
#469/469 - 7s - loss: 0.0232 - accuracy: 0.9918 - val_loss: 0.0687 - val_accuracy: 0.9839
#Epoch 12/20
#469/469 - 7s - loss: 0.0212 - accuracy: 0.9930 - val_loss: 0.0652 - val_accuracy: 0.9827
#Epoch 13/20
#469/469 - 6s - loss: 0.0211 - accuracy: 0.9927 - val_loss: 0.0747 - val_accuracy: 0.9833
#Epoch 14/20
#469/469 - 6s - loss: 0.0218 - accuracy: 0.9927 - val_loss: 0.0739 - val_accuracy: 0.9840
#Epoch 15/20
#469/469 - 6s - loss: 0.0178 - accuracy: 0.9943 - val_loss: 0.0644 - val_accuracy: 0.9845
#Epoch 16/20
#469/469 - 6s - loss: 0.0153 - accuracy: 0.9951 - val_loss: 0.0781 - val_accuracy: 0.9831
#Epoch 17/20
#469/469 - 6s - loss: 0.0155 - accuracy: 0.9951 - val_loss: 0.0754 - val_accuracy: 0.9833
#Epoch 18/20
#469/469 - 7s - loss: 0.0167 - accuracy: 0.9947 - val_loss: 0.0698 - val_accuracy: 0.9838
#Epoch 19/20
#469/469 - 6s - loss: 0.0147 - accuracy: 0.9952 - val_loss: 0.0759 - val_accuracy: 0.9829

#Epoch 20/20-Final
#469/469 - 8s - loss: 0.0176 - accuracy: 0.9941 - val_loss: 0.0779 - val_accuracy: 0.9837







