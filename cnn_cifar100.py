
from keras import datasets, Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# download the dataset from source and divide into training and test set
(xtr, ytr), (xt, yt) = datasets.cifar100.load_data()

# convert the 2d array output labels into 1D array
ytr = ytr.reshape(-1,)
yt = yt.reshape(-1,)

# plot a sample from the training class
plt.figure(figsize=(10,10))
for image in range(0,20):
    i=image
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    j=i+0# añadir de 25 en 25 para cambiar el bloque de fotos
    data_plot = xtr[j]
    plt.imshow(data_plot)
    plt.xlabel(str(yt[j]))
plt.show()

# normalizing the training and testing data
xtr = xtr/255
xt = xt/255

model = Sequential()
model.add(Conv2D(input_shape=(32, 32, 3), kernel_size=(2, 2), padding='same', strides=(2, 2), filters=32))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
model.add(Conv2D(kernel_size=(2, 2), padding='same', strides=(2, 2), filters=64))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.summary()


opt = 'adam'

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(xtr, ytr, epochs=25)

test_loss,test_acc=model.evaluate(xt, yt)
print("test accuracy: ",test_acc)

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.tight_layout()
    plt.ylabel('Observación')
    plt.xlabel('Predicción')
Y_pred = model.predict(xt)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
confusion_mtx = confusion_matrix(yt, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(100))
