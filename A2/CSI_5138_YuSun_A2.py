from mlxtend.data import loadlocal_mnist

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras_util import convert_drawer_model
print(tf.__version__)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

tf.random.set_seed(2)

def softmax():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10,  activation='softmax'))
    return model

def MLP(isDeeper,isDropout):
    model = tf.keras.models.Sequential()
    if(isDeeper):
        model.add(tf.keras.layers.Dense(2048, activation='relu'))
        if(isDropout):
            model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        if(isDropout):
            model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    if(isDropout):
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(10,  activation='softmax'))
    return model


def CNN(isDeeper, isBN, kernel_size=3, input_shape=(28, 28, 1)):
    model = tf.keras.models.Sequential()
    if(not isDeeper):
        model.add(tf.keras.layers.Conv2D(32, (kernel_size, kernel_size), activation='relu', input_shape=input_shape))
        if(isBN):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv2D(64, (kernel_size, kernel_size), activation='relu'))
        if(isBN):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(tf.keras.layers.Dropout(0.2))
    else:
        model.add(tf.keras.layers.Conv2D(16, (kernel_size, kernel_size), activation='relu',  input_shape=input_shape))
        if(isBN):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2),padding="same"))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv2D(32, (kernel_size, kernel_size), activation='relu'))
        if(isBN):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2),padding="same"))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Conv2D(64, (kernel_size, kernel_size), activation='relu',padding='same'))
        if(isBN):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2),padding="same"))
        model.add(tf.keras.layers.Conv2D(128, (kernel_size, kernel_size),
                                         activation='relu', padding='same'))
        if(isBN):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
    return model

def AnotherCNN():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def loadMnist():
    train_images, train_labels = loadlocal_mnist(
            images_path='MNIST/train-images-idx3-ubyte', 
            labels_path='MNIST/train-labels-idx1-ubyte')
    print(train_images.shape)
    print(train_labels.shape)
    test_images, test_labels = loadlocal_mnist(
            images_path='MNIST/t10k-images-idx3-ubyte', 
            labels_path='MNIST/t10k-labels-idx1-ubyte')
    return train_images,train_labels,test_images,test_labels

def unpickle(file):
    """load the cifar-10 data"""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def loadCifrt10():
    data_dir = 'cifar-10-batches-py'
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    train_images = None
    cifar_train_filenames = []
    train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_images = cifar_train_data_dict[b'data']
        else:
            train_images = np.vstack((train_images, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        train_labels += cifar_train_data_dict[b'labels']

    train_images = train_images.reshape((len(train_images), 3, 32, 32))
    train_images = np.rollaxis(train_images, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    train_labels = np.array(train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    test_images = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    test_labels = cifar_test_data_dict[b'labels']

    test_images = test_images.reshape((len(test_images), 3, 32, 32))
    test_images = np.rollaxis(test_images, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels

def plotMnist(train_data):
    # plot first few images
    for i in range(9):
	# define subplot
        plt.subplot(330 + 1 + i)
	# plot raw pixel data
        plt.imshow(train_images[i].reshape(28,28), cmap=plt.get_cmap('gray'))
    # show the figure

def plotCifar10(train_images):
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_images.shape[0])
            ax[m, n].imshow(train_images[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = loadMnist()
    plotMnist(train_images)

    train_images10, train_labels10, test_images10, test_labels10 = loadCifrt10()
    plotCifar10(train_images10)

    metrics = [
        'accuracy',
        # tf.keras.metrics.TruePositives(name='tp'),
        # tf.keras.metrics.FalsePositives(name='fp'),
        # tf.keras.metrics.TrueNegatives(name='tn'),
        # tf.keras.metrics.FalseNegatives(name='fn'),
        # tf.keras.metrics.Precision(name='precision'),
        # tf.keras.metrics.Recall(name='recall'),
        # tf.keras.metrics.AUC(name='auc'),
    ]

    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    train_images10 = train_images10 / 255.0
    test_images10 = test_images10 / 255.0
    train_labels10 = tf.keras.utils.to_categorical(train_labels10, 10)
    test_labels10 = tf.keras.utils.to_categorical(test_labels10, 10)
    train_images10 = train_images10.astype(np.float32).reshape(-1,3072)
    test_images10 = test_images10.astype(np.float32).reshape(-1,3072)

    model = softmax()
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    # loss, accuracy, precision, recall,_ = model.evaluate(test_images, test_labels)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('softmax_mnist.out',  np.array(result), fmt='%.04f')

    model = MLP(False, False)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('mlp_mnist1.out',  np.array(result), fmt='%.04f')

    model = MLP(False, True)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('mlp_mnist2.out',  np.array(result), fmt='%.04f')

    model = MLP(True, False)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('mlp_mnist3.out',  np.array(result), fmt='%.04f')

    model = MLP(True, True)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('mlp_mnist4.out',  np.array(result), fmt='%.04f')

    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    model = CNN(False, False,3)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('cnn_mnist1.out',  np.array(result), fmt='%.04f')

    model = CNN(False, True, 3)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('cnn_mnist2.out',  np.array(result), fmt='%.04f')

    model = CNN(False, False, 5)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('cnn_mnist3.out',  np.array(result), fmt='%.04f')

    model = CNN(True, False,3)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('cnn_mnist4.out',  np.array(result), fmt='%.04f')

    model = CNN(True, True, 3)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('cnn_mnist5.out',  np.array(result), fmt='%.04f')

    model = CNN(True, False, 5)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images, train_labels, epochs=10)
    result = model.evaluate(test_images, test_labels)
    np.savetxt('cnn_mnist6.out',  np.array(result), fmt='%.04f')


    model = softmax()
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=25,batch_size=64)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('softmax_cifar.out', np.array(result), fmt='%.04f')

    model = MLP(False, False)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=25,batch_size=64)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('mlp_cifar1.out', np.array(result), fmt='%.04f')

    model = MLP(False, True)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=25,batch_size=64)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('mlp_cifar2.out', np.array(result), fmt='%.04f')

    model = MLP(True, False)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=25,batch_size=64)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('mlp_cifar3.out', np.array(result), fmt='%.04f')

    model = MLP(True, True)
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=50)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('mlp_cifar4.out', np.array(result), fmt='%.04f')
    
    train_images10 = train_images10.reshape(-1, 32, 32, 3)
    test_images10 = test_images10.reshape(-1, 32, 32, 3)
    
    model = CNN(False, False, 3,input_shape=(32, 32, 3))
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=25,batch_size=64)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('cnn_cifar1.out', np.array(result), fmt='%.04f')

    model = CNN(False, True, 3,input_shape=(32, 32, 3))
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=25,batch_size=64)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('cnn_cifar2.out', np.array(result), fmt='%.04f')

    model = CNN(False, False, 5,input_shape=(32, 32, 3))
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=25,batch_size=64)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('cnn_cifar3.out', np.array(result), fmt='%.04f')

    model = CNN(True, False, 3,input_shape=(32, 32, 3))
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit1 = model.fit(train_images10, train_labels10, epochs=25,batch_size=64,validation_data=(test_images10, test_labels10))
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('cnn_cifar4.out', np.array(result), fmt='%.04f')
    # model = convert_drawer_model(model)
    # model.save_fig("CNN.svg")

    model = CNN(True, True, 3,input_shape=(32, 32, 3))
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit2 = model.fit(train_images10, train_labels10, epochs=25,batch_size=64, validation_data=(test_images10, test_labels10))
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('cnn_cifar5.out', np.array(result), fmt='%.04f')

    # plt.figure()
    # plt.plot(fit2.history['loss'], label='cnn5_accuracy')
    # plt.plot(fit2.history['val_accuracy'], label = 'cnn5_val_accuracy')
    # plt.plot(fit1.history['loss'], label='cnn_4_accuracy')
    # plt.plot(fit1.history['val_accuracy'], label = 'cnn_4_val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')

    model = CNN(True, False, 5,input_shape=(32, 32, 3))
    model.compile(optimizer = tf.optimizers.Adam(), 
              loss = 'categorical_crossentropy', metrics=metrics)
    fit = model.fit(train_images10, train_labels10, epochs=25,batch_size=64)
    result = model.evaluate(test_images10, test_labels10)
    np.savetxt('cnn_cifar6.out', np.array(result), fmt='%.04f')
   
    plt.show()
