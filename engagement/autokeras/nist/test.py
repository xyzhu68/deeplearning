from keras.datasets import mnist
# from autokeras.image.image_supervised import ImageClassifier

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape + (1,))
    print(x_train.shape)
    print(y_train.shape)
    print(y_train)

    # x_test = x_test.reshape(x_test.shape + (1,))

    # clf = ImageClassifier(verbose=True)
    # clf.fit(x_train, y_train, time_limit=1 * 60 * 60)
    # clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    # y = clf.evaluate(x_test, y_test)
    # print(y)