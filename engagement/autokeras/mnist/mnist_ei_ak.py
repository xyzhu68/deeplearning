from autokeras.image.image_supervised import ImageClassifier
import numpy as np
import datetime
import sys

#check arguments
nbArgs = len(sys.argv)
if nbArgs < 2:
    print("Please define drift type")
    exit()
drift_type = sys.argv[1]

hours = 12
if nbArgs > 2:
    hours = int(sys.argv[2])

print("train time limit: {0} hour(s)".format(hours))

beginTime = datetime.datetime.now()

data_file = "ei_data_{0}".format(drift_type)
data = np.load(data_file)

x_train = data["trainEX"]
y_train = data["trainEY"]
x_test = data["testEX"]
y_test = data["testEY"]

clf = ImageClassifier(verbose=True)
clf.fit(x_train, y_train, time_limit=hours * 60 * 60 )
#clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
y = clf.evaluate(x_test, y_test)
print(y)

cls_type = "Ei"
fileOfBestModel = "autokeras_mnist_{0}_{1}_{2}.h5".format(cls_type, drift_type, hours)
#clf.load_searcher().load_best_model().produce_keras_model().save(fileOfBestModel)
clf.export_keras_model(fileOfBestModel)

endTime = datetime.datetime.now()
print(endTime - beginTime)
