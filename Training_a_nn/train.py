# USAGE
# python train.py

# import the necessary packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from pyimagesearch import config
import numpy as np
import pickle
import os
import csv

def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
	# open the input file for reading
	f = open(inputPath, "r")

	# loop indefinitely
	while True:
		# initialize our batch of data and labels
		data = []
		labels = []

		# keep looping until we reach our batch size
		while len(data) < bs:
			# attempt to read the next row of the CSV file
			row = f.readline()

			# check to see if the row is empty, indicating we have
			# reached the end of the file
			if row == "":
				# reset the file pointer to the beginning of the file
				# and re-read the row
				f.seek(0)
				row = f.readline()

				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
				if mode == "eval":
					break

			# extract the class label and features from the row
			row = row.strip().split(",")
			label = row[0]
			label = to_categorical(label, num_classes=numClasses)
			features = np.array(row[1:], dtype="float")

			# update the data and label lists
			data.append(features)
			labels.append(label)

		# yield the batch to the calling function
		return (np.array(data), np.array(labels))

# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())

# derive the paths to the training, validation, and testing CSV files
trainPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TRAIN)])
valPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.VAL)])
testPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TEST)])

# determine the total number of images in the training and validation
# sets
totalTrain = sum([1 for l in open(trainPath)])
totalVal = sum([1 for l in open(valPath)])

# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)

# construct the training, validation, and testing generators
trainGen,trainLabel = csv_feature_generator(trainPath, 31010,
	len(config.CLASSES), mode="train")

testGen,testLabel = csv_feature_generator(testPath, 13290,
	len(config.CLASSES), mode="eval")
# define our simple neural network
model = Sequential()
model.add(Dense(6, input_shape=(9,), activation="relu"))
#model.add(Dense(6, activation="relu"))
#model.add(Dense(5, activation="relu"))
#model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="relu"))
# model.add(Dense(3, activation="relu"))
model.add(Dense(len(config.CLASSES), activation="softmax"))

# compile the model
opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3/10)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print(trainGen)
print('\n\n')
print(trainLabel)
print('\n\n')
print(trainGen.shape)
print('\n\n')
print(trainLabel[:,1].shape)
print('\n\n')

# train the network
print("[INFO] training simple network...")
H = model.fit(
	trainGen,
	trainLabel,
	batch_size = 1,
	epochs=10)

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print(testLabel.shape)
import sklearn.metrics
y_pred = model.predict(testGen,batch_size=1)
print(y_pred)

table_end = []
table_end.append(['Actual'] + list(testLabel))
table_end.append(['Neural Network'] + list(y_pred[:,1].round()))

with open('ensemble.csv', 'w+', newline='') as csvfile:
        csv.writer(csvfile).writerows(zip(*table_end))
#predIdxs = np.argmax(predIdxs, axis=1)
from sklearn.metrics import accuracy_score
ac=accuracy_score(testLabel[:,1], y_pred[:,1].round())
print(ac)