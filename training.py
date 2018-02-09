import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.externals import joblib

emotions = ["happy", "neutral", "sadness", "surprise", "anger"]  # Define emotion order
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clf = SVC(kernel='linear', probability=True, tol=1e-3)  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel

hight = 128  # height of the image
width = 64  # width of the image
hog = cv2.HOGDescriptor()

data = {}  # Make dictionary for all values


def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("sorted_set_cohn\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def get_face(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        print("No Face")
        return None
    return img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            gray = get_face(gray)
            if gray is None:
                continue
            gray = cv2.resize(gray, (width, hight), interpolation=cv2.INTER_CUBIC)  # resize images
            h = hog.compute(gray, winStride=(64, 128), padding=(0, 0))  # storing HOG features as column vector
            h_trans = h.transpose()  # transposing the column vector
            training_data.append(h_trans[0])  # append image array to training data list
            training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            gray = get_face(gray)
            if gray is None:
                continue
            gray = cv2.resize(gray, (width, hight), interpolation=cv2.INTER_CUBIC)  # resize images
            h = hog.compute(gray, winStride=(64, 128), padding=(0, 0))  # storing HOG features as column vector
            h_trans = h.transpose()  # transposing the column vector
            prediction_data.append(h_trans[0])
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


accur_lin = []
for i in range(0, 10):
    print("Making sets %s" % i)  # Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
    print("training SVM linear %s" % i)  # train SVM
    clf.fit(npar_train, training_labels)

    print("getting accuracies %s" % i)  # Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print("linear: ", pred_lin)
    accur_lin.append(pred_lin)  # Store accuracy in a list

print("Mean value lin svm: %s" % np.mean(accur_lin))  # FGet mean accuracy of the 10 runs

joblib.dump(clf, 'AngrySad_trial.pkl')