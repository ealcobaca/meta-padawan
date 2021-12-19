# Meta-Padawan
# Author: Edesio Alcobaca (ealcobaca)
# MetaDL competition - NIPS 2021
# 

import os
import logging
import csv 
import datetime

import numpy as np

import tensorflow as tf
from tensorflow import keras
import gin

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from metadl.api.api import MetaLearner, Learner, Predictor

class MyMetaLearner(MetaLearner):

    def __init__(self):
        super().__init__()
        print("GPU usage when creating MyMetaLearner...")
        os.system('nvidia-smi')


    def meta_fit(self, meta_dataset_generator) -> Learner:
        """
        Args:
            meta_dataset_generator : a DataGenerator object. We can access 
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.
        
        Returns:
            MyLearner object : a Learner that stores the meta-learner's 
                learning object. (e.g. a neural network trained on meta-train
                episodes)
        """
        return MyLearner()


@gin.configurable
class MyLearner(Learner):
    def __init__(self, 
                model=None,
                img_size=128,
                nclass=5
                ):
        """
        Args:
            model : A keras.Model object describing the Meta-Learner's neural
                network.
            N_ways : Integer, the number of classes to consider at meta-test
                time.
        """
        super().__init__()
        self.img_size = img_size
        self.nclass = nclass

        model_irnv2 = self.getInceptionResNetV2()
        # mdoel_vgg16 = self.getVGG16()
        model_vgg19 = self.getVGG16()
        
        self.model = [[model_vgg19, model_irnv2], None, None]

    def getInceptionResNetV2(self):
        model = keras.applications.InceptionResNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights="imagenet"
            )
        flat1 = keras.layers.Flatten()(model.layers[-1].output)
        model = keras.models.Model(inputs=model.inputs, outputs=flat1)
        
        model.trainable = False

        # model.summary()

        return model

    def getVGG16(self):
        model = keras.applications.vgg19.VGG19(include_top=False, input_shape=(128, 128, 3))
        flat1 = keras.layers.Flatten()(model.layers[-1].output)
        model = keras.models.Model(inputs=model.inputs, outputs=flat1)
        model.trainable = False

        return model

    def getVGG19(self):
        model = keras.applications.vgg19.VGG19(include_top=False, input_shape=(128, 128, 3))
        flat1 = keras.layers.Flatten()(model.layers[-1].output)
        model = keras.models.Model(inputs=model.inputs, outputs=flat1)
        model.trainable = False

        return model
         
    def augds(self, dataset_train):
        aug = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest"
        )

        X = []
        y = []

        naug = 10
        for images, labels in dataset_train:
            for image, label in zip(images, labels):
                    image = np.expand_dims(image, 0)
                    image_aug = aug.flow(image, batch_size=1)

                    aux_image = [image_aug.next() for i in range(naug)] + [image]
                    aux_label = (naug+1)*[label]

                    X += aux_image
                    y += aux_label



        X = tf.concat(X, 0)
        y = tf.concat(y, 0)

        return X, y

    def pipeline_PCA(self, X_train):
        estimators = [
            ('scaler', MinMaxScaler()),
            ('reduce_dim', PCA(n_components=0.95)),
        ]
        pipe = Pipeline(estimators)
        pipe.fit(X_train)
        return pipe
   
    def pipeline(self, X_train, y_train):
        estimators = [
            ('scaler', MinMaxScaler()),
            # ('reduce_dim', PCA(n_components=0.95)),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
            # ('cls', SVC(probability=True, max_iter=1000, random_state=42))
            ]
        # estimators = [
        #     ('rf', RandomForestClassifier(random_state=42))
        # ]
        pipe = Pipeline(estimators)
        pipe.fit(X_train, y_train)
        return pipe

    def fit(self, dataset_train) -> Predictor:
        """Fine-tunes the current model with the support examples of a new 
        unseen task. 

        Args:
            dataset_train : a tf.data.Dataset object. Iterates over the support
                examples. 
        Returns:
            a Predictor object that is initialized with the fine-tuned 
                Learner's neural network weights.
        """

        X, y = self.augds(dataset_train)

        # features from VGG and Resnet
        features = []
        for model in self.model[0]:
            features.append(model.predict(X))

        # features from PCA
        self.model[1] = self.pipeline_PCA(X.numpy().reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])))
        features.append(self.model[1].transform(X.numpy().reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))))

        # comcatenate features
        X_train = np.concatenate(features, axis=1)

        X_train, y_train = shuffle(X_train, y.numpy(), random_state=42)

        # train models
        self.model[2] = self.pipeline(X_train, y_train)
        return MyPredictor(self.model)

    def save(self, model_dir):
        """ Saves the learning object associated to the Learner. It could be 
        a neural network for example. 

        Note : It is mandatory to write a file in model_dir. Otherwise, your 
        code won't be available in the scoring process (and thus it won't be 
        a valid submission).
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        
        # Save a file for the code submission to work correctly.
        with open(os.path.join(model_dir,'dummy_sample.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Dummy example'])
            
    def load(self, model_dir):
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in save().
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))


class MyPredictor(Predictor):
    def __init__(self, model):
        """
        Args: 
            model : a keras.Model object. The fine-tuned neural network
        """
        super().__init__()
        self.model = model

    def predict(self, dataset_test):
        """ Predicts the logits or probabilities over the different classes
        of the query examples.

        Args:
            dataset_test : a tf.data.Dataset object. Iterates over the 
                unlabelled query examples.
        Returns:
            preds : tensors, shape (num_examples, N_ways). We are using the 
                Sparse Categorical Accuracy to evaluate the predictions. Valid 
                tensors can take 2 different forms described below.

        Case 1 : The i-th prediction row contains the i-th example logits.
        Case 2 : The i-th prediction row contains the i-th example 
                probabilities.

        Since in both cases the SparseCategoricalAccuracy behaves the same way,
        i.e. taking the argmax of the row inputs, both forms are valid.

        Note : In the challenge N_ways = 5 at meta-test time.
        """
        # print("testing")

        for images in dataset_test :
            features = []
            for model in self.model[0]:
                features.append(model.predict(images))
            
            X = images[0]
            features.append(self.model[1].transform(X.numpy().reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))))

            X_test = np.concatenate(features, axis=1)
            preds = self.model[2].predict_proba(X_test)
            preds = tf.constant(preds)

        return preds
