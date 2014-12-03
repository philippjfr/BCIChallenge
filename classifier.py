import pickle
import numpy as np
import mlpy
import scipy
from scipy import signal

import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score

import param

import holoviews
from holoviews.ipython.widgets import ProgressBar
from holoviews.interface.collector import AttrTree

class KnnDtw(param.Parameterized):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)

    """

    n_neighbors = param.Integer(default=50, doc="""
        Number of neighbors to use by default for KNN""")

    progress_bar = param.Boolean(default=False)

    def __init__(self, **params):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """
        self.cache = None
        self.x_prev = None
        super(KnnDtw, self).__init__(**params)

    def get_params(self, deep=False):
        return dict(self.get_param_values(onlychanged=not deep))

    def fit(self, x, l):
        self.x=x
        self.l=l


    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0
        x_s = np.shape(x)
        y_s = np.shape(y)
        dm = np.zeros((x_s[0], y_s[0]))
        dm_size = x_s[0]*y_s[0]

        total = dm_size
        p = ProgressBar()

        for i in xrange(0, x_s[0]):
            for j in xrange(0, y_s[0]):
                dm[i, j] = mlpy.dtw_std(x[i, :],
                                        y[j, :], dist_only=True)
                # Update progress bar
                dm_count += 1
            if self.progress_bar:
                p(float(dm_count)/total*100)

        return dm


    def predict(self, x, n_neighbors=1, weighted=False, train=False):
        """Predict the class labels or probability estimates for
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels
              (2) the knn label count probability
        """
        if np.array_equal(x, self.x_prev):
            dm = self.cache
        else:
            dm = self._dist_matrix(x, self.x)
            self.cache = dm
            self.x_prev = x

        # Identify the k nearest neighbors
        start = 1 if train else 0
        if train:
            knn_idx = dm.argsort()[:, 1:n_neighbors+1]
        else:
            knn_idx = dm.argsort()[:, :n_neighbors]
        # Identify k nearest labels
        knn_labels = self.l[knn_idx]
        if weighted:
            knn_weights = dm[dm.argsort()][:, start:n_neighbors]
            knn_weights = (1/knn_weights)
            knn_weights = knn_weights/knn_weights.max()
            mode_data = knn_labels * knn_weights
        else:
            # Model Label
            mode_data = np.mean(knn_labels, axis=1)
        mode_label = np.round(mode_data)

        return mode_label.ravel(), mode_data.ravel()


def get_data(event_map, splitter, time_window=(200., 300.), sample_rate=200, channel='TP8', dc=False, filter=False):
    x_train, y_train = [], []
    x_test, y_test = [], []
    labelled = 'Classification' in event_map.dimension_labels
    start, stop = float(time_window[0])/1000, float(time_window[1])/1000
    for (k, sample), train in zip(event_map.items(), splitter):
        if labelled:
            label = k[3]

        # Design and apply the bandpass filter
        if filter:
            a, b = signal.iirfilter(3, [0.5/(sample_rate/2.0), 100/(sample_rate/2.0)])
            sample.data[channel] = signal.filtfilt(a, b, np.array(sample.data[channel]), axis=0)

        baseline = np.array(sample.data[(0 <= sample.data['Time']) & (sample.data['Time'] < 100)][channel]).mean()
        data = np.array(sample.data[(start <= sample.data['Time']) & (sample.data['Time'] < stop)][channel])
        data = data if dc else data-baseline
        if train:
            x_train.append(data)
            if labelled:
                y_train.append(label)
        else:
            x_test.append(data)
            if labelled:
                y_test.append(label)
    return np.vstack(x_train), np.array(y_train), np.vstack(x_test) if len(x_test) else x_test, np.array(y_test)

def scorer(classifier, x, y):
    label, proba = classifier.predict(x, n_neighbors=50)
    return roc_auc_score(y, proba)

def classify(channel, (start, stop), cross_val=True, full_train=True, predict=True):
    data = AttrTree()

    with open('train_map_long.pkl', 'rb') as f:
        train_map = pickle.load(f)
    with open('test_map_long.pkl', 'rb') as f:
        test_map = pickle.load(f)

    splitter = np.ones(len(train_map))
    print "Starting channel %s, time window from %s ms to %s ms." % (channel, str(start), str(stop))
    x_train, y_train, _, _ = get_data(train_map, splitter, channel=channel, time_window=(start, stop), filter=False)
    x_test, _, _, _ = get_data(test_map, splitter, channel=channel, time_window=(start, stop))
    classifier = KnnDtw(progress_bar=False)
    data.Classifier = classifier
    classifier.fit(x_train, y_train)
    if cross_val:
        data.CV_score = cross_val_score(classifier, x_train, y_train, scoring=scorer, cv=3, n_jobs=-1)
        print "CV score: ", data.CV_score.mean()
    if full_train:
        scores, train_probas = [], []
        nn = [1, 10, 50, 100, 200, 300, 500]
        for i in nn:
            label, train_proba = classifier.predict(x_train, n_neighbors=i, train=True)
            scores.append(roc_auc_score(y_train, train_proba))
            train_probas.append(train_proba)
        data.Train.Scores = holoviews.Curve(zip(nn, scores), value='ROC Score', label=channel)
        data.Train.Probas = train_probas
        print "Training performance for 1: %f, 10: %f, 50: %f, 100: %f, 200: %f, 300: %f and 500: %f neighbors" % tuple(scores)
    if predict:
        print "Predicting test data"
        label, proba = classifier.predict(x_test, n_neighbors=50)
        data.Test.Labels = label
        data.Test.Probas = proba
        print "---------------------------------\n\n"

    return data