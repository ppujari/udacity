import csv
import os
import shutil

from scipy.sparse.construct import hstack
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score, classification_report

from extractors.utils.common_parameters import Parameters
from extractors.utils import log4p
from model import Model
import numpy as np
import json


logger = log4p.get(__name__)

class Trainer():
    def __init__(self, params, load_data=False, random_seed=1, feature_extraction_type='tfidf', use_custom_features=True):
        self.data_file = params.training_data_file
 #       self.data_folder = params.attribute_data_folder
        self.results_folder = params.results_folder
        self.load_cached_data = load_data
        self.random_seed = random_seed
        self.feature_extraction_type = feature_extraction_type
        self.x_train = None
        self.x_test = None
        self.num_features_title = 0
        self.num_features_description = 0
        self.class_labels_train = None
        self.class_labels_test = None
        self.best_estimator = None
        self.metric = params.metric
        self.model = Model(params.model_folder)
        self.model.use_custom_features = use_custom_features

        if params.train:
            self.train_fraction = 1.0
        else:
            self.train_fraction = params.train_fraction

        if os.path.exists(params.model_folder):
            shutil.rmtree(params.model_folder)
        os.mkdir(params.model_folder)


    def __parse_training_data(self):
        logger.info('Using data file: %s' % os.path.abspath(self.data_file))
        with open(self.data_file, 'r') as data_file:
            for line in data_file:
                product_doc = json.loads(line.strip())
                title = product_doc.get('title')
                self.model.title_corpus.append(title)

                description = product_doc.get("description")
                self.model.description_corpus.append(description)

                class_label = product_doc.get("attribute_value").strip()

                self.model.class_labels_corpus.append(class_label)

                if self.model.use_custom_features:
                    custom_features = product_doc.get("custom_features")
                    self.model.custom_features.append(custom_features)

        self.model.title_corpus = np.asarray(self.model.title_corpus)
        if self.model.use_custom_features:
            self.model.custom_features = np.asarray(self.model.custom_features)

        self.model.class_labels_corpus = np.asarray(self.model.class_labels_corpus)


    def __get_train_test_indices(self):
        logger.info("Creating train/test split")
        size_data = len(self.model.class_labels_corpus)
        np.random.seed(self.random_seed)
        indices = np.random.permutation(size_data)
        split_index = int(self.train_fraction * size_data)
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        logger.info("Training data size = %s" % len(train_indices))
        logger.info("Testing data size = %s" % len(test_indices))
        return train_indices, test_indices

    def __get_train_test_matrices_td(self):
        train_indices, test_indices = self.__get_train_test_indices()
        titles_train = self.model.title_corpus[train_indices]
        titles_test = self.model.title_corpus[test_indices]

        if self.model.use_custom_features:
            custom_features_train = self.model.custom_features[train_indices]
            custom_features_test = self.model.custom_features[test_indices]

        logger.info("Generating features")
        self.__get_feature_extractor(data_type='title')
        if self.model.use_custom_features:
            self.model.feature_extractor_custom = DictVectorizer()

        x_titles_train = self.model.feature_extractor_titles.fit_transform(titles_train)
        if self.model.use_custom_features:
            x_custom_features_train = self.model.feature_extractor_custom.fit_transform(custom_features_train)
        self.model.pickle_transforms()

        x_titles_test = self.model.feature_extractor_titles.transform(titles_test)
        if self.model.use_custom_features:
            x_custom_features_test = self.model.feature_extractor_custom.transform(custom_features_test)

        self.num_features_title = x_titles_train.shape[1]

        if self.model.use_custom_features:
            self.x_train = hstack((x_titles_train, x_custom_features_train))
            self.x_test = hstack((x_titles_test, x_custom_features_test))
        else:
            self.x_train = x_titles_train
            self.x_test = x_titles_test

        logger.info("Number of features = %s" % self.x_train.shape[1])

        self.class_labels_train = self.model.class_labels_corpus[train_indices]
        self.class_labels_test = self.model.class_labels_corpus[test_indices]

    def __get_train_only_matrices_td(self):
        titles_train = self.model.title_corpus[:]

        if self.model.use_custom_features:
            custom_features_train = self.model.custom_features[:]

        logger.info("Generating features")
        self.__get_feature_extractor(data_type='title')
        if self.model.use_custom_features:
            self.model.feature_extractor_custom = DictVectorizer()

        x_titles_train = self.model.feature_extractor_titles.fit_transform(titles_train)
        if self.model.use_custom_features:
            x_custom_features_train = self.model.feature_extractor_custom.fit_transform(custom_features_train)
        self.model.pickle_transforms()

        self.num_features_title = x_titles_train.shape[1]

        if self.model.use_custom_features:
            self.x_train = hstack((x_titles_train, x_custom_features_train))
        else:
            self.x_train = x_titles_train

        logger.info("Number of features = %s" % self.x_train.shape[1])

        self.class_labels_train = self.model.class_labels_corpus[:]

    def __get_feature_extractor(self, data_type, feature_type='tfidf', hyper_params={}):
        if feature_type == 'tfidf':
            if len(hyper_params) == 0:
                feature_extractor = TfidfVectorizer(min_df=2, ngram_range=(1,3), stop_words='english', token_pattern=u"(?u)\\b[\\w\\'-]+\\b")
            else:
                feature_extractor = TfidfVectorizer(**hyper_params)
        elif feature_type == 'count':
            if len(hyper_params) == 0:
                feature_extractor = CountVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2), token_pattern=u"(?u)\\b[\\w\\'-]+\\b")
            else:
                feature_extractor = CountVectorizer(**hyper_params)

        if data_type == 'title':
            self.model.feature_extractor_titles = feature_extractor

    def __build_model_object(self, model_name, hyper_params={}):
        if model_name == 'rf':
            self.x_train = self.x_train.toarray()
            self.x_test = self.x_test.toarray()
            if len(hyper_params) == 0:
                clf_base = RandomForestClassifier(verbose=10, n_jobs=8)
                grid = {'n_estimators': [10, 100, 1000],
                        'criterion': ["gini", "entropy"]}
            else:
                clf = RandomForestClassifier(**hyper_params)
        elif model_name == 'svm':
            if len(hyper_params) == 0:
                clf_base = svm.SVC(probability=True)
                grid = {'kernel': ['linear', 'rbf'],
                        'C': 10.0 ** np.arange(-2, 3),
                        'gamma': 10.0 ** np.arange(-2, 3)}
            else:
                clf = svm.SVC(**hyper_params)
        elif model_name == 'nb':
            if len(hyper_params) == 0:
                clf_base = MultinomialNB()
                grid = {'alpha': 0.1 * np.arange(1, 11, 2),
                        'fit_prior': [True, False]}
            else:
                clf = MultinomialNB(**hyper_params)
        elif model_name == 'lr':
            if len(hyper_params) == 0:
                clf_base = LogisticRegression()
                grid = {'C': 10.0 ** np.arange(-2, 3),
                        'penalty': ['l1', 'l2'],
                        'class_weight': [None, 'auto']}
            else:
                clf = LogisticRegression(**hyper_params)
        elif model_name == 'knn':
            if len(hyper_params) == 0:
                clf_base = KNeighborsClassifier()
                grid = {'n_neighbors': [1, 5, 10],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2]}
            else:
                clf = KNeighborsClassifier(**hyper_params)
        elif model_name == 'nc':
            if len(hyper_params) == 0:
                clf_base = NearestCentroid()
                grid = {'metric': ['euclidean', 'manhattan']}
            else:
                clf = NearestCentroid(**hyper_params)
        elif model_name == 'sgd':
            if len(hyper_params) == 0:
                clf_base = SGDClassifier(warm_start=True, shuffle=True)
                grid = {'loss': ['modified_huber', 'squared_hinge', 'perceptron'],
                        'penalty': ['l1', 'l2', 'elasticnet'],
                        'alpha': 10.0 ** np.arange(-4, 3, 2),
                        'n_iter': [10, 100, 1000],
                        'class_weight': [None, 'auto']}
            else:
                clf = SGDClassifier(**hyper_params)
        elif model_name == 'rc':
            if len(hyper_params) == 0:
                clf_base = RidgeClassifier()
                grid = {'normalize': [True, False],
                        'alpha': 10.0 ** np.arange(-4, 3, 2),
                        'class_weight': [None, 'auto']}
            else:
                clf = RidgeClassifier(**hyper_params)
        elif model_name == 'dt':
            if len(hyper_params) == 0:
                clf_base = DecisionTreeClassifier()
                grid = {'criterion': ["gini", "entropy"],
                        'max_features': ['auto', 'sqrt', 'log2'],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 5, 10],
                        'class_weight': [None, 'auto']}
            else:
                clf = DecisionTreeClassifier(**hyper_params)
        elif model_name == 'ab':
            if len(hyper_params) == 0:
                clf_base = AdaBoostClassifier()
                grid = {'n_estimators': [50, 100, 500]}
            else:
                clf = AdaBoostClassifier(**hyper_params)
        else:
            raise Exception("Unsupported algorithm specified")


        if len(hyper_params) == 0:
            cv = KFold(self.x_train.shape[0], n_folds=5, shuffle=True, random_state=0)
            clf = GridSearchCV(clf_base, grid, cv=cv, n_jobs=8, verbose=10, scoring=self.metric)

        self.model.classifier = clf

    def train(self, model_name):
        if self.load_cached_data:
            self.model.unpickle_data()
        else:
            self.__parse_training_data()
            self.model.pickle_data()

        if self.train_fraction == 1.0: #train on all data set
            self.__get_train_only_matrices_td()

            logger.info("Building model")
            self.__build_model_object(model_name, [])

            logger.info(self.x_train.shape)
            logger.info(self.class_labels_train.shape)

            self.model.classifier.fit(self.x_train, self.class_labels_train)

            self.best_estimator = self.model.classifier.best_estimator_
            self.model.class_names = self.model.classifier.best_estimator_.classes_

            logger.info("Best estimator %s" % str(self.best_estimator))

        elif 0 < self.train_fraction < 1.0:  #evaluation 
            self.__get_train_test_matrices_td()

            logger.info("Building model")
            self.__build_model_object(model_name)

            logger.info(self.x_train.shape)
            logger.info(self.class_labels_train.shape)

            logger.info(self.model.classifier)

            self.model.classifier.fit(self.x_train, self.class_labels_train)

            self.best_estimator = self.model.classifier.best_estimator_
            self.model.class_names = self.model.classifier.best_estimator_.classes_

            logger.info("Best estimator %s" % str(self.best_estimator))

        else:
            raise Exception("Invalid train test split")

        self.model.pickle_model()


    def evaluate(self, model_name, attribute_name, show_charts=True):
        if self.train_fraction <= 0 or self.train_fraction >= 1:
            raise Exception("Invalid train test split")

        self.train(model_name)
        class_labels_train_pred = self.model.classifier.predict(self.x_train)
        class_labels_test_pred = self.model.classifier.predict(self.x_test)

        logger.info("Precision on training data = {}".format(precision_score(self.class_labels_train, class_labels_train_pred, average='micro', pos_label=None)))
        logger.info("Precision on test data = {}".format(precision_score(self.class_labels_test, class_labels_test_pred, average='micro', pos_label=None)))

        logger.info("Accuracy on training data = {}".format(accuracy_score(self.class_labels_train, class_labels_train_pred)))
        logger.info("Accuracy on test data = {}".format(accuracy_score(self.class_labels_test, class_labels_test_pred)))

        logger.info("\n Classification Report \n {}".format(classification_report(self.class_labels_test, class_labels_test_pred, target_names=self.model.class_names)))

        if show_charts:
            self.__visualize(attribute_name, model_name, class_labels_test_pred)

        return round(precision_score(self.class_labels_test, class_labels_test_pred, average='micro', pos_label=None), 4), len(class_labels_test_pred)


    def __visualize(self, attribute_name, model_name, class_labels_test_pred):

        class_labels = self.model.classifier.best_estimator_.classes_

        if model_name not in ['knn', 'nc', 'sgd', 'rc']:
            class_labels_prob = self.model.classifier.predict_proba(self.x_test)
            #Lists consisting of probability values for correct/incorrect outcomes
            probabilities_correct = []
            probabilities_incorrect = []
            for i in range(len(self.class_labels_test)):
                if self.class_labels_test[i] == class_labels_test_pred[i]:
                    probabilities_correct.append(class_labels_prob[i][np.where(class_labels == class_labels_test_pred[i])][0])
                else:
                    probabilities_incorrect.append(class_labels_prob[i][np.where(class_labels == class_labels_test_pred[i])][0])

            bin_size = 0.05
            bins = np.arange(0, 1.01, bin_size)
            frequency_counts_correct = np.histogram(probabilities_correct, bins)[0]
            cumulative_frequency_counts_correct = np.cumsum(frequency_counts_correct[::-1])[::-1].astype(np.float32)
            total_correct = float(cumulative_frequency_counts_correct[0])
            frequency_counts_incorrect = np.histogram(probabilities_incorrect, bins)[0]
            cumulative_frequency_counts_incorrect = np.cumsum(frequency_counts_incorrect[::-1])[::-1].astype(np.float32)
            total_incorrect = float(cumulative_frequency_counts_incorrect[0])
            # Overall precision is defined as the fraction of correct predictions (regardless of label) of the predictions which meet a given threshold
            precision_vs_threshold = np.divide(cumulative_frequency_counts_correct,
                                               cumulative_frequency_counts_correct + cumulative_frequency_counts_incorrect)

            total_products = total_correct + total_incorrect

            # Overall recall is defined as the fraction of correct predictions (regardless of label) of all products and meeting a given threshold
            recall = np.divide(cumulative_frequency_counts_correct, total_products)
            products_with_predictions = np.divide(cumulative_frequency_counts_correct + cumulative_frequency_counts_incorrect, total_products)

            precision_data = [[bins[i], precision_vs_threshold[i]] for i in range(len(bins) - 1)]
            recall_data = [[bins[i], recall[i]] for i in range(len(bins) - 1)]
            predicted_products_data = [[bins[i], products_with_predictions[i]] for i in range(len(bins) - 1)]
            precision_recall_data = [[recall[len(bins) - 2 - i], precision_vs_threshold[len(bins) - 2 - i]] for i in range(len(bins) - 1)]

            logger.info("Printing model stats")
            logger.info("Overall precision = %s" % precision_vs_threshold[0])
            thresholds = [0.5, 0.7, 0.9]
            for threshold in thresholds:
                bin_position = int(len(bins) * threshold)
                logger.info("At threshold %s\tPrecision = %s\tRecall = %s" % (threshold, precision_vs_threshold[bin_position], recall[bin_position]))


            from extractors.supervised_extractors.charts.chart_builder import ChartBuilder
            chart_builder = ChartBuilder(os.path.abspath(self.results_folder), attribute_name)

            chart_builder.generate_precision_recall_chart(precision_recall_data)
            chart_builder.generate_precision_fraction_chart(precision_data, recall_data, predicted_products_data)

            class_labels = [str(class_label) for class_label in class_labels]
            conf_matrix = confusion_matrix(class_labels_test_pred, self.class_labels_test, labels=class_labels)

            logger.info('\n%s' % conf_matrix)

            chart_builder.generate_column_chart(conf_matrix, class_labels)

            # Word cloud of top features - meaningful only for linear classifiers
            chart_builder.print_top_features(self.model.feature_extractor_titles, self.model.feature_extractor_description,
                                             [self.num_features_title, self.num_features_description],
                                             self.best_estimator, class_labels)
