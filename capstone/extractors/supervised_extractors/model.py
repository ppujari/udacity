import os
import joblib

class Model():
    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.title_corpus = []
        self.description_corpus = []
        self.custom_features = []
        self.class_labels_corpus = []
        self.feature_extractor_titles = None
        self.feature_extractor_description = None
        self.feature_extractor_custom = None
        self.class_names = None
        self.classifier = None
        self.use_custom_features = False
        self.is_multi_label = False

    def get_pickled_files(self):
        class_labels_file = os.path.join(self.model_folder, 'class_labels.pkl')
        title_file = os.path.join(self.model_folder, 'title_file.pkl')
        description_file = os.path.join(self.model_folder, 'description_file.pkl')
        custom_features_file = os.path.join(self.model_folder, 'custom_features_file.pkl')
        title_transform_file = os.path.join(self.model_folder, 'titles_transform.pkl')
        description_transform_file = os.path.join(self.model_folder, 'description_transform.pkl')
        custom_feature_transform_file = os.path.join(self.model_folder, 'custom_feature_transform.pkl')
        class_names_file = os.path.join(self.model_folder, 'class_names.pkl')
        model_file = os.path.join(self.model_folder, 'model.pkl')
        return class_labels_file, title_file, description_file, custom_features_file, title_transform_file, \
               description_transform_file, custom_feature_transform_file, class_names_file, model_file

    def pickle_data(self):
        class_labels_file, title_file, description_file, custom_features_file, _, _, _, _, _ = self.get_pickled_files()

        joblib.dump(self.class_labels_corpus, class_labels_file)
        joblib.dump(self.title_corpus, title_file)
        joblib.dump(self.description_corpus, description_file)
        if self.use_custom_features:
            joblib.dump(self.custom_features, custom_features_file)

    def unpickle_data(self):
        class_labels_file, title_file, description_file, custom_features_file, _, _, _, _, _ = self.get_pickled_files()

        self.class_labels_corpus = joblib.load(class_labels_file)
        self.title_corpus = joblib.load(title_file)
        self.description_corpus = joblib.load(description_file)
        if self.use_custom_features:
            self.custom_features = joblib.load(custom_features_file)

    def pickle_transforms(self):
        _, _, _, _, title_transform_file, description_transform_file, custom_feature_transform_file, _, _ = self.get_pickled_files()

        joblib.dump(self.feature_extractor_titles, title_transform_file)
        joblib.dump(self.feature_extractor_description, description_transform_file)
        if self.use_custom_features:
            joblib.dump(self.feature_extractor_custom, custom_feature_transform_file)

    def unpickle_transforms(self):
        _, _, _, _, title_transform_file, description_transform_file, custom_feature_transform_file, _, _ = self.get_pickled_files()

        self.feature_extractor_titles = joblib.load(title_transform_file)
        self.feature_extractor_description = joblib.load(description_transform_file)
        if self.use_custom_features:
            self.feature_extractor_custom = joblib.load(custom_feature_transform_file)

    def pickle_model(self):
        _, _, _, _, _, _, _, class_names_file, model_file = self.get_pickled_files()
        joblib.dump(self.class_names, class_names_file)
        joblib.dump(self.classifier, model_file)

    def unpickle_model(self):
        _, _, _, _, _, _, _, class_names_file, model_file = self.get_pickled_files()
        self.class_names = joblib.load(class_names_file)
        self.classifier = joblib.load(model_file)