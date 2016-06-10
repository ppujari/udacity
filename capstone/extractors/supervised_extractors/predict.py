from model import Model
import numpy as np
from scipy.sparse import hstack

class Predictor():
    def __init__(self, params, use_custom_features=False):
        self.model = Model(params.model_folder)
        self.model.use_custom_features = use_custom_features
        self.is_multi_label = params.is_multi_label
        self.target_pcp_list = params.target_pcp_list
        self.pcp_full_match = params.pcp_full_match
        self.model.unpickle_transforms()
        self.model.unpickle_model()

    def predict(self, product_doc):

        title = product_doc.get("title")
        description = product_doc.get("description")
        if self.model.use_custom_features:
            custom_features = product_doc.get("custom_features")
        if title is None or len(title.strip()) == 0:
            raise ValueError('Title is empty')
        if description is None or len(description.strip()) == 0:
            raise ValueError('Description is empty')

        x_title = self.model.feature_extractor_titles.transform(np.asarray([title]))
        x_description = self.model.feature_extractor_description.transform(np.asarray([description]))
        if self.model.use_custom_features:
            x_custom_features_train = self.model.feature_extractor_custom.transform(custom_features)
            x = hstack((x_title, x_description, x_custom_features_train))
        else:
            x = hstack((x_title, x_description))

        if self.is_multi_label:
            preds = self.model.classifier.predict(x)[0]
            probs = self.model.classifier.predict_proba(x)[0]
            pred_class_indices = np.where(preds == 1)[0].tolist()
            predictions = []
            probabilities = []
            for idx in pred_class_indices:
                predictions.append(self.model.class_names[idx])
                probabilities.append(round(probs[idx], 4))

            return predictions, probabilities
        else:
            prediction = self.model.classifier.predict(x)[0]
            probability = round(self.model.classifier.predict_proba(x)[0][np.where(self.model.class_names == prediction)][0], 4)

            return prediction, probability



