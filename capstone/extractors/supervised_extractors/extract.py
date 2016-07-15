import csv
import sys

from extractors.supervised_extractors.train import Trainer
from extractors.supervised_extractors.predict import Predictor

from extractors.utils import log4p

logger = log4p.get(__name__)


def predict_tags(line, param_obj, predictor):
    with open(param_obj.results_file, 'a') as results_file:
        writer = csv.writer(results_file, delimiter='\t')
        title = line['title']
        description = line['description']
        item_id = line['item_id']
        try:
            prediction, probability = predictor.predict({"title": title, "description": description})
        except ValueError:
            return

        title = title.encode("ascii", "ignore").decode("ascii")
        logger.info("Title: {}".format(title))
        logger.info((prediction, probability))
        if float(probability) > param_obj.threshold and prediction not in param_obj.ignore_tags:
            writer.writerow([item_id, title, prediction, probability])

#Use this method for single label attributes
def extract(params):

    if params.evaluate and params.train:
        raise Exception("Both evaluate and train params set to true - choose one")

    if params.evaluate:
        trainer = Trainer(params)
        trainer.evaluate(params.algo, params.attribute_name)
        sys.exit("Done")
    if params.train:
        trainer = Trainer(params)
        trainer.train(params.algo)
        sys.exit("Done")

    predictor = Predictor(params)

    with open(params.rse_path) as rse_file:
        for line in rse_file:
            predict_tags(line, params, predictor)