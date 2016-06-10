from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import HoverTool
from bokeh.charts import Bar
from bokeh.charts.operations import blend
from bokeh.charts.attributes import cat, color

from extractors.utils import log4p

from tag_cloud_generator import generate_tag_cloud

import numpy as np
import pandas as pd
import webbrowser


logger = log4p.get(__name__)

class ChartBuilder():
    def __init__(self, charts_folder, attribute_name):
        self.attribute_name = attribute_name
        self.charts_folder = charts_folder
        self.width = 1000
        self.height = 700


    def generate_precision_recall_chart(self, precision_recall_data, display=True):
        p = figure(title="Precision vs Recall for {}".format(self.attribute_name), plot_width=self.width, plot_height=self.height)
        p.xaxis.axis_label = "Recall"
        p.yaxis.axis_label = "Precision"
        x, y = zip(*precision_recall_data)
        p.line(x=x, y=y, line_width=3, line_color="#e67e22")
        p.circle(x=x, y=y, size=6, line_color="#c0392b", fill_color="#2ecc71")
        p.background_fill_color = "#ecf0f1"
        hover = HoverTool(
            tooltips = [
                ("Recall", "$x"),
                ("Precision", "$y"),
            ]
            )

        p.add_tools(hover)

        html = file_html(p, CDN, "Precision Recall Curve")
        file_path = "/".join([self.charts_folder, "precision_recall.html"])
        with open(file_path, "w") as b:
            b.write(html)

        if display:
            url = "file://" + file_path
            webbrowser.open(url)

        return html

    def generate_precision_fraction_chart(self, precision_data, recall_data, predicted_products_data=None, display=True):
        p = figure(title="Precision/Recall as a function of probability threshold".format(self.attribute_name),
                   plot_width=self.width, plot_height=self.height)
        p.xaxis.axis_label = "Threshold"
        p.background_fill_color = "#ecf0f1"
        x1, y1 = zip(*precision_data)
        x2, y2 = zip(*recall_data)

        if predicted_products_data is not None:
            x3, y3 = zip(*predicted_products_data)
            p.multi_line(xs = [x1, x2, x3], ys = [y1, y2, y3], color=["#e67e22", "#1abc9c", "#9b59b6"], line_width=3)
            p.circle(x=x3, y=y3, size=6, line_color="#9b59b6", fill_color="#9b59b6", legend="Fraction of products with predictions")
        else:
            p.multi_line(xs = [x1, x2], ys = [y1, y2], color=["#e67e22", "#1abc9c"], line_width=3)
        p.circle(x=x1, y=y1, size=6, line_color="#e67e22", fill_color="#e67e22", legend="Precision")
        p.circle(x=x2, y=y2, size=6, line_color="#1abc9c", fill_color="#1abc9c", legend="Recall")
        p.legend.location = "bottom_left"
        hover = HoverTool(
            tooltips = [
                ("Threshold", "@x"),
                ("Value", "@y"),
            ]
            )

        p.add_tools(hover)

        html = file_html(p, CDN, "Precision, Recall vs Threshold")
        file_path = "/".join([self.charts_folder, "precision__recall_threshold.html"])
        with open(file_path, "w") as b:
            b.write(html)

        if display:
            url = "file://" + file_path
            webbrowser.open(url)


    def generate_column_chart(self, confusion_matrix, labels, display=True):
        confusion_matrix = np.array(confusion_matrix).transpose()
        df = pd.DataFrame(confusion_matrix, columns=labels)
        df = df.divide(df.sum(axis=1), axis='rows')
        df = df.round(4)
        df['labels'] = labels

        bar = Bar(df, values=blend(*labels, name='attributes', labels_name='attribute'),
                  label=cat(columns='labels'),
                  group=cat(columns='attribute'),
                  color=color(columns='attribute'),
                  legend='top_right',
                  tooltips=[('label', '@attribute'), ('value', '@height')],
                  width=self.width,
                  height=self.height,
                  ylabel="Predicted Labels",
                  title="Comparison of predicted vs actual tags")

        html = file_html(bar, CDN, "Confusion Matrix")
        file_path = "/".join([self.charts_folder, "confusion_matrix.html"])
        with open(file_path, "w") as b:
            b.write(html)

        if display:
            url = "file://" + file_path
            webbrowser.open(url)


    def print_top_features(self, titles_vect, desc_vect, shapes, clf, class_labels):
        feature_names_titles = titles_vect.get_feature_names()
        titles_shape = shapes[0]
        desc_shape = shapes[1]

        tags = []

        num_class_labels = len(class_labels)

        try:
            if num_class_labels > 2:
                for i in range(num_class_labels):
                        top10 = np.argsort(clf.coef_[i])[-10:]
                        for j in top10:
                            if 0 < j < titles_shape:
                                tags.append({'color': (46, 204, 113), 'tag': feature_names_titles[j], 'size': 10 * int(clf.coef_[i][j])})
            elif num_class_labels == 2:
                i = 0
                top10 = np.argsort(clf.coef_[i])[-10:]
                for j in top10:
                    if 0 < j < titles_shape:
                        tags.append({'color': (46, 204, 113), 'tag': feature_names_titles[j], 'size': 10 * int(clf.coef_[i][j])})

            generate_tag_cloud(tags, 'cloud.html')
        except:
            logger.info("Word cloud of top features is only generated for linear classifiers")


