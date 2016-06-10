from extractors.utils.PyHighcharts import Highchart
import highcharts_config
import numpy as np
from tag_cloud_generator import generate_tag_cloud
from extractors.utils import log4p


logger = log4p.get(__name__)

class ChartBuilder():
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name
        self.precision_recall_chart = None
        self.precision_fraction_chart = None
        self.column_chart = None
        self.tag_cloud = None

    def generate_precision_recall_chart(self, precision_recall_data):
        precision_chart = Highchart()
        precision_chart.title("Precision vs Recall for {} Tagging".format(self.attribute_name))
        precision_chart.add_data_set(precision_recall_data, series_type="spline", name="Precision vs Recall", color="#e67e22")
        CONFIG_PR_LINE = highcharts_config.define_highcharts_config_pr_line()
        precision_chart.set_options(CONFIG_PR_LINE)
        self.precision_recall_chart = precision_chart

    def generate_precision_fraction_chart(self, precision_data, accepted_products_data, predicted_products_data):
        precision_chart = Highchart()
        precision_chart.title("Precision/Fraction of products as a function of probability threshold")
        precision_chart.add_data_set(precision_data, series_type="spline", name="Precision", color="#e67e22")
        precision_chart.add_data_set(accepted_products_data, series_type="spline", name="Fraction of products correctly tagged", color="#1abc9c")
        precision_chart.add_data_set(predicted_products_data, series_type="spline", name="Fraction of products with predictions", color="#9b59b6")
        CONFIG_LINE = highcharts_config.define_highcharts_config_line()
        precision_chart.set_options(CONFIG_LINE)
        self.precision_fraction_chart = precision_chart

    def generate_column_chart(self, confusion_matrix, labels):
        column_chart = Highchart()
        column_chart.title("Comparison of predicted versus actual tags")
        for index, label in enumerate(labels):
            column_chart_stack = confusion_matrix[:,index]
            column_chart.add_data_set([round(entry, 4) for entry in column_chart_stack], series_type="column", name=label)
        CONFIG_COLUMN = highcharts_config.define_highcharts_config_column()
        CONFIG_COLUMN['xAxis']['categories'] = labels
        column_chart.set_options(CONFIG_COLUMN)
        self.column_chart = column_chart

    def print_top_features(self, titles_vect, desc_vect, shapes, clf, class_labels):
        feature_names_titles = titles_vect.get_feature_names()
        feature_names_desc = desc_vect.get_feature_names()
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
                            elif j < titles_shape + desc_shape:
                                tags.append({'color': (231, 76, 60), 'tag': feature_names_desc[j - titles_shape], 'size': 10 * int(clf.coef_[i][j])})
            elif num_class_labels == 2:
                i = 0
                top10 = np.argsort(clf.coef_[i])[-10:]
                for j in top10:
                    if 0 < j < titles_shape:
                        tags.append({'color': (46, 204, 113), 'tag': feature_names_titles[j], 'size': 10 * int(clf.coef_[i][j])})
                    elif j < titles_shape + desc_shape:
                        tags.append({'color': (231, 76, 60), 'tag': feature_names_desc[j - titles_shape], 'size': 10 * int(clf.coef_[i][j])})

            generate_tag_cloud(tags, 'cloud.html')
        except ValueError:
            logger.info("Word cloud of top features is only generated for linear classifiers")


cb = ChartBuilder("Attr")
cb.generate_precision_recall_chart([[0.956907,0.99473],[0.967047,0.99349],[0.969582,0.992218],[0.973384,0.992248],[0.974651,0.990979],[0.975919,0.989717],[0.975919,0.989717],[0.980989,0.988506],[0.983523,0.986023],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791],[0.984791,0.984791]])
# cb.precision_recall_chart.show()

import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.models import HoverTool, GlyphRenderer

p = figure(title="line", plot_width=800, plot_height=600)
data = [[0.956907,0.99473],[0.967047,0.99349],[0.969582,0.992218],[0.973384,0.992248],[0.974651,0.990979],[0.975919,0.989717],[0.975919,0.989717],[0.980989,0.988506],[0.983523,0.986023],[0.984791,0.984791]]
x, y = zip(*data)
p.line(x=x, y=y, line_width=3, line_color="#e67e22")
p.circle(x=x, y=y, size=6, line_color="#c0392b", fill_color="#2ecc71")
p.background_fill = "#ecf0f1"
hover = HoverTool(
    tooltips = [
        ("Precision", "$x"),
        ("Recall", "$y"),
    ]
    )

p.add_tools(hover)

html = file_html(p, CDN, "facet click distribution")
with open("/Users/amore/Downloads/bokeh.html", "w") as b:
    b.write(html)


