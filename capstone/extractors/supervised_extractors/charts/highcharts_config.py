def define_highcharts_config_line():
    CONFIG_LINE = {
        "chart": {
            "plotBackgroundColor": "#D2D7D3",
            "backgroundColor": 'rgba(255, 255, 255, .9)',
            "plotShadow": True,
            "plotBorderWidth": 0,
            "height": 500,
            "spacingRight": 100,
            "renderTo": "container1"
        },
        "tooltip": {
            "shared": True,
            "pointFormat": '<span style="color:{series.color}">{series.name}</span>: <b>{point.y}</b><br/>'
            },
        "xAxis": {
            "gridLineWidth": 0,
            "lineWidth": 0,
            "tickLength": 0,
            "title_text": "Probability threshold"
        },
        "yAxis": {
            "gridLineWidth": 1,
            "title_text": "",
            "max": 1
        },
        "legend": {
            "verticalAlign": "top",
            "align": "left",
            "layout": "vertical",
        }
    }
    return CONFIG_LINE


def define_highcharts_config_pr_line():
    CONFIG_PR_LINE = {
        "chart": {
            "plotBackgroundColor": "#D2D7D3",
            "backgroundColor": 'rgba(255, 255, 255, .9)',
            "plotShadow": True,
            "plotBorderWidth": 0,
            "height": 500,
            "spacingRight": 100,
            "renderTo": "container1"
        },
        "tooltip": {
            "shared": True,
            "pointFormat": '<span style="color:{series.color}">Precision</span>: <b>{point.y}</b><br/> <span style="color:{series.color}">Recall</span>: <b>{point.x}</b><br/>',
            },
        "xAxis": {
            "title_text": "Recall"
        },
        "yAxis": {
            "gridLineWidth": 1,
            "max": 1,
            "title_text": "Precision"
        },
        "legend": {
            "verticalAlign": "top",
            "align": "left",
            "layout": "vertical",
        }
    }
    return CONFIG_PR_LINE


def define_highcharts_config_pie():
    CONFIG_PIE = {
        "chart": {
            "plotBackgroundColor": "#D2D7D3",
            "backgroundColor": 'rgba(255, 255, 255, .9)',
            "plotShadow": True,
            "plotBorderWidth": 0,
            "height": 500,
            "spacingRight": 100,
            "spacingLeft": 300,
            "renderTo": "container2"
        },
        "tooltip": {
            "shared": True,
            "pointFormat": '<span style="color:{series.color}">{series.name}</span>: <b>{point.y}</b><br/>'
            },
        "xAxis": {
            "gridLineWidth": 0,
            "lineWidth": 0,
            "tickLength": 0,
            "title_text": "Probability threshold"
        },
        "yAxis": {
            "gridLineWidth": 1,
            "title_text": "",
            "max": 1
        },
        "legend": {
            "verticalAlign": "top",
            "align": "left",
            "layout": "vertical",
        }
    }
    return CONFIG_PIE


def define_highcharts_config_column():
    CONFIG_COLUMN = {
        "chart": {
            "plotBackgroundColor": "#D2D7D3",
            "backgroundColor": 'rgba(255, 255, 255, .9)',
            "plotShadow": True,
            "plotBorderWidth": 0,
            "height": 500,
            "spacingRight": 100,
            "spacingLeft": 300,
            "renderTo": "container2"
        },
        "tooltip": {
            "shared": True,
            "pointFormat": '<span style="color:{series.color}">{series.name}</span>: <b>{point.y}</b><br/>'
        },
        "xAxis": {
            "categories": ['Women', 'Men', 'Girls', 'Boys', 'Unisex']
        },
        "yAxis": {
            "title_text": "Predicted tags",
            "max": 1
        }
    }
    return CONFIG_COLUMN
