'''
Created on Aug 13, 2014

@author: rquimon
'''
'''
    REST API methods that are wrappers for the methods in dashboard_backend.operations package
    Naming convention: _service appended to the method name 
'''

from importlib import import_module
import json
import os
import socket
import types
import requests

from flask import Flask, request, Response, redirect, url_for
from flask.helpers import make_response
from flaskext.uploads import UploadSet, configure_uploads

from extractors.sequence_labeling_extractors import predict as seq_label_predict_module
from extractors.supervised_extractors import predict as supervised_predict_module
from extractors.unit import unit_extraction_upload
from extractors.unit.unit_extraction_upload import extract_uom
from extractors.utils import config_manager, get_pkginfo
from extractors.utils import log4p, get_full_filepath
from extractors.utils import pcf, pcf_constants
from extractors.utils.common_parameters import Parameters
from extractors.utils.config_manager import config
from extractors.utils.pcf_constants import *
from rake import Rake

app = Flask(__name__, static_url_path='')
logger = log4p.get(__name__)


p = get_full_filepath('SmartStoplist.txt', app.root_path + '/static'  )
rake = Rake( p )    

global tag_service_url

class Services():
    settings = {'config':{}}
    def __init__(self):
        global tag_service_url
        try:
            self.old_tmp = os.environ.get('TMPDIR', '')
            os.environ['TMPDIR'] = config.defaults().get('crfsuite_tmp_folder')
            if not os.path.exists(os.environ['TMPDIR']):
                os.makedirs(os.environ['TMPDIR'])
            for f in os.listdir(os.environ['TMPDIR']):
                try:
                    os.remove(os.path.join(os.environ['TMPDIR'], f))
                except OSError:
                    pass
            logger.info('Setting TMP directory to: ' + os.environ['TMPDIR'])
            for section in config.sections():
                if config.has_option(section, 'predict_module'):
                    logger.info('Dynamically instantiating: ' + section)
                    predict_module = import_module(config.get(section, 'predict_module'))
                    model_folder = os.path.join(config.get(section, 'model_folder'), section)
                    if os.path.exists(model_folder):
                        # get parameters of this extractor 
                        try:
                            env = config_manager.ENV
                            if env != 'local':
                                env = 'prod'
                            params_file = get_full_filepath('params-%s.json' % env, section)
                            common_params_file = get_full_filepath('common_params-%s.json' % env)
                            params = Parameters(params_file, common_params_file)
                            # overwrite the model_folder parameter with the one from the webapp config
                            setattr(params, 'model_folder', model_folder)
                            with open(params_file, 'r') as f:
                                extractor_params = json.load(f)
                                Services.settings['config'][section] = extractor_params
                        except Exception:
                            logger.warn('Config for %s extractor was not loaded.' % section, exc_info=True)
                            raise
                        if predict_module is seq_label_predict_module:
                            # if it's sequence labeling, instantiate a normalizer
                            normalizer_module = import_module(predict_module.__package__ + '.' + section + '.normalize')
                            normalizer_file = None
                            normalizer_blacklist_file = None
                            if config.has_option(section, 'normalizer_file'):
                                normalizer_file = get_full_filepath(config.get(section, 'normalizer_file'))
                            if config.has_option(section, 'normalizer_blacklist_file'):
                                normalizer_blacklist_file = get_full_filepath(config.get(section, 'normalizer_blacklist_file'))
                            normalizer = normalizer_module.Normalizer(normalizer_file, normalizer_blacklist_file)
                            try:
                                threshold = extractor_params.get(PREDICTION_THRESHOLD)
                                if threshold is None:
                                    threshold = DEFAULT_PROBABILITY
                            except Exception:
                                logger.warn('Config for %s extractor not found, using defaults: [use_custom_features:False, is_multi_label:False]' % section, exc_info=True)
                            predictor = predict_module.Predictor(params, normalizer)
                            setattr(predictor, PREDICTION_THRESHOLD, threshold)
                            setattr(self, section, predictor)
                        else:
                            # read the extractor config to set custom_features/multi_label
                            try:
                                threshold = extractor_params.get(PREDICTION_THRESHOLD)
                                if threshold is None:
                                    threshold = DEFAULT_PROBABILITY
                            except Exception:
                                logger.warn('Config for %s extractor not found, using defaults: [use_custom_features:False, is_multi_label:False]' % section, exc_info=True)
                            predictor = predict_module.Predictor(params)
                            setattr(predictor, PREDICTION_THRESHOLD, threshold)
                            setattr(self, section, predictor)
                    else:
                        logger.error('Model folder for %s does not exist: %s' % (section, model_folder))
            # self.color = color_tagger_service.Service(opts)
            self.unit = unit_extraction_upload.Service()
            self.all = None
            self.get_settings()
            tag_service_url = config.defaults().get("tag_service_url")
            logger.info('Services started')
        finally:
            os.environ['TMPDIR'] = self.old_tmp
            delattr(self, 'old_tmp')
            logger.info('Setting TMP directory back to: ' + os.environ['TMPDIR'])

    def get_settings(self):
        if Services.settings.get('ip') is None:
            Services.settings['ip'] = socket.gethostbyname(socket.gethostname())
            try:
                Services.settings['pkginfo'] = get_pkginfo('productstatus', '/app/attribute-extraction/current/attribute-extraction*.tar.gz')
            except Exception as e:
                logger.error(unicode(e), exc_info=True)
        return Services.settings

services = Services()

def generate_exception(e, output, attribute):
    if isinstance(e, Warning):
        output['status'] = 'OK'
        output['message'] = unicode(e)
        logger.warn(unicode(e))
    else:
        output['status'] = 'INTERNAL SERVER ERROR'
        output['message'] = 'Could not predict %s' % attribute
        output['exception'] = unicode(e)
        logger.error(unicode(e), exc_info=True)

@app.route('/', methods=['GET'])
def root():
    return app.send_static_file('index.html')

@app.route('/reload', methods=['POST', 'GET'])
def reload_services():
    services.__init__()
    output = {}
    output['status'] = 'OK'
    output['message'] = 'Services restarted'
    return Response(json.dumps(output), status=200, mimetype='text/html')

@app.route('/status', methods=['POST', 'GET'])
def status():
    output = {}
    output['status'] = 'OK'
    output['message'] = 'Service is up'
    output['end_points'] = ['/predict/%s' % attribute for attribute in services.__dict__.keys()]
    output['settings'] = services.get_settings()
    return Response(json.dumps(output), status=200, mimetype='application/json')

@app.route('/predict', methods=['POST', 'GET'])
def end_points():
    output = {}
    output['status'] = 'OK'
    output['end_points'] = ['/predict/%s' % attribute for attribute in services.__dict__.keys()]
    return Response(json.dumps(output), status=200, mimetype='application/json')

@app.route('/predict/<attribute>', methods=['POST', 'GET'], strict_slashes=False)
def predict(attribute):
    payload_schema = '''
    Payload Schema:
    [
        {
            "id":"Optional id",
            "title":"Required title"
            "description":"Optional description"
        }, 
        {...}, ...
    ]
    '''
    if attribute == 'unit':
        return redirect(url_for('predict_unit'))
    output = {}
    try:
        if request.method == 'GET' or not request.data:
            return Response(payload_schema, status=200, mimetype='text/html')
        show_exceptions = request.args.get('show_exceptions')
        if show_exceptions and show_exceptions.lower() == 'false':
            show_exceptions = False
        else:
            show_exceptions = True
        batch = True
        inputdata = json.loads(request.data)
        if type(inputdata) is not types.ListType:
            inputdata = [inputdata]
            batch = False
        results = []
        for item in inputdata:            
            res = {}
            if item.get('id'):
                res['id'] = item['id']
            if attribute == 'all':
                whitelist = request.args.get('whitelist')
                if whitelist:
                    whitelist = whitelist.lower().split(',')
                res['predictions'] = {}
                for name in services.__dict__.keys():
                    if name == 'unit' or name == 'all':
                        continue
                    if whitelist and name not in whitelist:
                        continue
                    try:
                        predictions = []
                        predict_obj = get_prediction(name, item)
                        if type(predict_obj.get('prediction')) is types.ListType:
                            for index, prediction in enumerate(predict_obj.get('prediction')):
                                predictions.append({'prediction':prediction, 'probability':predict_obj.get('probability')[index]})
                        else:
                            predictions.append(predict_obj)
                        res['predictions'][name] = predictions
                    except Exception as e:
                        if not show_exceptions:
                            continue
                        res['predictions'][name] = [{'exception':unicode(e)}]
                results.append(res)
            else:
                res.update(get_prediction(attribute, item))
                results.append(res)
        if not batch:
            results = results[0]
        output['result'] = results
        output['message'] = 'Successfully predicted %s' % attribute
        output['status'] = 'OK'
        return Response(json.dumps(output), status=200, mimetype='application/json')
    except Exception as e:
        generate_exception(e, output, attribute)
        return Response(json.dumps(output), status=500, mimetype='application/json')

def get_prediction(attribute, item):
    res = {}
    prod_doc = {}
    prod_doc['title'] = item.get('title')

    try:
        predictor = getattr(services, attribute)
        if isinstance(predictor, supervised_predict_module.Predictor):
#            prediction, probability = predictor.predict(item)
            prediction, probability = predictor.predict(prod_doc)
        else:
            unnormalized_prediction, probability = predictor.predict(prod_doc)
            try:
                prediction = predictor.normalizer.normalize(unnormalized_prediction.lower())
            except KeyError as e:
                res['unnormalized_prediction'] = unnormalized_prediction
            if Services.settings['config'][attribute].get('range_facet'):
                uom = Services.settings['config'][attribute].get('unit_of_measurement')
                prediction = '%s %s' % (prediction, uom)
        if 'unnormalized_prediction' not in res:
            res['prediction'] = prediction
        res['probability'] = probability
    except Exception as e:
        res['error'] = unicode(e)
    return res

# ----------------------------- COLOR ----------------------------- #

@app.route('/color/predict/<pid>', methods=['GET'])
def predict_color_pid(pid):
    output = {}
    try:
        if pid is None or pid.strip() == '':
            output['message'] = 'Product Id is empty'
        else:
            logger.info('Predict color for product id: %s' % pid)
            result = services.color.get_color_prediction(pid).split('\t')
            color = result[1]
            title = result[4]
            if 'ERROR' in color.upper():
                raise Exception(color)
            res = {
                  'Product Id':pid,
                  'Predicted Color':color,
                  'Title': title
                  }
            output['message'] = 'Color successfully predicted'
            output['result'] = res
        output['status'] = 'OK'
        return Response(json.dumps(output), status=200, mimetype='application/json')
    except Exception as e:
        generate_exception(e, output, 'color')
        return Response(json.dumps(output), status=500, mimetype='application/json')

# ----------------------------- UNIT ----------------------------- #

@app.route('/unit/predict/<pid>', methods=['POST', 'GET'])
def predict_unit_pid(pid):
    output = {}
    try:
        logger.info('Predict unit and unit of measure for product id: %s' % pid)
        result = services.unit.get_unit_prediction(pid)
        amount = result[0]
        unit = result[1]
        qty = result[2]
        res = {
              'Product Id':pid,
              'Predicted Unit Of Measure':unit,
              'Predicted Amount': amount,
              'Predicted Quantity': qty
              }
        output['message'] = 'Unit and Unit of measure successfully predicted'
        output['result'] = res
        output['status'] = 'OK'
        return Response(json.dumps(output), status=200, mimetype='application/json')
    except Exception as e:
        generate_exception(e, output, 'unit')
        return Response(json.dumps(output), status=500, mimetype='application/json')


@app.route('/unit/predict', methods=['POST'])
def predict_unit():
    '''
    Payload Schema:
    [
        {
            "id":"<optional>",
            "title":"<Your title here>"
        },
        {...}, ...
    ]
    '''
    output = {}
    try:
        if not request.data:
            raise Exception('Payload is empty')
        inputdata = json.loads(request.data)
        results = []
        for item in inputdata:
            res = {}
            if item.get('title') is not None and item.get('title').strip() != '':
                result = services.unit.get_unit_prediction(pid=None, title=item['title'])
                amount = result[0]
                unit = result[1]
                qty = result[2]
                res = {
                      'Predicted Unit Of Measure':unit,
                      'Predicted Amount': amount,
                      'Predicted Quantity': qty
                      }
            else:
                res['Error'] = 'No title present'
            if item.get('id'):
                res['id'] = item['id']
            results.append(res)
        output['result'] = results
        output['message'] = 'Unit and Unit of measure successfully predicted'
        output['status'] = 'OK'
        return Response(json.dumps(output), status=200, mimetype='application/json')
    except Exception as e:
        generate_exception(e, output, 'unit')
        return Response(json.dumps(output), status=500, mimetype='application/json')

app.config['UPLOADS_DEFAULT_DEST'] = config.get('unit', 'uploaded_file_dir')
csvfile = UploadSet('reports', ('csv',))
configure_uploads(app, (csvfile,))

@app.route('/unit/upload', methods=['POST'])
def upload():
    output = {}
    try:
        if 'csv_file' not in request.files:
            raise Exception('File not uploaded')
        input_file = csvfile.path(csvfile.save(request.files['csv_file']))
        output_filename = os.path.basename(input_file).split('.')[0] + '_results.csv'
        output_file = os.path.join(os.path.dirname(input_file), output_filename)
        extract_uom(input_file, output_file)

        with open(output_file, 'r') as f:
            response = make_response(f.read())
            response.headers['Content-Disposition'] = 'attachment; filename={}'.format(output_filename)
            response.headers['Content-type'] = 'text/csv'
            response.headers['Content-length'] = os.fstat(f.fileno())[6]
        return response
#         output['result'] = {'File uploaded' : input_filename}
#         output['message'] = '%s successfully uploaded' % input_filename
#         output['status'] = 'OK'
#         return Response(json.dumps(output), status=200, mimetype='application/json')
    except Exception as e:
        generate_exception(e, output, 'unit')
        return Response(json.dumps(output), status=500, mimetype='application/json')

# --------------------------------------------------------- Entry point from pipe line client ----------------------------- #
@app.route('/v1/attribute/predict/', methods=['POST'])
def attribute_predict():
    incoming_pcf = json.loads(request.data)
    #logger.debug("incoming PCF: %s" %(json.dumps(incoming_pcf)))
    result = {}
    request_id = None
    attribute = None
    predicted_value = -1.0
    probability = -1.0
    try:
        #This will happen if the incoming PCF is not valid
        try:
            input_pcf = pcf.pcf(incoming_pcf)
            request_id = input_pcf.request_id
            prod_name = input_pcf.prod_name
            if prod_name is None or len(prod_name.strip()) == 0:
                logger.error("PCF title Parse Error: in attribute_predict() Request_id: %s Attribute: %s Error Code : %s Error Message : %s" %(request_id, attribute, PRODUCT_NAME_EMPTY_CODE, PRODUCT_NAME_EMPTY_DESC))
                #input_pcf.set_error(PRODUCT_NAME_EMPTY_CODE, LEVEL_ERROR, PRODUCT_NAME_EMPTY_DESC, ERROR_CATEGORY_DATA)
                return Response(json.dumps(input_pcf._json), status=200, mimetype='application/json')
        except Exception as exc:
                error_code = PCF_PARSE_ERROR_CODE
                logger.error("PCF Parse Error: in attribute_predict() Request_id: %s Attribute: %s Error Code : %s Error Message : %s" %(request_id, attribute, error_code, exc))
#                input_pcf.set_error(error_code, LEVEL_ERROR, exc, ERROR_CATEGORY_DATA)
                return Response(json.dumps(incoming_pcf), status=200, mimetype='application/json')
                
        for attribute in ATTRIBUTE_LIST:
            if input_pcf.is_found(attribute):
                #Call model and print the difference if any, No difference, Do not print. Need more requirement
                continue
                #return Response(json.dumps(input_pcf._json), status=200, mimetype='application/json')
            #start = datetime.now()
            #------------------------------------------------------------------------------------------------------------------------------------------
            #Tag service call here
            try:
                prod_id = input_pcf.product_id
                if prod_id is None:
                    qid = input_pcf.qid
                    payload = {"qid": qid, "attr_id": attribute}
                else:
                    payload = {PRODUCT_ID_KEY: prod_id, "attr_id": attribute}
                r = requests.post(tag_service_url, json=payload)
                logger.info( r.text )
            #if r is error OR empty
                if r.status_code == requests.codes.ok:
                    resp = r.json()
                    if 'tags' in resp:
                        if resp['tags']:
                            tags = resp['tags']['merged'][0]
                            if 'value_id' in tags:
                                v = tags.get("value_id")
                                if attribute in ATTRIBUTE_DISPLAY_NAME:
                                    input_pcf.set_attribute(attribute, ATTRIBUTE_ID[attribute],  v, ATTRIBUTE_DISPLAY_NAME[attribute], 100.00)
                                    return Response(json.dumps(input_pcf._json), status=200, mimetype='application/json')
                        else:
                            logger.info("Bravos response empty tag: in attribute_predict() Request_id: %s Attribute: %s" %(request_id, attribute))
                else:
                    logger.info("Bravos response code: in attribute_predict() Request_id: %s Attribute: %s status_code %s" %(request_id, attribute, r.status_code))
                                
            except Exception as exc:
                logger.info("Bravos Error: in attribute_predict() Request_id: %s Attribute: %s Error Message : %s" %(request_id, attribute, exc))
            #----------------------------------------------------------------------------------------------------------------------------------------------    
            result = extract_attribute(attribute,prod_name)
            #end = datetime.now()
            #elapsed = (end - start )
            #print int(elapsed.total_seconds() * 1000000)
            
            if 'error_code' in result:
                logger.warn("Key Error: in attribute_predict() Request_id: %s Attribute: %s Error Code : %s Error Message : %s" %(request_id, attribute, result['error_code'], result['error_desc']))
                #update msg bag with error code and error description
                #input_pcf.set_error(result['error_code'], result['status'], result['error_desc'], ERROR_CATEGORY_SYSTEM)
                #DO NOT RETURN BACK, there may be other attributes to process
                #return json.dumps(input_pcf._json)
            else:
                probability = result['probability'] 
                predicted_value = result['prediction']
                threshold = result[PREDICTION_THRESHOLD] 
            
                if probability > threshold:
                    if attribute in ATTRIBUTE_DISPLAY_NAME:
                        input_pcf.set_attribute(attribute, ATTRIBUTE_ID[attribute],  predicted_value, ATTRIBUTE_DISPLAY_NAME[attribute], probability)
                #else:
                    #input_pcf.set_error(SCORE_BELOW_THRESHOLD, LEVEL_WARN, attribute + ' : ' + prod_name + " : " + predicted_value, ERROR_CATEGORY_SYSTEM)
                logger.info("Request_id: %s Attribute: %s : prediction : %s : probability: %s : %s" %(request_id, attribute, predicted_value, probability, prod_name))
                    
    except Exception as ex:
        generate_exception(ex, result, attribute)
        logger.info("Error: in attribute_predict() Request_id: %s Attribute: %s Error Code %s Error Message : %s" %(request_id, attribute, result['error_code'], result['error_desc']))
        #update msg bag with error code and description
        if 'exception' in result:
            #input_pcf.set_error(result['message'], result['status'], result['exception'], ERROR_CATEGORY_SYSTEM)
            return Response(json.dumps(input_pcf._json), status=200, mimetype='application/json')
        
    return Response(json.dumps(input_pcf._json), status=200, mimetype='application/json')

def extract_attribute(attribute, title):
    res = {}
    prod_doc = {}
    prod_doc['title'] = title
    try:
        predictor = getattr(services, attribute)
        if isinstance(predictor, supervised_predict_module.Predictor):
            prediction, probability = predictor.predict(prod_doc)
        else:
            #prediction, probability = predictor.predict(title)
            prediction, probability = predictor.predict(prod_doc)
            prediction = predictor.normalizer.normalize(prediction)
        res['prediction'] = prediction
        res['probability'] = probability
        res[PREDICTION_THRESHOLD] = getattr(predictor,PREDICTION_THRESHOLD)
    except ValueError as ve:
        res['error_code'] = ve
        res['error_desc'] = ve
        res['status'] = LEVEL_ERROR
    except KeyError as ke:
        res['error_code'] = ke
        res['error_desc'] = ke
        res['status'] = LEVEL_ERROR
    except Exception as ex:
        res['error_code'] = UNABLE_TO_PREDICT 
        res['error_desc'] = 'Could not predict attribute value :' + attribute
        res['status'] = LEVEL_WARN
        
    return res

@app.route('/v1/rating/predict', methods=['POST', 'GET'], strict_slashes=False)
def predict_rating():
    text = json.loads(request.data)
    review_text = text['review_text']
    if review_text is None or len(review_text.strip()) == 0:
        return Response(json.dumps(text), status=200, mimetype='application/json')
    prod_doc = {}
#    prod_doc['title'] = review_text.split('.', 1 )[0]
    prod_doc['title'] = extract_title_rake(review_text,rake)

    prod_doc['description'] = review_text
    result = extract_attribute_hack('review_rating',prod_doc)
    if 'error_code' in result:
        text['rating'] = 1
        return Response(json.dumps(text), status=200, mimetype='application/json')
    
    text['probability'] = result['probability'] 
    text['rating'] = result['prediction']
    text['title'] = prod_doc['title']
    p_list, n_list = extract_keywords_with_sentiment(review_text, rake)
    text['positive'] = p_list
    text['negative'] = n_list
    
#    from random import randint
#    review_text = json.loads(request.data)
#    review_text['rating'] = randint(1,5) 
    return Response(json.dumps(text), status=200, mimetype='application/json')

def extract_attribute_hack(attribute, prod_doc):
    res = {}

    try:
        predictor = getattr(services, attribute)
        if isinstance(predictor, supervised_predict_module.Predictor):
            prediction, probability = predictor.predict(prod_doc)
        else:
            #prediction, probability = predictor.predict(title)
            prediction, probability = predictor.predict(prod_doc)
            prediction = predictor.normalizer.normalize(prediction)
        res['prediction'] = prediction
        res['probability'] = probability
        res[PREDICTION_THRESHOLD] = getattr(predictor,PREDICTION_THRESHOLD)
    except ValueError as ve:
        res['error_code'] = ve
        res['error_desc'] = ve
        res['status'] = LEVEL_ERROR
    except KeyError as ke:
        res['error_code'] = ke
        res['error_desc'] = ke
        res['status'] = LEVEL_ERROR
    except Exception as ex:
        res['error_code'] = UNABLE_TO_PREDICT 
        res['error_desc'] = 'Could not predict attribute value :' + attribute
        res['status'] = LEVEL_WARN
        
    return res

def extract_title_rake(review, rake):
    keywords = rake.run(review)
    title = ''
    for keyword in keywords:
        if float(keyword[1]) < 0.8:
            continue
        if len(keyword[0].split()) < 2:
            continue
        title = keyword[0]
        break
    if title is '':
        return "This Product"
    return title.lstrip()

def extract_sentiment (keywords):
    keywords_with_sentiment = list()
    pos_keyword_list = list()
    neg_keyword_list = list()
    for keyword in keywords:
        data = "text=" + keyword
        response = requests.post("http://text-processing.com/api/sentiment/", data=data)
        print keyword
        json_response = json.loads(response._content)
        pos_prob = json_response.get("probability").get("pos")
        neg_prob = json_response.get("probability").get("neg")
        neut_prob = json_response.get("probability").get("neutral")
        if float(pos_prob) > 0.6:
            pos_keyword_list.append(keyword)
            continue
        if float(neg_prob) > 0.6:
            neg_keyword_list.append(keyword)
            continue

    return pos_keyword_list, neg_keyword_list

def extract_keywords_with_sentiment(review, rake):
    keywords = rake.run(review)
    meaningful_keywords = list()
    for keyword in keywords:
        if len(keyword[0].split()) >= 2:
            meaningful_keywords.append(keyword[0])

    pos_keyword_list, neg_keyword_list = extract_sentiment(meaningful_keywords)
    return pos_keyword_list, neg_keyword_list


if __name__ == '__main__':
#     app.debug = True
    app.run(host='0.0.0.0',port=5000)
#    app.run(host='172.28.90.191',port=5000)