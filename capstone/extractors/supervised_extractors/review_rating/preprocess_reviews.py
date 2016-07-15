import numpy as np
import json
import re
from numpy import nan
from collections import OrderedDict


with open("/Users/ppujari/Downloads/work/yelp_reviews_train.txt", "w") as mtt:
    with open('/Users/ppujari/Downloads/work/yelp_academic_dataset_review.json') as json_data:
        one_cnt = 0;
        two_cnt = 0;
        three_cnt = 0;
        four_cnt = 0;
        five_cnt = 0;
        total_cnt = 0;
        for line in json_data:
            data = json.loads(line)
            review_data = {}
            total_cnt += 1
            try:
                review_data["item_id"] = data['review_id']
                review_data["title"] = data['text']
                review_data["attribute_value"] = str(data['stars'] )
                custom_features = data['votes']
                custom_features["length"] = str(len(data['text']))
                
                review_data["custom_features"] = custom_features

                if review_data["attribute_value"] is "1":
                    one_cnt += 1
                if review_data["attribute_value"] is "2":
                    two_cnt += 1
                if review_data["attribute_value"] is "3":
                    three_cnt = three_cnt + 1
                if review_data["attribute_value"] is "4":
                    four_cnt += 1
                if review_data["attribute_value"] is "5":
                    five_cnt += 1                    
            except:
                print "exception occured"
                continue
            
            mtt.write(json.dumps(OrderedDict(review_data)) + "\n")
        print 'one_cnt:', (one_cnt/float(total_cnt))*100.00
        print 'two_cnt:', (two_cnt/float(total_cnt))*100.00
        print 'three_cnt:', (three_cnt/float(total_cnt))*100.00
        print 'four_cnt:', (four_cnt/float(total_cnt))*100.00
        print 'five_cnt:', (five_cnt/float(total_cnt))*100.00
        print 'Total: ', total_cnt
        
print '----------------'
