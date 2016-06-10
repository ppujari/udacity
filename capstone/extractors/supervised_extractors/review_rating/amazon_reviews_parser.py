import zipfile as zip
import json
from collections import OrderedDict


def read_in_chunks(filename, chunk_size=1024):
    with open("/Users/ppujari/hack2016/" + filename, 'rU') as content_file:
        while True:
            content = content_file.read(chunk_size)
            if not content:
                break
            yield content
filename="/Users/ppujari/hack2016/AmazonReviews.zip"
with open("/Users/ppujari/hack2016/amazon_reviews_train.txt", "w") as mtt:
    with zip.ZipFile(filename) as z:
        review_data = {}
        for filename in z.namelist():
            print filename
            s=''
            if filename.endswith('.json'):
                for f in read_in_chunks(filename):
                    s = s + f
#                print s
                reviews = json.loads(s)
                rev_data = reviews["Reviews"]
                for r in rev_data:
                    review_data["item_id"] = r["ReviewID"]
                    review_data["title"] = r["Title"]
                    if not r["Title"]:
                        review_data["title"] = " "
                    review_data["description"] = r["Content"] 
                    if not r["Content"]: 
                        review_data["description"] = " " 
                    review_data["attribute_value"] = r["Overall"]
                    mtt.write(json.dumps(OrderedDict(review_data)) + "\n")