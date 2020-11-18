import os
import numpy as np
from PIL import Image

import glob

from datetime import datetime
from flask import Flask, request, render_template
from multiprocessing import Process,Queue
app = Flask(__name__)
from feature_extractor import FeatureExtractor
    #import pickle
fe = FeatureExtractor()
q=Queue()
q1=Queue()
q2=Queue()



# Read image features
def extract_features(q,q1,q2):
    from feature_extractor import FeatureExtractor
    import pickle
    fe = FeatureExtractor()
    features = []
    img_paths = []
    for feature_path in glob.glob("static/feature/*"):
        features.append(pickle.load(open(feature_path, 'rb')))
        img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')
    q.put(features)
    print("features done",features)
    while True:
        if q1.qsize()>0:
            ltemp=q1.get()
            img=ltemp[0]
            query = fe.extract(img)
            print("query done",query)
            q2.put([query])

features=[]
@app.route('/', methods=['GET', 'POST'])
def index():
    
    global features
    first=True
    if request.method == 'POST':
        
        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        #uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
        #img.save(uploaded_img_path)
        img=np.array(img)
        q1.put([img])
        while first:
            if q.qsize()==0:
                
                continue
            else:
                features=q.get()
                features=np.array(features)
                first=False
        while q2.qsize()==0:
            print("2")
            continue 
               
        ltemp=q2.get()
        print(ltemp)
        query=ltemp[0]
        #print(features.shape)
        dists = np.linalg.norm(features - query, axis=1)  # Do search
        # print(dists)
        #return sorted indices of sorted vector.
        ids = np.argsort(dists)[:30] # Top 30 results
        # print(ids)
        result_images = [img_paths[id] for id in ids]

        return render_template('index.html', query_path=uploaded_img_path, result_images = result_images)
    else:
        p=Process(target=extract_features,args=(q,q1,q2),daemon=True)
        p.start()
        print("started")
        return render_template('index.html')

if __name__=="__main__":
    app.run("0.0.0.0")
