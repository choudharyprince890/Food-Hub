from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from src.logger import logging
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

from src.food_prediction.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_image_detection_model

df = pd.read_csv("data/food_prediction_cleaned_data.csv")
df1 = pd.read_csv("data/food_prediction_cleaned_data_again2.csv")
# df1 = pd.read_csv(os.path.join("data","food_prediction_cleaned_data_again2.csv"))


application=Flask(__name__)

app=application

## Route for a home page

# @app.route('/')
# def index():
#     return render_template('index.html') 

# @app.route('/disher',methods=['GET','POST'])
# def show_dish():
#     if request.method=='GET':
#         return render_template('show_dish.html') 


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        cuisines = df1['cuisine'].unique()
        courses = df1['course'].unique()
        diets = df1['diet'].unique()
    

        dict = {
            'cuisines': cuisines,
            'courses': courses,
            'diets': diets
        }

        return render_template('home.html', dict=dict)
    else:
        cuisine=request.form.get('cuisine')
        course=request.form.get('course')
        diet=request.form.get('diet')
        prep_time=int(request.form.get('prep_time'))

        contain_milk=int(request.form.get('contain_milk'))

        contain_curd=int(request.form.get('contain_curd'))

        onion_garlic=int(request.form.get('onion_garlic'))


        # cuisine='Kashmiri'
        # course='Dinner'
        # diet='Non Vegeterian'
        # prep_time=50
        # contain_milk='0'
        # contain_curd='0'
        # onion_garlic='1'

        
        data=CustomData(cuisine,course,diet,prep_time,contain_milk,contain_curd,onion_garlic)
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print(pred_df.info())
        print("this is the dataframe")

        import numpy as np
        a = np.int64(pred_df['prep_time'])
        pred_df['prep_time'] = a.astype('int32')
        
        b = np.int64(pred_df['contain_milk'])
        pred_df['contain_milk'] = b.astype('int32')

        c = np.int64(pred_df['contain_curd'])
        pred_df['contain_curd'] = c.astype('int32')
        
        d = np.int64(pred_df['onion/garlic'])
        pred_df['onion/garlic'] = d.astype('int32')

        print(pred_df.info())
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        print("this is recipe ")
        print(result)

        pd.set_option('max_colwidth', 300)

        image_url = df[df['name']== result[0]]['image_url']


        desc = df[df['name'] == result[0]]['description']
        inst = df[df['name'] == result[0]]['instructions']
        ingr = df[df['name'] == result[0]]['ingredients']


        print("this is the image url -> ")
        print(image_url.iloc[0])
        print(type(image_url))
        print("this is the name of dish ---->>>",result[0])
        logging.info(image_url)
        
        suggestion_dict = {
            'result': result[0],
            'image': image_url.iloc[0],
            'desc': desc.iloc[0],
            'inst': inst.iloc[0],
            'ingr': ingr.iloc[0],
        }
        global recommended_dish
        recommended_dish = result[0]

        return render_template('home.html',dict=suggestion_dict)
        # return render_template('home.html',result=result[0])





# this function fetch the model asnd image path
@app.route('/imagedetect',methods=['GET','POST'])
def food_detect():
    if request.method=='GET':
        return render_template('food_recognision.html')
    else:
        # Get the file from post request
        f = request.files['f_image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        model_path = os.path.join("src/artifcats/food_image_classification","save_at_50.h5")
        pred = load_image_detection_model(model_path,file_path)

        dict = {
            "prediction": pred
        }
        logging.info("image is identified and returned in the template")
        return render_template('food_recognision.html',dict=dict)




@app.route('/showrecommendation',methods=['GET','POST'])
def food_recommendation():
    print("rec")
    if request.method=='GET':
        return render_template('home.html')
    else:
        recommendation_df = pd.read_csv("data/food_recommendation_cleaned_data.csv")
        cv = CountVectorizer(max_features=5000)
        vector = cv.fit_transform(recommendation_df['main_tag']).toarray()
        simimlarity = cosine_similarity(vector)

        # name = dict.result[0]
        # name = "Thayir Semiya Recipe (Curd Semiya)"
        name = recommended_dish
        name_index = recommendation_df[recommendation_df['name'] == name].index[0]
        distances = simimlarity[name_index]
        dish_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:4]
        l = []
        for a in dish_list:
            dishes = recommendation_df.iloc[a[0]]
            l.append(dishes[['name','image_url','description']])
        dict = {
            "rec": l
            # "rec": l[0]
        }
        print("l ->>>>> ",l[0])
        # print("dict---> ",dict)

        return render_template('home.html',dict=dict)










if __name__=="__main__":
    app.run(host="0.0.0.0")        
