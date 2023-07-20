from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from src.logger import logging
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

from src.food_prediction.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_image_detection_model

df = pd.read_csv("data/food_prediction_cleaned_data.csv")
df1 = pd.read_csv("data/food_prediction_cleaned_data_again2.csv")


application=Flask(__name__)

app=application

## Route for a home page



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
        # recommended_dish_name = "Thayir Semiya Recipe (Curd Semiya)"
        recommended_dish_name = recommended_dish
        name_index = recommendation_df[recommendation_df['name'] == recommended_dish_name].index[0]
        distances = simimlarity[name_index]
        dish_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:4]
        l = []
        for a in dish_list:
            dishes = recommendation_df.iloc[a[0]]
            l.append(dishes[['name','image_url','description']])
            
        dict = {
            "rec": l,
        }

        return render_template('home.html',dict=dict)






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

        print("these are dict values...",pred)

        logging.info("image is identified and returned in the template")
        return render_template('food_recognision.html',pred=pred)












from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
# loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
# loading the model
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

# skipping all the special tokens from the output
def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]
    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)
        for k, v in tokens_map.items():
            text = text.replace(k, v)
        new_texts.append(text)
    return new_texts

# Parameters that control the length of the output
generation_kwargs = {
    "max_length": 512, # The maximum length the generated tokens can have
    "min_length": 64,   
    "no_repeat_ngram_size": 3,
    "do_sample": True,  # Whether or not to use sampling
    "top_k": 60, # model will only consider the top 60 most probable tokens when generating text. default value for top_k is 50 
    "top_p": 0.95  # it will consider all tokens whose cumulative probability is greater than or equal to 10.95.default value for top_p is 1.0
}
special_tokens = tokenizer.all_special_tokens
tokens_map = {"<sep>": "--","<section>": "\n"}


@app.route('/generaterecipe',methods=['GET','POST'])
def generate_recipe():
    if request.method=='GET':
        return render_template('generate_recipe.html')
    else:
        ingredients_list=request.form.get('ingredients')
        ingredients_list = [ingredients_list]
        print("ingredients list--",ingredients_list)

        # ingredients_list = ["chicken, rice, salt, spices, oil, onion"]

        _inputs = ingredients_list if isinstance(ingredients_list, list) else [ingredients_list]
        inputs = ["items :" + inp for inp in _inputs]
        # tokenize the input ingredients 
        inputs = tokenizer(inputs,max_length=256,padding="max_length",truncation=True,return_tensors="jax")

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        # print("input ids--", input_ids)
        # print("attention mask--", attention_mask)

        # use model to generate recipe from tokenized output
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,**generation_kwargs)
        generated = output_ids.sequences
        generated_recipe = target_postprocessing(tokenizer.batch_decode(generated, skip_special_tokens=False),special_tokens)

        split_data = generated_recipe[0].split('\n')
        split_data = [item.strip() for item in split_data if item.strip()]

        # Create three different lists for title, ingredients, and directions
        titles = []
        ingredients = []
        directions = []

        # Loop through the split_data list and populate the three lists accordingly
        for item in split_data:
            if item.startswith('title:'):
                titles.append(item.replace('title:', '').strip())
                recipe_name = titles[0]
            elif item.startswith('ingredients:'):
                ingredients.append(item.replace('ingredients:', '').strip())
                string_with_items = ingredients[0]
                recipe_ingredients = string_with_items.split("-- ")
            elif item.startswith('directions:'):
                directions.append(item.replace('directions:', '').strip())
                string_with_steps  = directions[0]
                full_recipe = string_with_steps.split(".-- ")

        # Print the result
        print("Titles:", recipe_name)
        print("Ingredients:", recipe_ingredients)
        print("Directions:", full_recipe)

        recipe_dict = {
            "status": True,
            "recipe_name": recipe_name,
            "recipe_ingredients": recipe_ingredients,
            "full_recipe": full_recipe,
        }

        return render_template('generate_recipe.html',dict = recipe_dict)

                               





















if __name__=="__main__":
    app.run(host="0.0.0.0")        
