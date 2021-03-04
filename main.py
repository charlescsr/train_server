from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import aiofiles
from pathlib import Path
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import shutil

app = FastAPI()
models = {"lr": LinearRegression(), "lor": LogisticRegression(), "nb": GaussianNB(), "knn": KNeighborsClassifier(), 
    "dtc": DecisionTreeClassifier(), "rfc": RandomForestClassifier()}
dataset_name = None

html_start = """
{% extends "base.html" %}
"""

html_title = """
{% block title %}

"""

end_block = """
{% endblock %}

"""

html_content = """
{% block content %}


"""

button_code = """
<a role="button" href="{{{{url_for('make_predict'}}}}">Head to Prediction</a>

"""

form_start = """
<form align="center" action="{{{{url_for('predict')}}}}" method="POST">


"""

form_end = """
</form>

"""

number_field = """


"""

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.post("/model_set/{model_name}/", response_class=FileResponse)
async def read_item(model_name: str, data: UploadFile = File(...)):
    global dataset_name
    location = os.environ['MAIN_PATH']
    model = models[model_name]
    async with aiofiles.open(location+data.filename, 'wb') as dataset:
        content = await data.read()
        await dataset.write(content)
    
    dataset_name = data.filename

    df = pd.read_csv(location+data.filename)
    df.dropna(how="any", inplace=True)
    cols = []
    for col in df.columns:
        cols.append(str(col))

    X = df.drop(cols[-1], axis=1)
 
    y = df[cols[-1]]
    
    reg = model.fit(X, y)
    
    pickle.dump(reg, open('model.pkl', 'wb'))
    
    return FileResponse(Path('model.pkl'), media_type=".pkl", filename="model.pkl")

@app.post('/create_html')
async def create_html(base_html: UploadFile = File(...)):
    location = os.environ['MAIN_PATH']
    path = os.path.join(os.environ['MAIN_PATH'], 'templates')
    os.mkdir(path)
    async with aiofiles.open(location+base_html.filename, 'w') as html_file:
        content = await base_html.read()
        await html_file.write(content)

    html_title_1 = html_title + '\n' + "Main Page" + '\n' + end_block

    html_content_1 = html_content + '\n' + "<h2>Welcome to the generated application.</h2> <br> <h3>Click the button to start.</h3><br><br>" + button_code + '\n' +end_block
    html_1 = html_start + '\n' + html_title_1 + '\n' + html_content_1
    h1 = open(path+"index.html", 'w')
    h1.write(html_1)
    h1.close()
    df = pd.read_csv(dataset_name)
    X = df.drop(df.columns[-1], axis=1)
 
    


    

    return {"OK"}