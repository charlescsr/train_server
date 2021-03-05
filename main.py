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
<input type="number">

"""

text_field = """
<input type="text">

"""

float_field = """
<input type="number" step=any>

"""

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.post("/model_set/{model_name}/", response_class=FileResponse)
async def read_item(model_name: str, data: UploadFile = File(...)):
    location = os.environ['MAIN_PATH']
    model = models[model_name]
    async with aiofiles.open(location+data.filename, 'wb') as dataset:
        content = await data.read()
        await dataset.write(content)
    

    df = pd.read_csv(location+data.filename)
    if any(df.columns.str.contains('^Unnamed')):
        df = pd.read_csv(location+data.filename, index_col=0)

    df.dropna(how="any", inplace=True)
    cols = []
    for col in df.columns:
        cols.append(str(col))

    X = df.drop(cols[-1], axis=1)
 
    y = df[cols[-1]]
    
    reg = model.fit(X, y)
    
    pickle.dump(reg, open('model.pkl', 'wb'))
    os.remove(location+data.filename)
    
    return FileResponse(Path('model.pkl'), media_type=".pkl", filename="model.pkl")


@app.post('/create_html/')
async def create_html(data: UploadFile = File(...)):
    location = os.environ['MAIN_PATH']
    path = os.path.join(os.environ['MAIN_PATH'], 'templates')
    os.mkdir(path)
    async with aiofiles.open(location+data.filename, 'wb') as dataset:
        content = await data.read()
        await dataset.write(content)

    html_title_1 = html_title + '\n' + "Main Page" + '\n' + end_block

    html_content_1 = html_content + '\n' + "<h2>Welcome to the generated application.</h2> <br> <h3>Click the button to start.</h3><br><br>" + button_code + '\n' +end_block
    html_1 = html_start + '\n' + html_title_1 + '\n' + html_content_1
    h1 = open(path+"/index.html", 'w')
    h1.write(html_1)
    h1.close()
    df = pd.read_csv(location+data.filename)
    if any(df.columns.str.contains('^Unnamed')):
        df = pd.read_csv(location+data.filename, index_col=0)

    X = df.drop(df.columns[-1], axis=1)
    html_title_2 = html_title + '\n' + "Prediction" + '\n' + end_block
    html_content_2 = html_content + '\n' + form_start 
    for col in X.columns:
        if df[col].dtype == 'int':
            html_content_2 += number_field

        elif df[col].dtype == 'float':
            html_content_2 += float_field

        else:
            html_content_2 += text_field

    html_content_2 += form_end + end_block

    html_2 = html_start + '\n' + html_title_2 + '\n' + html_content_2
    h2 = open(path+"/make_predict.html", 'w')
    h2.write(html_2)
    h2.close()

    html_title_3 = html_title + '\n' + "Answer" + end_block
    html_content_3 = html_content + '\n' + "The " + df.columns[-1] + " is {{{{answer}}}}<br><br> with accuracy of {{{{acc}}}}" + '\n' + end_block
    html_3 = html_start + '\n' + html_title_3 + '\n' + html_content_3
    h3 = open(path+"/predict_ans.html", 'w')
    h3.write(html_3)
    h3.close()
    shutil.make_archive('templates', 'zip', path)
    os.remove(location+data.filename)
    shutil.rmtree(path)

    return FileResponse(Path('templates.zip'), media_type=".zip", filename="templates.zip")