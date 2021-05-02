from typing import Optional

from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.responses import FileResponse
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
<a role="button" href="{{url_for('make_predict')}}" style='color:white;'>Head to Prediction</a>

"""

form_start = """
<form align="center" action="{{url_for('predict_ans')}}" method="POST">


"""

form_end = """
<input type="submit" value="Predict">
</form>

"""

number_field = """<input type='number'"""

text_field = """<input type='text'"""

float_field = """<input type='number' step=any"""

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

    html_content_1 = html_content + '\n' + "<div align='center'><h2 style='color:white;'>Welcome to the generated application.</h2> <br> <h3 style='color:white;'>Click the button to start.</h3><br><br>" + button_code + '\n</div>' +end_block
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
            label = "<label for=" + "'{}'".format(str(col)) + " style='color:white;'>" + str(col) + "</label>"
            html_content_2 += label + '\n' + number_field + ' name=' + "'{}'".format(str(col)) + "><br><br><br><br>"

        elif df[col].dtype == 'float':
            label = "<label for=" + "'{}'".format(str(col)) + " style='color:white;'>" + str(col) + "</label>"
            html_content_2 += label + '\n' + float_field + ' name=' + "'{}'".format(str(col)) + "><br><br><br><br>"

        else:
            label = "<label for=" + "'{}'".format(str(col)) + " style='color:white;'>" + str(col) + "</label>"
            html_content_2 += label + '\n' + text_field + ' name=' + "'{}'".format(str(col)) + "><br><br><br><br>"

    html_content_2 += form_end + end_block

    html_2 = html_start + '\n' + html_title_2 + '\n' + html_content_2
    h2 = open(path+"/predict_get.html", 'w')
    h2.write(html_2)
    h2.close()

    html_title_3 = html_title + '\n' + "Answer" + end_block
    html_content_3 = html_content + '\n' + "<div align='center'><h2 style='color:white;'>The " + df.columns[-1] + " is {{answer}}<br><br> with accuracy of {{acc}}%</h2></div>" + '\n' + end_block
    html_3 = html_start + '\n' + html_title_3 + '\n' + html_content_3
    h3 = open(path+"/predict_post.html", 'w')
    h3.write(html_3)
    h3.close()
    shutil.make_archive('templates', 'zip', path)
    os.remove(location+data.filename)
    shutil.rmtree(path)

    return FileResponse(Path('templates.zip'), media_type=".zip", filename="templates.zip")

if __name__ == "__main__":
    uvicorn.run("main:app")
