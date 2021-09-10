'''
FastAPI server program to train a model and save it to a pickle file.

Along with this it generates the HTML and style files for the web app.

Author: Charles Samuel R

Email: rcharles.samuel99@gmail.com
'''
import shutil
import pickle
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import aiofiles
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()
TOKEN_FOR_STATIC = os.environ['TOKEN_FOR_STATIC']
models = {"lr": LinearRegression(), "lor": LogisticRegression(),
    "nb": GaussianNB(), "knn": KNeighborsClassifier(),
    "dtc": DecisionTreeClassifier(), "rfc": RandomForestClassifier()}


HTML_START = '''
{% extends "base.html" %}
'''

TITLE_START = '''
{% block title %}
'''

END_BLOCK = '''
{% endblock %}
'''

CONTENT_START = '''
{% block content %}
'''

CONTENT_TAG_START = '''
<div class="main-content" >
    <!-- Navbar -->
    <nav class="navbar navbar-top navbar-expand-md navbar-dark" id="navbar-main"  >
        <div class="container-fluid">
            <!-- Brand -->
            <p class="h4 mb-0 text-white text-uppercase d-none d-lg-inline-block">
                Generated App
            </p>
            <!-- Form -->

            <!-- User -->

        </div>
    </nav>
    <!-- End Navbar -->
    <!-- Header -->
    <div class="header bg-gradient-primary pb-8 pt-5 pt-md-8" >
        <div class="container-fluid">
            <div class="header-body">
                <!-- Card stats -->

            </div>
        </div>
    </div>
    <div class="container-fluid mt--7">
        <!-- Form -->
        <div class="row">

            <div class="col-xl-12 order-xl-1">

                <div class="card bg-secondary shadow">

                    <div class="card-header bg-white border-0">
                        <div class="row align-items-center">
                            <div class="col-8">
'''

FORM_START = '''<form action="/result" method="POST" enctype="multipart/form-data">'''

FLOAT_FIELD = '''<h6 class="heading-small text-muted mb-4">{}</h6>
<div class="pl-lg-4">
    <div class="row">
        <div class="col-md-12">
            <div class="form-group">
                <label class="form-control-label" for="{}"></label>
                <input name="{}" class="form-control form-control-alternative" type="number" step=any>
            </div>
        </div>
    </div>
</div>
'''

INT_FIELD = '''<h6 class="heading-small text-muted mb-4">{}</h6>
<div class="pl-lg-4">
    <div class="row">
        <div class="col-md-12">
            <div class="form-group">
                <label class="form-control-label" for="{}"></label>
                <input name="{}" class="form-control form-control-alternative" type="number">
            </div>
        </div>
    </div>
</div>
'''

TEXT_FIELD = '''<h6 class="heading-small text-muted mb-4">{}</h6>
<div class="pl-lg-4">
    <div class="row">
        <div class="col-md-12">
            <div class="form-group">
                <label class="form-control-label" for="{}"></label>
                <input name="{}" class="form-control form-control-alternative" type="text">
            </div>
        </div>
    </div>
</div>
'''

FORM_END = '''<hr class="my-4" />
                    <div class="form-group">
                        <button type="submit" class="btn btn-primary my-4">Predict</button>
                    </div>
                </form>
            </div>
            <div class="col-4 text-right">
            </div>
        </div>
                </div>
            </div>
        </div>
    </div>
    </div>
</div>'''

RESULT_CONTENT = '''
<div class="main-content" >
    <!-- Navbar -->
    <nav class="navbar navbar-top navbar-expand-md navbar-dark" id="navbar-main"  >
        <div class="container-fluid">
            <!-- Brand -->
            <p class="h4 mb-0 text-white text-uppercase d-none d-lg-inline-block">
                Generated App
            </p>
            <!-- Form -->

            <!-- User -->

        </div>
    </nav>
    <!-- End Navbar -->
    <!-- Header -->
    <div class="header bg-gradient-primary pb-8 pt-5 pt-md-8" >
        <div class="container-fluid">
            <div class="header-body">
                <!-- Card stats -->

            </div>
        </div>
    </div>
    <div class="container-fluid mt--7">
        <!-- Form -->
        <div class="row">

            <div class="col-xl-12 order-xl-1">

                <div class="card bg-secondary shadow">

                    <div class="card-header bg-white border-0">
                        <div class="row align-items-center">
                                <div class="pl-lg-4">
                                    <div class="row">
                                        <div class="container" margin="mx-auto d-block" style="padding-left:400px">
                                            <h2 class="heading-medium text-muted mb-2">{} is {} with accuracy of {}%</h2>
                                        </div>
                                    </div>
                                </div>
                            <div class="col-4 text-right">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
'''

@app.get("/ping")
async def ping():
    '''
        Function to ping the server
    '''
    return "pong"

@app.post("/model_set/{model_name}/", response_class=FileResponse)
async def model_set(model_name: str, data: UploadFile = File(...)):
    '''
        Function to train a Machine Learning model based on the dataset given
    '''
    location = os.environ['MAIN_PATH']
    model = models[model_name]
    if os.path.exists("model.pkl"):
        os.remove("model.pkl")

    async with aiofiles.open(location+data.filename, 'wb') as dataset:
        content = await data.read()
        await dataset.write(content)
    dataset.close()

    dataset = pd.read_csv(location+data.filename)
    if any(dataset.columns.str.contains('^Unnamed')):
        dataset = pd.read_csv(location+data.filename, index_col=0)

    dataset.dropna(how="any", inplace=True)
    cols = []
    for col in dataset.columns:
        cols.append(str(col))

    x_var = dataset.drop(cols[-1], axis=1)

    y_var = dataset[cols[-1]]

    reg = model.fit(x_var, y_var)

    with open('model.pkl', 'wb') as pkl_file:
        pickle.dump(reg, pkl_file)

    pkl_file.close()

    os.remove(location+data.filename)

    return FileResponse(Path('model.pkl'), media_type=".pkl", filename="model.pkl")

@app.post('/get-static/{token}')
async def get_static(token: str):
    '''
        Function to get the static files(CSS, JS, etc.) for the web app
    '''
    if token == TOKEN_FOR_STATIC:
        return FileResponse(Path('static.zip'), media_type=".zip", filename="static.zip")

    raise HTTPException(status_code=401, detail="You ain't authorised")

@app.post('/create_html/')
async def create_html(data: UploadFile = File(...)):
    '''
        Function to create the HTML files for the web app
    '''
    location = os.environ['MAIN_PATH']
    if os.path.isfile("templates.zip"):
        os.remove("templates.zip")

    path = os.path.join(os.environ['MAIN_PATH'], 'templates')
    os.mkdir(path)
    async with aiofiles.open(location+data.filename, 'wb') as dataset:
        content = await data.read()
        await dataset.write(content)

    dataset = pd.read_csv(location+data.filename)
    if any(dataset.columns.str.contains('^Unnamed')):
        dataset = pd.read_csv(location+data.filename, index_col=0)

    x_var = dataset.drop(dataset.columns[-1], axis=1)

    html_title_1 = TITLE_START + '\n' + "Prediction" + '\n' + END_BLOCK
    html_title_2 = TITLE_START + '\n' + "Result" + '\n' + END_BLOCK

    html_content_1 = CONTENT_START + '\n' + CONTENT_TAG_START + '\n' + FORM_START

    for col in x_var.columns:
        if dataset[col].dtype == 'int':
            html_content_1 += INT_FIELD.format(col, col, col)

        elif dataset[col].dtype == 'float':
            html_content_1 += FLOAT_FIELD.format(col, col, col)

        else:
            html_content_1 += TEXT_FIELD.format(col, col, col)

    html_content_1 += FORM_END + END_BLOCK
    html_1 = HTML_START + '\n' + html_title_1 + '\n' + html_content_1
    html_file_1 = open(path+"/predict.html", 'w')
    html_file_1.write(html_1)
    html_file_1.close()

    html_content_2 = CONTENT_START + '\n' + RESULT_CONTENT.format(dataset.columns[-1],
        "{{answer}}", "{{acc}}")
    html_2 = HTML_START + '\n' + html_title_2 + '\n' + html_content_2 + '\n' + END_BLOCK
    html_2_file = open(path+"/result.html", 'w')
    html_2_file.write(html_2)
    html_2_file.close()

    shutil.make_archive('templates', 'zip', path)
    os.remove(location+data.filename)
    shutil.rmtree(path)

    return FileResponse(Path('templates.zip'), media_type=".zip", filename="templates.zip")


if __name__ == "__main__":
    uvicorn.run("main:app")
