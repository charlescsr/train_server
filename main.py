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


html_predict = """


{% extends "base.html" %}


{% block title %}


Prediction

{% endblock %}



{% block content %}


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
                                <form action="{{url_for('result')}}" method="POST" enctype="multipart/form-data">
                                    <h6 class="heading-small text-muted mb-4">experience</h6>
                                    <div class="pl-lg-4">
                                        <div class="row">
                                            <div class="col-md-12">
                                                <div class="form-group">
                                                    <label class="form-control-label" for="experience"></label>
                                                    <input name="experience" class="form-control form-control-alternative" type="number" step=any>
                                                </div>
                                            </div>
                                        </div>
                                
                                    </div>
                                    <hr class="my-4" />
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
</div>




{% endblock %}

"""

html_result = """

{% extends "base.html" %}


{% block title %}


Answer
{% endblock %}



{% block content %}



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
                                        <div class="col-md-12">
                                            <h2 class="heading-medium text-muted mb-2">The salary is {{answer}}<br><br> with accuracy of {{acc}}%</h2>
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





{% endblock %}

"""
html_start = '''
{% extends "base.html" %}
'''

title_start = '''
{% block title %}
'''

end_block = '''
{% endblock %}
'''

content_start = '''
{% block content %}
'''

content_tag_start = '''
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

form_start = '''<form action="/result" method="POST" enctype="multipart/form-data">'''

float_field = '''<h6 class="heading-small text-muted mb-4">{}</h6>
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

int_field = '''<h6 class="heading-small text-muted mb-4">{}</h6>
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

text_field = '''<h6 class="heading-small text-muted mb-4">{}</h6>
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

form_end = '''<hr class="my-4" />
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

result_content = '''
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
                                        <div class="col-md-12">
                                            <h2 class="heading-medium text-muted mb-2">{} is {{answer}}<br><br> with accuracy of {{acc}}%</h2>
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

'''
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

    html_content_2 += form_end_1 + end_block

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
'''

@app.post('/create_html_nuvo/')
async def create_html_nuvo(data: UploadFile = File(...)):
    location = os.environ['MAIN_PATH']
    if os.path.isfile("templates.zip"):
        os.remove("templates.zip")

    path = os.path.join(os.environ['MAIN_PATH'], 'templates')
    os.mkdir(path)
    async with aiofiles.open(location+data.filename, 'wb') as dataset:
        content = await data.read()
        await dataset.write(content)

    df = pd.read_csv(location+data.filename)
    if any(df.columns.str.contains('^Unnamed')):
        df = pd.read_csv(location+data.filename, index_col=0)

    X = df.drop(df.columns[-1], axis=1)

    html_title_1 = title_start + '\n' + "Prediction" + '\n' + end_block
    html_title_2 = title_start + '\n' + "Result" + '\n' + end_block

    html_content_1 = content_start + '\n' + content_tag_start + '\n' + form_start

    for col in X.columns:
        if df[col].dtype == 'int':
            html_content_1 += int_field.format(col, col, col)

        elif df[col].dtype == 'float':
            html_content_1 += float_field.format(col, col, col)

        else:
            html_content_1 += text_field.format(col, col, col)

    html_content_1 += form_end + end_block
    html_1 = html_start + '\n' + html_title_1 + '\n' + html_content_1
    h2 = open(path+"/predict.html", 'w')
    h2.write(html_1)
    h2.close()

    html_content_2 = content_start + '\n' + result_content.format(df.columns[-1])
    html_2 = html_start + '\n' + html_title_2 + '\n' + html_content_2
    h3 = open(path+"/result.html", 'w')
    h3.write(html_2)
    h3.close()

    shutil.make_archive('templates', 'zip', path)
    os.remove(location+data.filename)
    shutil.rmtree(path)

    return FileResponse(Path('templates.zip'), media_type=".zip", filename="templates.zip")
    

if __name__ == "__main__":
    uvicorn.run("main:app")
