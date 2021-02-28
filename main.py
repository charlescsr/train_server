from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()
models = {"lr": LinearRegression(), "lor": LogisticRegression(), "nb": GaussianNB(), "knn": KNeighborsClassifier(), 
    "dtc": DecisionTreeClassifier(), "rfc": RandomForestClassifier()}

class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

@app.get("/model_set/{model_name}")
async def read_item(model_name: str):
    return {"model_name": str(models[model_name])}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}