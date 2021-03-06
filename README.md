# Train Server

FastAPI server to help in training the ML model and generate the HTML templates for Flask application over [here](https://github.com/charlescsr/implogn-visintei-dating) 

## Procedure:

* The Flask application first pings the server with the dataset.
* The server trains the dataset based on the model specified by the user
* The server then pickles the model and sends it back to the Flask app
* The server then sends a request to generate the HTML file for the application.
* It generates the HTML based on the data given to it.
* Finally it zips up the HTML templates and sends it back to the user.

## Setup

To set up this project, you require the following:

* Python (Preferably 3.8.x)
* Pipenv
  * Can be installed with ```python -m pip install pipenv```

Once all this is set up, just run:

```
$ pipenv shell
$ pipenv install
```

Finally, the server can be run by typing this:
```uvicorn main:app --reload```

Putting reload is up to you.
