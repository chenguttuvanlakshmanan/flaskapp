# importing needed modules
from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
#import pickle
from model import model

#intializing flask app
app = Flask(__name__)
api = Api(app)

def model_build(data):
    model = pickle.load(open("model.pkl","rb"))
    ypred = model.predict(data)
    return str(ypred)

#--------------------------------------------------------------------------------------------
# Model got from pickle file
#----------------------------------------------------------------------------------------------  
class Topredict(Resource):
    def post(self):
        data = pd.DataFrame(request.get_json())
        # build the model and insert in to db
        status = model_build(data)
        return { "result" : status}

    def get(self):
        return "Success"

# api end point
api.add_resource(Topredict,'/predict')

if __name__ == '__main__':
    app.run(debug=False)