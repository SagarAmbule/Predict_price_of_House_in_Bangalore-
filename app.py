from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
pipe=pickle.load(open('RidgeModel.pkl','rb'))
car=pd.read_csv('cleaned_data1.csv')

@app.route('/',methods=['GET','POST'])
def index():
    location=sorted(car['location'].unique())
    area_type=sorted(car['area_type'].unique())
    availability=sorted(car['availability'].unique())
    total_sqft=sorted(list(range(300,10000,300)))
    bhk=sorted(car['bhk'].unique())
    bath = sorted(car['bath'].unique())

    location.insert(0,'Select location')
    area_type.insert(0, 'Select = Plot-Area:1, Carpet-Area:2, Built-up-Area:3, Super-built-up-Area:4')   # Plot-Area:1, Carpet-Area:2, Built-up-Area:3, Super-built-up-Area:4
    availability.insert(0, 'Select-1--For Not ready to Move & select-2--For Ready to Move')
    total_sqft.insert(0, 'Select Total SquareFoot Area Required')
    bhk.insert(0, 'Select Number of BHKs Required')
    bath.insert(0, 'Select Number of Bathrooms you want')
    return render_template('index.html',location=location, area_type=area_type, availability=availability,total_sqft=total_sqft,bhk=bhk,bath=bath)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    area_type = request.form.get('area_type')
    availability = request.form.get('availability')
    location=request.form.get('location')
    total_sqft=request.form.get('total_sqft')
    bath = request.form.get('bath')
    bhk = request.form.get('bhk')

    #prediction=model.predict(pd.DataFrame(columns=['location', 'area_type', 'availability', 'total_sqft', 'bhk','bath'],data=np.array([location,area_type,availability,total_sqft,bhk,bath]).reshape(1, 6)))
    #print(prediction)
    print(area_type,availability,location,total_sqft,bath,bhk)
    input=pd.DataFrame([[area_type,availability,location,total_sqft,bath,bhk]],columns=['area_type','availability','location','total_sqft','bath','bhk'])
    prediction=pipe.predict(input)[0]*1e5
    return str(np.round(prediction,2))



if __name__=='__main__':
    app.run(debug=True)