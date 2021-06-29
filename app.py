import re
import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
# modelreg = pickle.load(open('C:/Users/HP/Desktop/Snehal/PlacementML/modelreg.pkl','rb'))

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def predict():
    # pred_txt = 'Snehal Gamit'
    form = request.form
    # print(form)
    # for key in form:
    #     print(form[key])
    feature =[]
    for key in form:
        x=form[key]
        feature.append(x)
    # print(feature)
    final_features = [np.array(feature)]
    # feature = final_features.reshape(0,1)
    # print(feature)
    prediction = model.predict(final_features)
    if(prediction == 1):
        # salarypred = modelreg.predict(final_features)
        return render_template('index.html',prediction_text = 'Placed')
    else:
        return render_template('index.html',prediction_text= 'Not Placed')
    


if __name__ == "__main__":
    app.run(debug=True)
