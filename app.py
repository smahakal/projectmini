from flask import Flask,render_template,request
import numpy as np
import pickle
import project as p
app=Flask(__name__)
model=pickle.load(open('diabetic.pkl','rb'))

@app.route("/",methods=['GET'])
def home():
    return render_template("first.html")
@app.route("/predict",methods=["POST"])
def predict():
    if request.method == "POST":
        PREGNANCIES = float(request.form['PREGNANCIES'])
        GLUCOSE = float(request.form['GLUCOSE'])
        BLOODPRESSURE = float(request.form['BLOODPRESSURE'])
        SKINTHICKNESS = float(request.form['SKINTHICKNESS'])
        INSULIN = float(request.form['INSULIN'])
        DIABETES = float(request.form['DIABETES'])
        AGE = float(request.form['AGE'])

        values = np.array([[PREGNANCIES,GLUCOSE,BLOODPRESSURE,SKINTHICKNESS,INSULIN,DIABETES,AGE]])
        prediction = model.predict(values)




        return render_template("res.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
