import pickle
from flask import Flask, render_template, request
app = Flask(__name__)

# open a file, where you ant to store the data
file = open('model.pkl', 'rb')

clf = pickle.load(file)

# close the file
file.close()


@app.route('/', methods=['GET', 'POST'])
def hello_world():

    if request.method == "POST":

        myDict = request.form
        fever = int(myDict['fever'])
        bodyPain = int(myDict['pain'])
        age = int(myDict['age'])
        runningNose = int(myDict['runningNose'])
        diffBreathing = int(myDict['diffBreathing'])

        # for inference

        inputFeatures = [fever, bodyPain, age, runningNose, diffBreathing]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    # "Hello WOrld " + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)
