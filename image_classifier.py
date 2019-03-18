from flask import Flask, request
from flask_restful import Resource, Api
from fastai.vision import load_learner, learner, open_image
import os

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        self.load_model()
        img_fpath = os.path.join("data", "MNIST_example_3.png")
        data = open_image(img_fpath)
        pred_class, pred_idx, outputs = self.model.predict(data)
        return{'pred': str(pred_class)}

    def load_model(self):
        self.model = load_learner("models", "simple_cnn.pkl")


api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)