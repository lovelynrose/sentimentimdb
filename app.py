import flask
from flask import request, render_template
import tensorflow as tf
from sentiment.visualization.gauge_chart import plot_guage

from sentiment.training.predictcomment import Sentiment

app = flask.Flask(__name__)
app.config["DEBUG"] = True
global graph
graph = tf.get_default_graph()

@app.route('/comment')
def getcomment():
    if 'value' in request.args:
        model = 'imdb//model_fasttext.h5'
        obj = Sentiment(choice='imdb')
        print("request.args : ", request.args)
        comment = str(request.args['value'])
        print("type : ", type(comment))
        with graph.as_default():
            pred, prob =  obj.commentSentiment(model, comment)
        print("pred = ",pred)
        print("prob = ",prob)
        plot_guage(prob)
        return render_template("prediction.html", sentiment=pred, prob=prob)
    else:
        return "Comment not Found"

@app.route('/cloud')
def drawingcloud():
    #CloudDisplay().drawcloud()
    return render_template("prediction.html", name = 'cloud',)

if __name__ == '__main__':
        app.run(threaded=False)
