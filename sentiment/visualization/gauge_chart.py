import plotly.graph_objects as go
from plotly.offline import plot_mpl

from sentiment.training.predictcomment import Sentiment

def plot_guage(probability):
    print(probability)
    probability = probability[0][0]

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"}},
        title = {'text': "Sentiment Analysis"}))

    fig.show()
    #fig.write_image("static\\images\\prob.png")
    #plot_mpl(fig)
    #plot_mpl(fig, image='static\\images\\guage.png')
