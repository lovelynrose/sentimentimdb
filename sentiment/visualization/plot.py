from sentiment.visualization.gauge_chart import gauge

class Guage:

    def sentimentvalue(self, sent_val):
        if (sent_val <= 1):
            arrow = 1
        elif (sent_val <=2):
            arrow = 2
        elif (sent_val <= 3):
            arrow = 3
        elif (sent_val <=4):
            arrow = 4
        else:
            arrow = 5

        gauge(arrow=arrow, title=('VALUE: {}'.format(sent_val)))