from flask import Flask, render_template, request
import gensim.downloader

from code.main import main
from code.vector_similarity import load_data

app = Flask(__name__)

history = []
pretrained_model = gensim.downloader.load('word2vec-google-news-300')
df, _, _ = load_data()
questions = df.Question
answers = df.Answer


@app.route('/')
def index():
    return render_template('index.html', history=history)


@app.route('/', methods=['POST'])
def process_input():
    user_input = request.form['user_input']
    most_similar_question, answer_r, sentiment = main(user_input, pretrained_model, questions, answers)
    # history.append({'input': user_input, 'output': answer_r[:45]})
    return render_template('index.html', most_similar_question=most_similar_question, answer_r=answer_r,
                           sentiment=sentiment) #, history=history


if __name__ == '__main__':
    app.run()
