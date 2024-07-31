from flask import Flask, request, render_template
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)

# Load the VADER Sentiment Intensity Analyzer
with open('sia.pkl', 'rb') as f:
    sia = pickle.load(f)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('tokenizer')
model = AutoModelForSequenceClassification.from_pretrained('model')

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    
    # Get sentiment scores
    roberta_result = polarity_scores_roberta(comment)
    
    # Determine sentiment
    if roberta_result['roberta_pos'] > roberta_result['roberta_neg']:
        sentiment = "Positive"
        score = roberta_result['roberta_pos']
    else:
        sentiment = "Negative"
        score = roberta_result['roberta_neg']

    return render_template('index.html', prediction={'sentiment': sentiment, 'score': score})

if __name__ == '__main__':
    app.run(debug=True)
