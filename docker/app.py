from flask import Flask, request, render_template_string
import string
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and tokenizer
model = load_model('model.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

HTML_FORM = """
<!doctype html>
<html>
<head><title>Spam Classifier Test</title></head>
<body>
  <h1>Test Spam Detection</h1>
  <form method="POST">
    <textarea name="text" rows="4" cols="50" placeholder="Enter message..."></textarea><br>
    <button type="submit">Check</button>
  </form>
{% if result %}
    <div style="background: {{ '#ffcccc' if result.prediction == 'spam' else '#ccffcc' }};
                padding: 10px; margin-top: 20px; border-radius: 5px;">
        <strong>Prediction:</strong> {{ result.prediction|upper }}<br>
        <strong>Confidence:</strong> {{ (result.confidence * 100)|round(1) }}%
        <div style="margin-top: 5px; height: 5px; background: #ddd; width: 100%">
            <div style="height: 100%; width: {{ (result.confidence * 100)|round(1) }}%; 
                        background: {{ '#ff0000' if result.prediction == 'spam' else '#00aa00' }};"></div>
        </div>
    </div>
{% endif %}
</body>
</html> """

def clean_text(text):
    return text.translate(str.maketrans('', '', string.punctuation))

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        
        # Clean and preprocess the text
        cleaned_text = clean_text(text)
        
        # Convert text to sequence and pad
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=500)
        
        # Make prediction
        prediction = model.predict(padded_sequence)[0][0]
        
        result = {
            'prediction': 'spam' if prediction > 0.5 else 'ham',
            'confidence': abs(prediction - 0.5) * 2
        }
    
    return render_template_string(HTML_FORM, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)