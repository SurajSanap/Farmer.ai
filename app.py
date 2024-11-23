from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
from keras.models import load_model
import cv2
from werkzeug.utils import secure_filename
from utils.fertilizer import fertilizer_dict
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with your secret key for flash messages
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size is 16 MB

# Load models
classifier = load_model('best_model.keras')
crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict pest
def predict_pest(image):
    try:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))  # Resize to the input shape expected by the model
        img = np.expand_dims(img, axis=0)
        result = classifier.predict(img)
        pest_class = np.argmax(result, axis=-1)[0]
        return pest_class
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Route for pest prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No file part')
        return render_template('error.html')

    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return render_template('error.html')

    if file and allowed_file(file.filename):
        try:
            pest_class = predict_pest(file)
            if pest_class is not None:
                pest_names = [
                    'aphids', 'armyworm', 'beetle', 'bollworm', 'earthworm',
                    'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem borer'
                ]
                pest_identified = pest_names[pest_class]
                return render_template(f'{pest_identified}.html', pred=pest_identified)
            else:
                flash('Error predicting pest. Please try again.')
        except Exception as e:
            print(f"Exception during prediction: {e}")
            flash('Error predicting pest. Please try again.')
    else:
        flash('File format is not appropriate. Kindly upload an image file.')
    return render_template('error.html')

# Route for fertilizer recommendation
@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():
    try:
        crop_name = request.form['cropname']
        N_filled = int(request.form['nitrogen'])
        P_filled = int(request.form['phosphorous'])
        K_filled = int(request.form['potassium'])

        df = pd.read_csv('Data/Crop_NPK.csv')
        crop_data = df[df['Crop'] == crop_name]

        if not crop_data.empty:
            N_desired = crop_data['N'].iloc[0]
            P_desired = crop_data['P'].iloc[0]
            K_desired = crop_data['K'].iloc[0]

            n = N_desired - N_filled
            p = P_desired - P_filled
            k = K_desired - K_filled

            key1 = "NHigh" if n < 0 else "Nlow" if n > 0 else "NNo"
            key2 = "PHigh" if p < 0 else "Plow" if p > 0 else "PNo"
            key3 = "KHigh" if k < 0 else "Klow" if k > 0 else "KNo"

            response1 = fertilizer_dict[key1]
            response2 = fertilizer_dict[key2]
            response3 = fertilizer_dict[key3]

            return render_template('Fertilizer-Result.html', recommendation1=response1,
                                   recommendation2=response2, recommendation3=response3,
                                   diff_n=abs(n), diff_p=abs(p), diff_k=abs(k))
        else:
            return render_template('error.html', message='Crop not found')
    except Exception as e:
        print(f"Exception during fertilizer recommendation: {e}")
        flash('Error in processing request. Please check your inputs and try again.')
        return render_template('error.html')

# Route for crop recommendation
@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    try:
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return render_template('crop-result.html', prediction=final_prediction, pred=f'img/crop/{final_prediction}.jpg')
    except Exception as e:
        print(f"Exception during crop prediction: {e}")
        flash('Error in processing request. Please check your inputs and try again.')
        return render_template('error.html')

# Routes for main pages
@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/chatbot.html")
def chatbot():
    return render_template("chatbot.html")

if __name__ == '__main__':
    app.run(debug=True)
