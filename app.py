from flask import Flask, request, jsonify,render_template,redirect, url_for, session
from flask_cors import CORS
import pickle
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import joblib
import pandas as pd
import numpy as np
from collections import Counter
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)
app.secret_key = "secret_key"
# Load the saved ML model
model = joblib.load(open('Pattern_Categorizer.pkl', 'rb'))

# Function to scrape web data
def scrape_data(url):
    # Automatically manage ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    # Open the webpage
    driver.get(url)

    wait = WebDriverWait(driver, 10)

    # Set the page limit
    page_limit = 5
    page_count = 0
    all_texts = []
    dropdown_locator = (By.CSS_SELECTOR, "select[name='sortFilter']")
    dropdown = wait.until(EC.presence_of_element_located(dropdown_locator))
    # # Re-locate again before interacting to avoid stale element exception
    dropdown = wait.until(EC.element_to_be_clickable(dropdown_locator))
    select = Select(dropdown)
    select.select_by_value("MOST_RECENT")
    time.sleep(2)
    while page_count < page_limit:
        try:
            
            # Wait for the main review container to appear
            parent_divs = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "ZmyHeo")))

            for parent_div in parent_divs:
                child_divs = parent_div.find_elements(By.TAG_NAME, "div")
                if len(child_divs) > 1 and child_divs[1].text.strip():
                    text = child_divs[1].text.strip()  # Extract text from the second div
                    all_texts.append(text)

            # Scroll to the bottom to make the 'Next' button visible
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            try:
                next_button = wait.until(EC.presence_of_element_located(
                    (By.XPATH, "//a[@class='_9QVEpD']/span[text()='Next']")
                ))

                # Wait until the button is clickable
                next_button = wait.until(EC.element_to_be_clickable(
                    (By.XPATH, "//a[@class='_9QVEpD']/span[text()='Next']")
                ))

                next_button.click()
                page_count += 1
                time.sleep(5)  # Allow page to load
            except TimeoutException:
                print("No more pages available or 'Next' button not found.")
                break
        except StaleElementReferenceException:
            print("Page refreshed, retrying the loop...")
            continue  # Restart the loop iteration
        except Exception as e:
            print("Error:", e)
            break

    driver.quit()
    return all_texts

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data["user_input"]
    # data = request.json.get("user_input")
    # print(url)
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    scraped_text = scrape_data(url)
    # print(scraped_text)
    df = pd.DataFrame(scraped_text, columns=["Review"])
    df.dropna(subset=["Review"], inplace=True)
    df = df[df['Review'].str.strip() != '']
    df=df.drop_duplicates()#Remove duplicates
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Define preprocessing function
    def preprocess_text(text):
        text = text.lower()  # Convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        tokens = word_tokenize(text)  # Tokenize text
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
        return ' '.join(tokens)  # Rejoin tokens

    # Apply preprocessing
    df['Processed_Review'] = df['Review'].apply(preprocess_text)
    df=df[df['Processed_Review'].str.strip() != '']
    session['processed_reviews'] = df['Processed_Review'].tolist()
    # print(df['Processed_Review'])
    tfidf_vectorizer =joblib.load("tfidf_vectorizer.pkl")
    tfidf_features = tfidf_vectorizer.transform(df['Processed_Review'])
    # Sentiment Analysis
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity  # Returns value between -1 (negative) and 1 (positive)

    df['Sentiment'] = df['Processed_Review'].apply(get_sentiment)
    # Word Count and Character Count
    df['Word_Count'] = df['Processed_Review'].apply(lambda x: len(x.split()))
    df['Char_Count'] = df['Processed_Review'].apply(len)
    # print(df)
    # Dimensionality Reduction
    svd = joblib.load("truncated_svd.pkl")
    reduced_features = svd.transform(tfidf_features)
    reduced_features = np.nan_to_num(reduced_features, nan=0.0)
    # Convert to DataFrame
    reduced_df = pd.DataFrame(reduced_features, columns=[f'SVD_Component_{i+1}' for i in range(100)])
    # Select Additional Features
    df_select = df[['Sentiment', 'Word_Count', 'Char_Count']]
    model_final_features = pd.concat([df_select, reduced_df], axis=1)
    # Combine all features
    # print(model_final_features.isnull().sum())
    model_final_features.fillna(0, inplace=True)
    # Handle class imbalance using SMOTE
    scaler =joblib.load("scaler_MinMax.pkl")
    X_scaled = scaler.transform(model_final_features)  # Scale first!
    # print(X_scaled)
    # print("Any NaNs in input?", np.isnan(X_scaled).any())
    # print("NaN indices:", np.where(np.isnan(X_scaled)))
    # Load the saved selector
    selector = joblib.load('Select_features.pkl')
    # Apply feature selection (only transform, no fit)
    X_test_selected = selector.transform(X_scaled)
    xgb_model = joblib.load("Pattern_Categorizer.pkl")
    # # Predict using both models
    y_pred_xgb = xgb_model.predict(X_test_selected)
    label_encoder=joblib.load("label_encoder.pkl")
    y_pred=label_encoder.inverse_transform(y_pred_xgb)
    # print(y_pred)
    session['predictions'] = y_pred.tolist()
    return redirect(url_for('result'))
@app.route('/result')
def result():
    predictions = session.get('predictions', [])  # Retrieve predictions
    processed_reviews = session.get('processed_reviews', [])  # Retrieve processed reviews
    return render_template('MainPage.html', data=list(zip(predictions, processed_reviews)))
@app.route('/', methods=['GET'])
def home():
    return render_template('HomePage.html')
if __name__ == '__main__':
    app.run(debug=True)
