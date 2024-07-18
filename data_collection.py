import requests
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging

# Step 1: Web Scraping Setup
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

logging.basicConfig(level=logging.INFO)

def collect_reddit_posts(subreddit_name, limit=100):
    url = f'https://www.reddit.com/r/{subreddit_name}/hot/.json?limit={limit}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        logging.error(f"Failed to retrieve data: {response.status_code}")
        return []

    posts = response.json().get('data', {}).get('children', [])
    post_data = []

    for post in posts:
        post_data.append({
            'text': post['data']['title'] + " " + post['data']['selftext'],
            'upvotes': post['data']['score'],
            'comments': post['data']['num_comments'],
            'timestamp': post['data']['created_utc']
        })
    return post_data

# Use a predefined subreddit
subreddit_name = "news"
reddit_data = collect_reddit_posts(subreddit_name, 100)

# Check if data was collected successfully
if not reddit_data:
    logging.error("No data collected. Please check your subreddit name.")
    exit()

# Step 2: Data Preprocessing
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

for post in reddit_data:
    post['processed_text'] = preprocess_text(post['text'])

# Step 3: Feature Extraction
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

for post in reddit_data:
    post['sentiment'] = analyze_sentiment(post['text'])
    post['hour'] = datetime.datetime.fromtimestamp(post['timestamp']).hour  # Extracting hour for temporal pattern analysis
    post['day_of_week'] = datetime.datetime.fromtimestamp(post['timestamp']).weekday()  # Extracting day of the week
    post['post_length'] = len(post['text'])  # Length of the post

# Step 4: Keyword Extraction using TF-IDF
corpus = [post['processed_text'] for post in reddit_data]
vectorizer = TfidfVectorizer(max_features=20)
X_tfidf = vectorizer.fit_transform(corpus)
keywords = vectorizer.get_feature_names_out()
keyword_scores = X_tfidf.sum(axis=0).A1
keyword_score_dict = dict(zip(keywords, keyword_scores))

logging.info(f'Most Common Keywords: {keywords}')

# Step 5: Model Development
X = [[post['sentiment'], len(post['processed_text']), post['hour'], post['day_of_week'], post['post_length']] for post in reddit_data]
y = [post['upvotes'] + post['comments'] for post in reddit_data]

if not X or not y:
    logging.error("No data available for model training.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
logging.info(f'Mean Squared Error: {mse}')
logging.info(f'R-squared: {r2}')

# Step 6: Calculate Engagement Rate
def calculate_engagement_rate(upvotes, comments):
    return (upvotes + comments)

for post in reddit_data:
    post['engagement_rate'] = calculate_engagement_rate(post['upvotes'], post['comments'])

average_engagement_rate = np.mean([post['engagement_rate'] for post in reddit_data])
logging.info(f'Average Engagement Rate: {average_engagement_rate}')

# Calculate Trend Correlation
def calculate_trend_correlation(data, feature):
    feature_values = [post[feature] for post in data]
    engagement_values = [post['upvotes'] + post['comments'] for post in data]
    correlation = np.corrcoef(feature_values, engagement_values)[0, 1]
    return correlation

sentiment_correlation = calculate_trend_correlation(reddit_data, 'sentiment')
logging.info(f'Sentiment Correlation with Engagement: {sentiment_correlation}')

# Save model predictions for further evaluation
results = {
    'predictions': predictions.tolist(),
    'actual': np.array(y_test).tolist(),
    'engagement_rate': average_engagement_rate,
    'sentiment_correlation': sentiment_correlation,
    'mean_squared_error': mse,
    'r_squared': r2,
    'common_keywords': keywords.tolist()
}

with open('model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Visualization
plt.scatter(y_test, predictions)
plt.xlabel("Actual Engagement")
plt.ylabel("Predicted Engagement")
plt.title("Actual vs Predicted Engagement")
plt.show()

# Engagement Rate Distribution
engagement_rates = [post['engagement_rate'] for post in reddit_data]
plt.hist(engagement_rates, bins=20)
plt.xlabel("Engagement Rate")
plt.ylabel("Frequency")
plt.title("Distribution of Engagement Rates")
plt.show()

# Temporal Pattern Analysis
hours = [post['hour'] for post in reddit_data]
plt.hist(hours, bins=24, range=(0, 24))
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Posts")
plt.title("Posts Distribution by Hour of the Day")
plt.show()

days = [post['day_of_week'] for post in reddit_data]
plt.hist(days, bins=7, range=(0, 7))
plt.xlabel("Day of the Week")
plt.ylabel("Number of Posts")
plt.title("Posts Distribution by Day of the Week")
plt.show()

# Keyword Frequency Visualization
plt.figure(figsize=(10, 6))
plt.bar(keyword_score_dict.keys(), keyword_score_dict.values())
plt.xlabel("Keywords")
plt.ylabel("TF-IDF Score")

plt.title("Most Common Keywords")
plt.xticks(rotation=45)
plt.show()
