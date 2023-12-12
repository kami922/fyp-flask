from flask import Flask, render_template, send_file
import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Load the CSV file
df_path = "Bitcoin_tweets_dataset_2.csv"
try:
    df = pd.read_csv(df_path, engine="python")
    df.dropna(subset=["hashtags"], inplace=True)
except pd.errors.ParserError as e:
    print(f"Error reading CSV: {e}")

# Get only required columns
df = df[['text']]
df.columns = ['tweets']

# Function to clean the tweets
def clean_tweet(twt):
    twt = re.sub("#bitcoin", 'bitcoin', twt)
    twt = re.sub("#Bitcoin", 'Bitcoin', twt)
    twt = re.sub("#[A-Za-z0-9]+", "", twt)
    twt = re.sub("\\n", "", twt)
    twt = re.sub("https:\/\/S+", "", twt)
    return twt

df['cleaned_tweets'] = df['tweets'].apply(clean_tweet)

# Function to apply subjectivity to tweets
def get_subjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity

# Function to apply polarity to tweets
def get_polarity(twt):
    return TextBlob(twt).sentiment.polarity

# Create 2 columns in dataframe to store subjectivity and polarity
df['subjectivity'] = df['cleaned_tweets'].apply(get_subjectivity)
df['polarity'] = df['cleaned_tweets'].apply(get_polarity)

# Function to get sentiment score
def get_sentiment(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'

df['sentiment'] = df['polarity'].apply(get_sentiment)

# Scatter plot route
@app.route("/scatter_plot")
def scatter_plot():
    plt.figure(figsize=(14, 16))
    for i in range(0, 2000):
        plt.scatter(
            df['polarity'].iloc[[i]].values[0],
            df['subjectivity'].iloc[[i]].values[0],
            color="Purple"
        )
    plt.title("Sentiment Analysis Scatter Plot")
    plt.xlabel("Polarity")
    plt.ylabel("Subjectivity")
    plot_data = io.BytesIO()
    plt.savefig(plot_data, format="png")
    plot_data.seek(0)
    return send_file(plot_data, mimetype="image/png")

# Bar chart route
@app.route("/bar_chart")
def bar_chart():
    plt.figure(figsize=(8, 6))
    df['sentiment'].value_counts().plot(kind="bar")
    plt.title("Sentiment Analysis Bar Chart")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plot_data = io.BytesIO()
    plt.savefig(plot_data, format="png")
    plot_data.seek(0)
    return send_file(plot_data, mimetype="image/png")

# Web app route
@app.route("/")
def read_root():
    return render_template("index.html", len_df=len(df))

if __name__ == "__main__":
    app.run(debug=True)
