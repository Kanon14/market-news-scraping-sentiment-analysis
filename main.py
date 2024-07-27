from bs4 import BeautifulSoup
import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Define the columns and create the dataframe
columns = ["datetime", "title", "source", "link", "top_sentiment", "sentiment_score"]
df = pd.DataFrame(columns=columns)

# Create a pipeline for the topic sentiment classification
def pipelineMethod(payload):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = classifier(payload)
    return result[0]

counter = 0 # Define the counter

# Run the loop for the news scrapping and execute the topic sentiment classification
for page in range(1, 2):
    url = f"https://markets.businessinsider.com/news/nvda-stock?p={page}"
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "lxml")
    articles = soup.find_all("div", class_="latest-news__story")
    for article in articles:
        datetime = article.find("time", class_="latest-news__date").get("datetime")
        title = article.find("a", class_="news-link").text
        source = article.find("span", class_="latest-news__source").text
        link = article.find("a", class_="news-link").get("href")
        output = pipelineMethod(title)
        top_sentiment = output['label']
        sentiment_score = output['score']

        df = pd.concat([pd.DataFrame([[datetime, title, source, link, top_sentiment, sentiment_score]], columns=df.columns), df], ignore_index=True)
        counter += 1

print(f'{counter} pages of news have been scraped.')
df.to_csv("news.csv")