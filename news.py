import requests

API_KEY = "Your NewsAPI key here"
GENERAL_NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
SPORTS_NEWS_URL = f"https://newsapi.org/v2/top-headlines?country=us&category=sports&apiKey={API_KEY}"

def fetch_headlines(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        
        if not articles:
            return []

        headlines = []
        for article in articles[:5]:  # Limit to top 5 articles for brevity
            title = article.get('title', 'No title')
            
            # Find the last dash and reformat the headline
            if ' - ' in title:
                parts = title.rsplit(' - ', 1)
                title = f"From {parts[1]}: {parts[0]}."
                
            headlines.append(title)
        return headlines
    else:
        return []

def get_latest_news():
    general_headlines = fetch_headlines(GENERAL_NEWS_URL)
    return "Here are the top news headlines for today:\n" + "\n".join(general_headlines) if general_headlines else "There are no news updates available for today."

def get_top_sports_news():
    sports_headlines = fetch_headlines(SPORTS_NEWS_URL)
    return "Here are the top sports headlines for today:\n" + "\n".join(sports_headlines) if sports_headlines else "There are no sports news updates available for today."

# Test the functions
if __name__ == "__main__":
    print(get_latest_news())
    print("\n")
    print(get_top_sports_news())
