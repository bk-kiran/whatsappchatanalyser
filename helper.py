from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import emoji
import pandas as pd


def fetchstats(user, df):
    if user != 'All Users':
        df = df[df['user'] == user]  # fetching for a particular user

    # fetching total number of messages
    total_messages = df.shape[0]

    # fetching number of words
    words = []
    for message in df['message']:
        words.extend(message.split())
    total_words = len(words)

    # fetching number of shared media
    total_media = 0
    for message in df['message']:
        if 'image omitted' in message or 'video omitted' in message or 'document omitted' in message or 'sticker omitted' in message or 'audio omitted' in message or 'message omitted' in message or 'card omitted' in message:
            total_media = total_media + 1

    # fethcing number of links
    total_links = 0
    for message in df['message']:
        if 'http' in message or '.com' in message:
            total_links = total_links + 1

    return total_messages, total_words, total_media, total_links


# finding which user is the most active
def busiest_users(df):
    data = df['user'].value_counts().head()
    df = round((df['user']).value_counts() / df.shape[0] * 100, 1).reset_index(
    ).rename(columns={'index': 'name', 'user': 'user', 'count': 'percent'})

    return data, df

# finding the most common words


def word_cloud(user, df):
    if user != 'All Users':
        df = df[df['user'] == user]  # fetching for a particular user

    stop_words = ["image", "omitted", "video",
                  "audio", "sticker", "message", "deleted"] + list(STOPWORDS)

    cloud = WordCloud(width=500, height=500, min_font_size=10,
                      background_color='white', stopwords=stop_words)
    df_cloud = cloud.generate(df['message'].str.cat(sep=" "))

    return df_cloud


# finding the most used emojis
def emoji_counter(user, df):
    if user != 'All Users':
        df = df[df['user'] == user]  # fetching for a particular user

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


# creating the yearly timeline linegraph
def year_timeline(user, df):
    if user != 'All Users':
        df = df[df['user'] == user]  # fetching for a particular user

    timeline = df.groupby(['year', 'month_num', 'month']).count()[
        'message'].reset_index()
    times = []
    for i in range(timeline.shape[0]):
        times.append(str(timeline['year'][i]))

    timeline['times'] = times

    return timeline


# finding the message count for each day of the week
def weekly_message_count(user, df):
    if user != 'All Users':
        df = df[df['user'] == user]

    weekly_count = df['day_name'].value_counts()

    return weekly_count


# finding the message count for each day of the week
def monthly_message_count(user, df):
    if user != 'All Users':
        df = df[df['user'] == user]

    monthly_count = df['month'].value_counts()

    return monthly_count


# creating a message heatmap
def heatmap(user, df):
    if user != 'All Users':
        df = df[df['user'] == user]

    activity_heatmap = df.pivot_table(
        index="day_name", columns="period", values="message", aggfunc="count").fillna(0)

    return activity_heatmap

# sentiment


def sentiment_count(user, df):
    if user != 'All Users':
        df = df[df['user'] == user]

    check = df['sentiment'].value_counts()
    return check


def sentiment_tracker(df):
    listofusers = []
    for user in df['user']:
        if user not in listofusers:
            listofusers.append(user)

    listofusers.remove('group_notification')

    sentiments_per_user = {}
    for new_user in listofusers:
        user_sentiments = df.loc[df['user'] == new_user, 'sentiment'].tolist()
        sentiments_per_user[new_user] = user_sentiments

    user_sentiment_totals = []
    for user, sentiments in sentiments_per_user.items():
        total = 0
        count = 0
        for i in range(0, len(sentiments)):
            if sentiments[i] == 'Positive':
                total = total + 1

            elif sentiments[i] == 'Negative':
                total = total - 1

            count = count + 1

        total_sentiment = round((total/count) * 100, 2)
        user_sentiment_totals.append(
            {'User': user, 'Total Sentiment': total_sentiment})

    sentiment_df = pd.DataFrame(user_sentiment_totals)
    sorted_sentiment_df = sentiment_df.sort_values(
        'Total Sentiment', ascending=False)
    top_3_highest = sorted_sentiment_df.head(3).reset_index(drop=True)
    top_3_lowest = sorted_sentiment_df.tail(
        3).sort_values('Total Sentiment', ascending=True).reset_index(drop=True)

    return top_3_highest, top_3_lowest, sentiment_df
