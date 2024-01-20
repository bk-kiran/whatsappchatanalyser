from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pandas as pd
import nltk
nltk.downloader.download('vader_lexicon')


def preprocess(data):
    pattern = "\[[^\]]*\]"
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # create a dataframe and convert date/time to a datetime type
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'].str.strip(
        '[').str.strip(']'), format="%m/%d/%y, %H:%M:%S")
    df.rename(columns={'message_date': 'date'}, inplace=True)

    test_message = df['user_message'][0]
    test_entry = re.split(r':\s', test_message, maxsplit=1)
    group_name = test_entry[0]

    # filter out the names of the users and the respective messages
    users = []
    messages = []

    for message in df['user_message']:
        entry = re.split(r':\s', message, maxsplit=1)
        if entry[0] == group_name:  # filtering for group notifications
            users.append('group_notification')
            messages.append(entry[1])
        # filtering for group notifications
        elif 'added' in entry[1] or 'changed group' in entry[1] or 'created group' in entry[1]:
            users.append('group_notification')
            messages.append(entry[1])
        else:
            users.append(entry[0])
            messages.append(entry[1])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['second'] = df['date'].dt.second

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour+1))
        else:
            period.append(str(hour) + "-" + str(hour+1))

    df['period'] = period

    sentiments = SentimentIntensityAnalyzer()
    df['positive'] = [sentiments.polarity_scores(
        i)['pos'] for i in df['message']]
    df['negative'] = [sentiments.polarity_scores(
        i)['neg'] for i in df['message']]

    for i, pos in enumerate(df['positive']):
        if pos >= 0.4:
            df.loc[i, 'sentiment'] = 'Positive'
        else:
            df.loc[i, 'sentiment'] = 'Neutral'

    for i, pos in enumerate(df['negative']):
        if pos >= 0.4:
            df.loc[i, 'sentiment'] = 'Negative'

    return df
