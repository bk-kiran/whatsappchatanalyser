import streamlit as st
import process
import helper
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='WhatsApp Chat Analyser', page_icon='💬')
st.sidebar.title("Discover your WhatsApp Group Statistics!")

uploaded_file = st.sidebar.file_uploader(
    "Choose a WhatsApp text file (.txt) to upload Time must be set to 24 hour clock")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = process.preprocess(data)

    # view all the users in the chat
    user_list = df['user'].unique().tolist()
    user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, 'All Users')

    # select the particular user(s)
    selected_user = st.sidebar.selectbox("Select User(s)", user_list)

    # button to show all stats
    if st.sidebar.button("Analyse Stats"):

        total_messages, total_words, total_media, total_links = helper.fetchstats(
            selected_user, df)  # fetching data

        st.header("Top Stats")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Number of Messages")
            st.title(total_messages)

        with col2:
            st.header("Number of Words")
            st.title(total_words)

        with col3:
            st.header("Media Shared")
            st.title(total_media)

        with col4:
            st.header("Links Shared")
            st.title(total_links)

        # this code only runs if all users are selected
        if selected_user == 'All Users':

            # sentiment tracker
            sentiment_highest, sentiment_lowest, sentiment_overall = helper.sentiment_tracker(
                df)
            col1, col2 = st.columns(2, gap='large')

            with col1:
                st.header("These 3 Users are the Nicest")
                st.dataframe(sentiment_highest)

            with col2:
                st.header("These 3 Users could be a little nicer")
                st.dataframe(sentiment_lowest)

            st.header('User Sentiment Data')
            fig, ax = plt.subplots()
            ax.bar(sentiment_overall["User"],
                   sentiment_overall["Total Sentiment"])
            plt.xticks(rotation="vertical")
            plt.ylabel("Sentimentality Score")
            st.pyplot(fig)

            # creating bar chart and dataframe to visualize busiest users
            st.header('Most Active Users')
            data, new_data = helper.busiest_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2, gap='large')

            with col1:
                ax.bar(data.index, data.values, color="green")
                plt.ylabel('Number of Messages')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_data)

        # create the yearly timeline linegraph
        st.header("Yearly Messages Timeline")
        month_timeline = helper.year_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(month_timeline['times'],
                month_timeline['message'], color='red')
        plt.xticks(rotation="vertical")
        st.pyplot(fig)

        # creating all the activity maps
        col1, col2 = st.columns(2, gap='large')

        with col1:  # creating the weekly activity count graph
            st.header("Daily Message Activity")
            busy_day = helper.weekly_message_count(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color="green")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        with col2:  # creating the monthly activity count graph
            st.header("Monthly Message Activity")
            busy_month = helper.monthly_message_count(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color="purple")
            plt.xticks(rotation="vertical")
            st.pyplot(fig)

        # activity heat map
        st.header("Activity Heat Map")
        activity_heatmap = helper.heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(activity_heatmap)
        st.pyplot(fig)

        # Creating a word cloud
        st.header('Most Common Words')
        df_cloud = helper.word_cloud(selected_user, df)
        fig, ax = plt.subplots()
        plt.imshow(df_cloud)
        st.pyplot(fig)

        # analysing emojis
        col1, col2 = st.columns(2, gap='large')
        with col1:
            emoji_df = helper.emoji_counter(selected_user, df)
            st.header("Most Common Emojis")
            st.dataframe(emoji_df)

        # finding total sentiment scores
        with col2:
            st.header("Total Sentiment Scores")
            sentiment_checker = helper.sentiment_count(selected_user, df)
            st.dataframe(sentiment_checker)
