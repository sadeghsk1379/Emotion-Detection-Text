# وارد کردن کتابخانه‌های مورد نیاز
import streamlit as st  # برای ساخت و اجرای اپلیکیشن وب
import altair as alt  # برای ایجاد نمودارهای تعاملی
import plotly.express as px  # برای ایجاد نمودارهای پیچیده‌تر
import pandas as pd  # برای کار با داده‌ها به صورت جدولی
import numpy as np  # برای کار با آرایه‌های عددی
from datetime import datetime  # برای کار با تاریخ و زمان
import joblib  # برای بارگذاری مدل‌های آموزش دیده شده
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST  # وارد کردن توابع مرتبط با پایگاه داده و منطقه‌زمانی از فایل track_utils

# Load Model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))  # بارگذاری مدل طبقه‌بندی احساسات از فایل

# Function
def predict_emotions(docx):
    """
    پیش‌بینی احساسات بر اساس متن ورودی با استفاده از مدل آموزش دیده شده.
    """
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    """
    برگرداندن احتمال‌های پیش‌بینی احساسات بر اساس متن ورودی.
    """
    results = pipe_lr.predict_proba([docx])
    return results

# دیکشنری احساسات به صورت ایموجی
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Main Application
def main():
    st.title("Emotion Classifier App")  # عنوان اصلی اپلیکیشن

    # منوی انتخابی
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)  # انتخاب گزینه از منو در نوار کناری

    create_page_visited_table()  # ایجاد جدول برای ردیابی صفحات بازدید شده
    create_emotionclf_table()  # ایجاد جدول برای ذخیره نتایج پیش‌بینی احساسات

    if choice == "Home":  # اگر گزینه "Home" انتخاب شود
        add_page_visited_details("Home", datetime.now(IST))  # ثبت جزئیات بازدید از صفحه "Home"
        st.subheader("Emotion Detection in Text")  # عنوان فرعی برای تشخیص احساسات در متن

        with st.form(key='emotion_clf_form'):  # استفاده از فرم برای ورود ورودی متنی
            raw_text = st.text_area("Type Here")  # جعبه متنی برای ورود متن
            submit_text = st.form_submit_button(label='Submit')  # دکمه ارسال فرم

        if submit_text:  # اگر دکمه ارسال فرم فشرده شود
            col1, col2 = st.columns(2)  # تقسیم ستونی صفحه به دو ستون

            prediction = predict_emotions(raw_text)  # پیش‌بینی احساسات برای متن ورودی
            probability = get_prediction_proba(raw_text)  # دریافت احتمال‌های پیش‌بینی احساسات برای متن ورودی

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))  # ثبت نتیجه پیش‌بینی در پایگاه داده

            with col1:
                st.success("Original Text")  # نمایش پیام موفقیت برای متن اصلی
                st.write(raw_text)  # نمایش متن اصلی ورودی

                st.success("Prediction")  # نمایش پیام موفقیت برای پیش‌بینی
                emoji_icon = emotions_emoji_dict[prediction]  # انتخاب ایموجی مربوط به پیش‌بینی
                st.write("{}:{}".format(prediction, emoji_icon))  # نمایش پیش‌بینی احساس به همراه ایموجی
                st.write("Confidence:{}".format(np.max(probability)))  # نمایش درصد اطمینان پیش‌بینی

            with col2:
                st.success("Prediction Probability")  # نمایش پیام موفقیت برای احتمال‌های پیش‌بینی
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)  # تبدیل احتمال‌ها به دیتافریم
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')  # ایجاد نمودار نواری تعاملی با altair
                st.altair_chart(fig, use_container_width=True)  # نمایش نمودار با استفاده از altair

    elif choice == "Monitor":  # اگر گزینه "Monitor" انتخاب شود
        add_page_visited_details("Monitor", datetime.now(IST))  # ثبت جزئیات بازدید از صفحه "Monitor"
        st.subheader("Monitor App")  # عنوان فرعی برای اپلیکیشن Monitor

        with st.expander("Page Metrics"):  # استفاده از expander برای گسترش بخش معیارهای صفحه
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])  # دریافت و نمایش جزئیات بازدید صفحه
            st.dataframe(page_visited_details)  # نمایش دیتافریم در اپلیکیشن

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')  # محاسبه تعداد بازدیدهای هر صفحه
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')  # ایجاد نمودار نواری با altair
            st.altair_chart(c, use_container_width=True)  # نمایش نمودار با استفاده از altair

            p = px.pie(pg_count, values='Counts', names='Page Name')  # ایجاد نمودار دایره‌ای با plotly
            st.plotly_chart(p, use_container_width=True)  # نمایش نمودار دایره‌ای با plotly

        with st.expander('Emotion Classifier Metrics'):  # استفاده از expander برای گسترش بخش معیارهای طبقه‌بند احساسات
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])  # دریافت و نمایش جزئیات پیش‌بینی‌های احساسات
            st.dataframe(df_emotions)  # نمایش دیتافریم در اپلیکیشن

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')  # محاسبه تعداد هر نوع پیش‌بینی احساسات
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')  # ایجاد نمودار نواری با altair
            st.altair_chart(pc, use_container_width=True)  # نمایش نمودار با استفاده از altair

    else:  # در غیر این صورت (اگر گزینه "About" انتخاب شود)
        add_page_visited_details("About", datetime.now(IST))  # ثبت جزئیات بازدید از صفحه "About"
        
        st.write("Welcome to the Emotion Detection in Text App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")
        # خوش آمدگویی و توضیح در مورد اپلیکیشن

        st.subheader("Our Mission")  # عنوان فرعی در مورد ماموریت

        st.write("At Emotion Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that emotions play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")
        # توضیح در مورد ماموریت اپلیکیشن

        st.subheader("How It Works")  # عنوان فرعی در مورد نحوه کارکرد

        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained model, which predicts the emotions associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional content of your text.")
        # توضیح در مورد نحوه کار اپلیکیشن

        st.subheader("Key Features:")  # عنوان فرعی در مورد ویژگی‌های کلیدی

        st.markdown("##### 1. Real-time Emotion Detection")  # عنوان کلیدی اول

        st.write("Our app offers real-time emotion detection, allowing you to instantly analyze the emotions expressed in any given text. Whether you're analyzing customer feedback, social media posts, or any other form of text, our app provides you with immediate insights into the emotions underlying the text.")
        # توضیح در مورد ویژگی اول

        st.markdown("##### 2. Confidence Score")  # عنوان کلیدی دوم

        st.write("Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions. This score helps you gauge the reliability of the emotion detection results and make more informed decisions based on the analysis.")
        # توضیح در مورد ویژگی دوم

        st.markdown("##### 3. User-friendly Interface")  # عنوان کلیدی سوم

        st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the emotions detected. Whether you're a seasoned data scientist or someone with limited technical expertise, our app is accessible to all.")
        # توضیح در مورد ویژگی سوم

        st.subheader("Applications")  # عنوان فرعی در مورد کاربردها

        st.markdown("""
          The Emotion Detection in Text App has a wide range of applications across various industries and domains. Some common use cases include:
          - Social media sentiment analysis
          - Customer feedback analysis
          - Market research and consumer insights
          - Brand monitoring and reputation management
          - Content analysis and recommendation systems
          """)
        # کاربردهای اپلیکیشن

if __name__ == '__main__':
    main()  # فراخوانی تابع اصلی برنامه
