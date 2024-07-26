import sqlite3  # وارد کردن کتابخانه SQLite برای عملیات پایگاه داده
import pytz  # وارد کردن کتابخانه pytz برای مدیریت منطقه‌زمانی
from datetime import datetime  # وارد کردن ماژول datetime برای عملیات تاریخ و زمان

# برقراری اتصال به فایل پایگاه داده SQLite './data/data.db'
conn = sqlite3.connect('./data/data.db', check_same_thread=False)
c = conn.cursor()  # ایجاد یک شیء cursor برای اجرای کوئری‌های SQL

IST = pytz.timezone('Asia/Tehran')  # تعریف منطقه‌زمانی Indian Standard Time (IST)

# تابع برای ایجاد جدول 'pageTrackTable' برای ردیابی بازدید صفحات
def create_page_visited_table():
    c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT, timeOfvisit TIMESTAMP)')
    """
    اگر وجود نداشت، جدولی به نام 'pageTrackTable' ایجاد می‌کند.
    این جدول دارای دو ستون است:
        - 'pagename': نام متنی صفحه بازدید شده
        - 'timeOfvisit': زمان بازدید به شکل timestamp
    """

# تابع برای افزودن جزئیات بازدید صفحه به جدول 'pageTrackTable'
def add_page_visited_details(pagename, timeOfvisit=None):
    if timeOfvisit is None:
        timeOfvisit = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timeOfvisit = timeOfvisit.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO pageTrackTable(pagename, timeOfvisit) VALUES (?, ?)', (pagename, timeOfvisit))
    conn.commit()
    """
    اطلاعات بازدید صفحه را به جدول 'pageTrackTable' اضافه می‌کند.
    پارامترها:
        - pagename: نام صفحه بازدید شده (TEXT)
        - timeOfvisit: timestamp بازدید (به شکل زمان فعلی در IST اگر مشخص نشده باشد)
    """

# تابع برای مشاهده تمام جزئیات بازدید صفحات از جدول 'pageTrackTable'
def view_all_page_visited_details():
    c.execute('SELECT * FROM pageTrackTable')
    data = c.fetchall()
    return data
    """
    همه رکوردهای جدول 'pageTrackTable' را بازیابی کرده و به عنوان یک لیست از تاپل‌ها باز می‌گرداند.
    """

# تابع برای ایجاد جدول 'emotionclfTable' برای پیش‌بینی‌های طبقه‌بندی احساسات
def create_emotionclf_table():
    c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(rawtext TEXT, prediction TEXT, probability NUMBER, timeOfvisit TIMESTAMP)')
    """
    اگر وجود نداشت، جدولی به نام 'emotionclfTable' ایجاد می‌کند.
    این جدول دارای چهار ستون است:
        - 'rawtext': متن اصلی ورودی
        - 'prediction': پیش‌بینی احساس یا طبقه‌بندی
        - 'probability': امتیاز اطمینان یا احتمال
        - 'timeOfvisit': timestamp زمان پیش‌بینی
    """

# تابع برای افزودن جزئیات پیش‌بینی طبقه‌بندی احساسات به جدول 'emotionclfTable'
def add_prediction_details(rawtext, prediction, probability, timeOfvisit=None):
    if timeOfvisit is None:
        timeOfvisit = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timeOfvisit = timeOfvisit.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO emotionclfTable(rawtext, prediction, probability, timeOfvisit) VALUES (?, ?, ?, ?)', (rawtext, prediction, probability, timeOfvisit))
    conn.commit()
    """
    اطلاعات پیش‌بینی طبقه‌بندی احساسات را به جدول 'emotionclfTable' اضافه می‌کند.
    پارامترها:
        - rawtext: متن اصلی ورودی (TEXT)
        - prediction: پیش‌بینی احساس یا طبقه‌بندی (TEXT)
        - probability: امتیاز اطمینان یا احتمال (NUMBER)
        - timeOfvisit: timestamp زمان پیش‌بینی (به شکل زمان فعلی در IST اگر مشخص نشده باشد)
    """

# تابع برای مشاهده تمام جزئیات پیش‌بینی طبقه‌بندی احساسات از جدول 'emotionclfTable'
def view_all_prediction_details():
    c.execute('SELECT * FROM emotionclfTable')
    data = c.fetchall()
    return data
    """
    همه رکوردهای جدول 'emotionclfTable' را بازیابی کرده و به عنوان یک لیست از تاپل‌ها باز می‌گرداند.
    """
