import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai
from google.cloud import vision 
import io
import urllib.parse 
# ruaccent 관련 import 제거 (외부 링크로 대체되었으므로)

# ---------------------- 0. 초기 설정 및 세션 상태 ----------------------

# 🌟 1. 교재 연습용 텍스트 데이터 정의
NEW_DEFAULT_TEXT = """МОЙ РАБОЧИЙ ДЕНЬ
(Рассказ японского банкира)
Разрешите представитьcя. Меня зовут Такеши Осада. Я работаю в банке «Сакура». Я живу недалеко от Токио, поэтому в рабочие дни я встаю в 5 часов утра, умываюсь, одеваюсь, завтракаю и иду на станцию. На станции я покупаю свежую газету. Я еду на работу на электричке. В электричке я обычно читаю или сплю. Дорога от дома до работы занимает 2 часа.
Банк начинает работать в 8 утра, а заканчивает в 8 вечера, то есть мой рабочий день продолжается 12 часов, включая 2 перерыва. Рабочий день для женщин, конечно, меньше.
В 8:30 обычно начинается собрание, где мы обсуждаем экономическую ситуацию, курс доллара, последние экономические новости, планируем работу на день. Потом я читаю документы, решаю важные вопросы, встречаюсь с клиентами, открываю им счёт в банке, даю им кредит, разговариваю по телефону и так далее.
После работы я возвращаюсь домой. Так как я очень устаю, дома я сразу ложусь спать."""

DEFAULT_TEST_TEXT = "Человек идёт по улице. Это тестовая строка. Хорошо. Я часто читаю эту книгу."


st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기") 

# --- 세션 상태 초기화 ---
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None
if "word_info" not in st.session_state:
    st.session_state.word_info = {}
if "current_search_query" not in st.session_state:
    st.session_state.current_search_query = ""
if "ocr_output_text" not in st.session_state:
    st.session_state.ocr_output_text = ""
# 🌟 display_text 대신 input_text_area (위젯 키)를 메인 텍스트 상태로 사용
if "input_text_area" not in st.session_state:
    st.session_state.input_text_area = DEFAULT_TEST_TEXT
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "last_processed_text" not in st.session_state:
    st.session_state.last_processed_text = "" 
if "last_processed_query" not in st.session_state:
    st.session_state.last_processed_query = ""


mystem = Mystem()

# ---------------------- 품사 변환 딕셔너리 및 Mystem 함수 ----------------------
POS_MAP = {
    'S': '명사', 'V': '동사', 'A': '형용사', 'ADV': '부사', 'PR': '전치사',
    'CONJ': '접속사', 'INTJ': '감탄사', 'PART': '불변화사', 'NUM': '수사',
    'APRO': '대명사적 형용사', 'ANUM': '서수사', 'SPRO': '대명사',
}

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    if ' ' in word.strip():
        return word.strip()
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        lemmas = mystem.lemmatize(word)
        return (lemmas[0] if lemmas else word).strip()
    return word

@st.cache_data(show_spinner=False)
def get_pos_ru(word: str) -> str:
    if ' ' in word.strip():
        return '관용구'
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        analysis = mystem.analyze(word)
        if analysis and 'analysis' in analysis[0] and analysis[0]['analysis']:
            grammar_info = analysis[0]['analysis'][0]['gr']
            pos_abbr = grammar_info.split('=')[0].split(',')[0].strip()
            return POS_MAP.get(pos_abbr, '품사')
    return '품사'

# ---------------------- OCR 함수 ----------------------
@st.cache_data(show_spinner="이미지에서 텍스트 추출 중")
def detect_text_from_image(image_bytes):
    try:
        if st.secrets.get("GCP_SA_KEY"):
            with open("temp_sa_key.json", "w") as f:
                json.dump(st.secrets["GCP_SA_KEY"], f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_sa_key.json"
        elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
