import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai
# vision 클라이언트 임포트 제거
import io
import urllib.parse
from typing import Union

# ---------------------- 0. 초기 설정 및 상수 ----------------------

NEW_DEFAULT_TEXT = """Том живёт в Санкт-Петербурге уже несколько месяцев. В субботу, когда была хорошая погода, Том решил пойти в Исаакиевский собор. Том давно мечтал побывать в этом соборе. Исаакиевский собор — одно из самых высоких зданий в Санкт-Петербурге, его можно увидеть
даже изда레가. Когда Том гулял по центру города, он отовсюду видел золотой купол собора. Сначала Том решил осмотреть собор снаружи. Он пришёл на Исаакиевскую площадь — отсюда открывается прекрасный вид на собор. Потом Том подошёл к собору поближе, осмотрел его спереди, сза디, 2 раза обошёл вокруг собора, потом вошёл внутрь. Внутри собор очень красивый. Том прочитал, что купол собора — третий по величине в Европе. Том поднял голову вверх и увидел, что под куполом «летает» серебряный голубь. Том посмотрел вокруг: впереди, сзади, справа, слева — везде были красивые иконы.
Потом Том решил подняться на колоннаду собора. В выходной день в соборе было много туристов: одни поднимались вверх, другие спускались вниз. Том тоже поднялся вверх. Оттуда, сверху, с высоты 43 (сорока трёх) метров, открывается прекрасный вид на центр города. Том увидел Дворцовую площадь, Петропавловскую крепость, крыши домов, а над крышами летали птицы.
Тому очень понравилась экскурсия. Он посоветовал друзьям посетить собор и обязательно подняться на колонна두."""

DEFAULT_TEST_TEXT = "Человек идёт по улице. Это тестовая строка. Хорошо. Я часто читаю эту книгу."

mystem = Mystem()
YOUTUBE_VIDEO_ID = "wJ65i_gDfT0" 
IMAGE_FILE_PATH = "banner.png"

def initialize_session_state():
    if "selected_words" not in st.session_state:
        st.session_state.selected_words = []
    if "clicked_word" not in st.session_state:
        st.session_state.clicked_word = None
    if "word_info" not in st.session_state:
        st.session_state.word_info = {}
    if "current_search_query" not in st.session_state:
        st.session_state.current_search_query = ""
    # ocr_output_text는 다른 로직과의 호환성을 위해 유지하되 UI에선 제거
    if "ocr_output_text" not in st.session_state:
        st.session_state.ocr_output_text = ""
    if "input_text_area" not in st.session_state:
        st.session_state.input_text_area = DEFAULT_TEST_TEXT
    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""
    if "last_processed_text" not in st.session_state:
        st.session_state.last_processed_text = ""
    if "last_processed_query" not in st.session_state:
        st.session_state.last_processed_query = ""

initialize_session_state()
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")

# 상단 배너 이미지는 유지
try:
    st.image(IMAGE_FILE_PATH, use_column_width=True)
except FileNotFoundError:
    st.markdown("### 러시아어 텍스트 분석 시스템")

# ---------------------- 0.2. YouTube 임베드 함수 (유지) ----------------------
def youtube_embed_html(video_id: str):
    embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=0&rel=0"
    return f'<div class="video-container-wrapper"><div class="video-responsive"><iframe src="{embed_url}" frameborder="0" allowfullscreen></iframe></div></div>'

# ---------------------- 품사 변환 및 Mystem 함수 (유지) ----------------------
POS_MAP = {
    'S': '명사', 'V': '동사', 'A': '형용사', 'ADV': '부사', 'PR': '전치사',
    'CONJ': '접속사', 'INTJ': '감탄사', 'PART': '불변화사', 'NUM': '수사',
    'APRO': '대명사적 형용사', 'ANUM': '서수사', 'SPRO': '대명사',
    'PRICL': '동사부사', 'COMP': '비교급', 'A=cmp': '비교급 형용사', 'ADV=cmp': '비교급 부사',
    'ADVB': '부사', 'NONLEX': '비단어', 'INIT': '머리글자', 'P': '불변화사/전치사', 'ADJ': '형용사', 'N': '명사',
}

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    if ' ' in word.strip(): return word.strip()
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        lemmas = mystem.lemmatize(word)
        return (lemmas[0] if lemmas else word).strip()
    return word

@st.cache_data(show_spinner=False)
def get_pos_ru(word: str) -> str:
    if ' ' in word.strip(): return '구 형태'
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        analysis = mystem.analyze(word)
        if analysis and 'analysis' in analysis[0] and analysis[0]['analysis']:
            grammar_info = analysis[0]['analysis'][0]['gr']
            parts = re.split(r'[,=]', grammar_info, 1)
            pos_abbr_base = parts[0].strip()
            pos_full = grammar_info.split(',')[0].strip()
            if pos_full in POS_MAP: return POS_MAP[pos_full]
            return POS_MAP.get(pos_abbr_base, '품사')
    return '품사'

# ---------------------- 1. Gemini 연동 함수 (핵심 API 호출부 유지) ----------------------
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    return genai.Client(api_key=api_key) if api_key else None

def get_word_info_schema(is_verb: bool):
    schema = {
        "type": "object",
        "properties": {
            "ko_meanings": {"type": "array", "items": {"type": "string"}},
            "examples": {"type": "array", "items": {"type": "object", "properties": {"ru": {"type": "string"}, "ko": {"type": "string"}}, "required": ["ru", "ko"]}}
        },
        "required": ["ko_meanings", "examples"]
    }
    if is_verb:
        schema['properties']['aspect_pair'] = {"type": "object", "properties": {"imp": {"type": "string"}, "perf": {"type": "string"}}, "required": ["imp", "perf"]}
        schema['required'].append('aspect_pair')
    return schema

@st.cache_data(show_spinner=False, ttl=300) 
def fetch_from_gemini(word, lemma, pos):
    client = get_gemini_client()
    if not client: return {"ko_meanings": ["API 키 없음"], "examples": []}
    is_verb = (pos == '동사')
    config = {"system_instruction": "너는 러시아어-한국어 학습 도우미이다. JSON으로만 답한다.", "response_mime_type": "application/json", "response_schema": get_word_info_schema(is_verb)}
    prompt = f"러시아어 단어: {word}. 기본형: {lemma}. 품사: {pos}. 정보를 요청합니다."
    try:
        res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config=config)
        return json.loads(res.text)
    except Exception as e:
        return {"ko_meanings": [f"API 오류: {str(e)}"], "examples": []}

@st.cache_data(show_spinner="번역 중...", ttl=600)
def translate_text(russian_text, highlight_words):
    client = get_gemini_client()
    if not client: return "API 키 없음"
    phrases = ", ".join([f"'{w}'" for w in highlight_words])
    prompt = f"러시아어 텍스트: '{russian_text}' 번역. 강조 단어: {phrases}. <PHRASE_START>, <PHRASE_END> 마크업 사용."
    try:
        res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config={"system_instruction": "자연스러운 한국어 번역기."})
        translated = res.text.strip().replace("<PHRASE_START>", '<span class="word-selected">').replace("<PHRASE_END>", '</span>')
        return translated
    except Exception as e: return f"번역 오류: {e}"

# ---------------------- 3. 스타일 및 UI 로직 ----------------------
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, .stApp { font-family: 'Nanum Gothic', sans-serif !important; }
    .text-container { line-height: 2.0; font-size: 1.25em; margin-bottom: 20px; }
    .word-selected { color: #007bff !important; font-weight: bold; background-color: #e0f0ff; padding: 2px 0px; border-bottom: 3px solid #007bff; border-radius: 2px; }
</style>""", unsafe_allow_html=True)

def load_default_text():
    st.session_state.input_text_area = NEW_DEFAULT_TEXT
    st.session_state.translated_text = ""
    st.session_state.selected_words = []

# --- OCR 이미지 업로드 섹션만 삭제됨 ---
st.button("중급러시아어연습 텍스트 반영하기(교재 2권 44페이지)", on_click=load_default_text)

st.subheader("분석 대상 텍스트")
current_text = st.text_area("러시아어 텍스트를 입력하거나 수정하세요", value=st.session_state.input_text_area, height=150, key="input_text_area")

if current_text != st.session_state.last_processed_text:
    st.session_state.translated_text = ""
    st.session_state.selected_words = []
    st.session_state.word_info = {}

# --- 단어 검색 및 상세 정보 로직 (유지) ---
st.divider()
manual_input = st.text_input("단어 또는 구를 입력하고 Enter", key="current_search_query")
if manual_input and manual_input != st.session_state.get("last_processed_query"):
    if manual_input not in st.session_state.selected_words:
        st.session_state.selected_words.append(manual_input)
    st.session_state.clicked_word = manual_input
    lemma = lemmatize_ru(manual_input)
    pos = get_pos_ru(manual_input)
    info = fetch_from_gemini(manual_input, lemma, pos)
    st.session_state.word_info[lemma] = {**info, "loaded_token": manual_input, "pos": pos}
    st.session_state.last_processed_query = manual_input

# --- 레이아웃 출력 (유지) ---
left, right = st.columns([2, 1])
with left:
    st.subheader("러시아어 텍스트 원문")
    # 원문 하이라이팅 로직 적용
    display_html = current_text
    for phrase in sorted(st.session_state.selected_words, key=len, reverse=True):
        display_html = re.sub(f'({re.escape(phrase)})', r'<span class="word-selected">\1</span>', display_html)
    st.markdown(f'<div class="text-container">{display_html}</div>', unsafe_allow_html=True)

with right:
    st.subheader("단어 상세 정보")
    if st.session_state.clicked_word:
        token = st.session_state.clicked_word
        lemma = lemmatize_ru(token)
        info = st.session_state.word_info.get(lemma, {})
        if info:
            st.markdown(f"### **{token}**")
            st.write(f"기본형: {lemma} / 품사: {info.get('pos')}")
            st.write("뜻:", ", ".join(info.get("ko_meanings", [])))
    else: st.info("단어를 검색하세요.")

# 번역본 및 하단 영상 유지
st.divider()
st.subheader("한국어 번역본")
if not st.session_state.translated_text:
    st.session_state.translated_text = translate_text(current_text, st.session_state.selected_words)
    st.session_state.last_processed_text = current_text
st.markdown(f'<div class="text-container" style="color: #333;">{st.session_state.translated_text}</div>', unsafe_allow_html=True)

# 홍보 영상 레이아웃 (유지)
_, col_video = st.columns([1, 1])
with col_video:
    st.subheader("🎬 프로젝트 홍보 영상")
    st.markdown(youtube_embed_html(YOUTUBE_VIDEO_ID), unsafe_allow_html=True)

st.caption("© 연세대학교 노어노문학과 프로젝트")
