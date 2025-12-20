import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai
import urllib.parse
from typing import Union

# ---------------------- 0. 초기 설정 및 상수 ----------------------

NEW_DEFAULT_TEXT = """Том живёт в Санкт-Петербурге уже несколько месяцев. В субботу, когда была хорошая погода, Том решил пойти в Исаакиевский собор. Том давно мечтал побывать в этом соборе. И사아키예프스키 소보르 — одно из самых высоких зданий в Санкт-Петербурге, его можно увидеть даже издалека. Когда Том гулял по центру города, он отовсюду видел золотой купол собора. Сначала Том решил осмотреть собор снаружи. Он пришёл на Исаакиевскую площадь — отсюда открывается прекрасный вид на собор. Потом Том подошёл к собору поближе, осмотрел его спереди, сзади, 2 раза обошёл вокруг собора, потом вошёл внутрь. Внутри собор очень красивый. Том прочитал, что купол собора — третий по величине в Европе. Том поднял голову вверх и увидел, что под куполом «летает» серебряный голубь. Том посмотрел вокруг: впереди, сзади, справа, слева — везде были красивые иконы. Потом Том решил подняться на колоннаду собора. В выходной день в соборе было много туристов: одни поднимались вверх, другие спускались вниз. Том тоже поднялся вверх. Отту다, сверху, с высоты 43 (сорока трёх) метров, открывается прекрасный вид на центр города. Том увидел Дворцовую площадь, Петропавловскую крепость, крыши домов, а над крышами летали птицы. Тому очень понравилась экскурсия. Он посоветовал друзьям посетить собор и обязательно подняться на колоннаду."""

DEFAULT_TEST_TEXT = "Человек идёт по улице. Это тестовая строка. Хорошо. Я часто читаю эту книгу."

mystem = Mystem()
YOUTUBE_VIDEO_ID = "wJ65i_gDfT0" 
IMAGE_FILE_PATH = "banner.png"

# --- 세션 상태 초기화 함수 ---
def initialize_session_state():
    if "selected_words" not in st.session_state:
        st.session_state.selected_words = []
    if "clicked_word" not in st.session_state:
        st.session_state.clicked_word = None
    if "word_info" not in st.session_state:
        st.session_state.word_info = {}
    if "current_search_query" not in st.session_state:
        st.session_state.current_search_query = ""
    if "input_text_area" not in st.session_state:
        st.session_state.input_text_area = DEFAULT_TEST_TEXT
    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""
    if "last_processed_text" not in st.session_state:
        st.session_state.last_processed_text = ""
    if "last_processed_query" not in st.session_state:
        st.session_state.last_processed_query = ""

# ---------------------- 0.1. 페이지 설정 및 배너 ----------------------

initialize_session_state()
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")

try:
    st.image(IMAGE_FILE_PATH, use_column_width=True)
except FileNotFoundError:
    st.warning(f"배너 이미지를 찾을 수 없습니다.")

# ---------------------- 0.2. YouTube 임베드 함수 ----------------------

def youtube_embed_html(video_id: str):
    embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=0&rel=0"
    return f"""
    <div class="video-container-wrapper">
        <div class="video-responsive">
            <iframe src="{embed_url}" frameborder="0" allowfullscreen></iframe>
        </div>
    </div>
    """

# ---------------------- 품사 및 형태소 분석 ----------------------
POS_MAP = {
    'S': '명사', 'V': '동사', 'A': '형용사', 'ADV': '부사', 'PR': '전치사',
    'CONJ': '접속사', 'INTJ': '감탄사', 'PART': '불변화사', 'NUM': '수사',
    'APRO': '대명사적 형용사', 'ANUM': '서수사', 'SPRO': '대명사',
    'PRICL': '동사부사', 'COMP': '비교급', 'ADVB': '부사',
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
            pos_full = grammar_info.split(',')[0].strip()
            # больница 처리를 위한 내부 로직 (필요 시 유지)
            return POS_MAP.get(pos_full.split('=')[0], '품사')
    return '품사'

# ---------------------- 1. Gemini 연동 함수 ----------------------

def get_word_info_schema(is_verb: bool):
    schema = {
        "type": "object",
        "properties": {
            "ko_meanings": {"type": "array", "items": {"type": "string"}},
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ru": {"type": "string"},
                        "ko": {"type": "string"}
                    },
                    "required": ["ru", "ko"]
                }
            }
        },
        "required": ["ko_meanings", "examples"]
    }
    if is_verb:
        schema['properties']['aspect_pair'] = {
            "type": "object",
            "properties": {"imp": {"type": "string"}, "perf": {"type": "string"}},
            "required": ["imp", "perf"]
        }
        schema['required'].append('aspect_pair')
    return schema

@st.cache_data(show_spinner=False, ttl=300) 
def fetch_from_gemini(word, lemma, pos):
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    client = genai.Client(api_key=api_key) if api_key else None
    if not client: return {"ko_meanings": ["API 키 없음"], "examples": []}
    
    is_verb = (pos == '동사')
    config = {
        "system_instruction": "너는 러시아어 학습 도우미다. JSON 형식으로만 답하며, 'больница' 같은 단어 처리에 유의하라.",
        "response_mime_type": "application/json",
        "response_schema": get_word_info_schema(is_verb),
    }
    prompt = f"단어: {word}. 기본형: {lemma}. 품사: {pos}."
    try:
        res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt, config=config)
        return json.loads(res.text)
    except Exception as e:
        return {"ko_meanings": [f"오류: {str(e)}"], "examples": []}

# ---------------------- 2. 텍스트 번역 함수 ----------------------

@st.cache_data(show_spinner="번역 중...", ttl=600)
def translate_text(russian_text, highlight_words):
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    client = genai.Client(api_key=api_key) if api_key else None
    if not client: return "API 키가 없습니다."
    
    phrases = ", ".join([f"'{w}'" for w in highlight_words])
    prompt = f"번역할 텍스트: '{russian_text}'. 하이라이트 대상: {phrases}. 한국어 번역 시 대상 단어는 <PHRASE_START>...<PHRASE_END>로 감싸줘."
    
    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"system_instruction": "너는 번역가다. 결과물에 부연설명 없이 오직 번역본만 출력한다."}
        )
        translated = res.text.strip().replace("<PHRASE_START>", '<span class="word-selected">').replace("<PHRASE_END>", '</span>')
        return translated
    except Exception as e:
        return f"번역 오류: {e}"

# ---------------------- 3. 전역 스타일 ----------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    html, body, .stApp { font-family: 'Nanum Gothic', sans-serif !important; }
    .text-container { line-height: 2.0; font-size: 1.25em; margin-bottom: 20px; }
    .word-selected { color: #007bff !important; font-weight: bold; background-color: #e0f0ff; border-bottom: 3px solid #007bff; }
    .video-responsive { overflow: hidden; padding-bottom: 56.25%; position: relative; height: 0; }
    .video-responsive iframe { left: 0; top: 0; height: 100%; width: 100%; position: absolute; }
</style>
""", unsafe_allow_html=True)

# ---------------------- 4. UI 배치 및 로직 ----------------------

def load_default_text():
    st.session_state.input_text_area = NEW_DEFAULT_TEXT
    st.session_state.translated_text = ""
    st.session_state.selected_words = []

st.subheader("분석 대상 텍스트")
st.button("중급러시아어연습 텍스트 불러오기", on_click=load_default_text)

current_text = st.text_area(
    "러시아어 텍스트를 입력하세요",
    value=st.session_state.input_text_area,
    height=150,
    key="input_text_area_widget"
)

# 텍스트 변경 감지
if current_text != st.session_state.last_processed_text:
    st.session_state.input_text_area = current_text
    st.session_state.translated_text = ""
    st.session_state.last_processed_text = current_text

st.divider()
st.subheader("단어/구 검색")
manual_input = st.text_input("검색할 단어 또는 구 입력")

if manual_input and manual_input != st.session_state.last_processed_query:
    if manual_input not in st.session_state.selected_words:
        st.session_state.selected_words.append(manual_input)
    st.session_state.clicked_word = manual_input
    
    lemma = lemmatize_ru(manual_input)
    pos = get_pos_ru(manual_input)
    info = fetch_from_gemini(manual_input, lemma, pos)
    st.session_state.word_info[lemma] = {**info, "loaded_token": manual_input, "pos": pos}
    st.session_state.last_processed_query = manual_input

# ---------------------- 5. 메인 레이아웃 ----------------------

left, right = st.columns([2, 1])

with left:
    st.subheader("러시아어 원문")
    # 하이라이팅 적용 로직 (간략화)
    display_html = current_text
    for word in sorted(st.session_state.selected_words, key=len, reverse=True):
        display_html = re.sub(f'({re.escape(word)})', r'<span class="word-selected">\1</span>', display_html)
    
    st.markdown(f'<div class="text-container">{display_html}</div>', unsafe_allow_html=True)
    
    if st.button("초기화"):
        initialize_session_state()
        st.rerun()

with right:
    st.subheader("상세 정보")
    if st.session_state.clicked_word:
        token = st.session_state.clicked_word
        lemma = lemmatize_ru(token)
        info = st.session_state.word_info.get(lemma, {})
        
        st.markdown(f"### {token}")
        if info:
            st.write(f"**기본형:** {lemma} ({info.get('pos')})")
            for m in info.get("ko_meanings", []):
                st.write(f"- {m}")
            st.divider()
            for ex in info.get("examples", []):
                st.caption(f"RU: {ex['ru']}")
                st.write(f"KO: {ex['ko']}")
    else:
        st.info("단어를 검색하면 정보가 표시됩니다.")

# ---------------------- 6. 하단 결과 ----------------------

st.divider()
st.subheader("한국어 번역본")
if not st.session_state.translated_text:
    st.session_state.translated_text = translate_text(current_text, st.session_state.selected_words)

st.markdown(f'<div class="text-container">{st.session_state.translated_text}</div>', unsafe_allow_html=True)

# 홍보 영상
_, col_v = st.columns([1, 1])
with col_v:
    st.subheader("🎬 프로젝트 홍보 영상")
    st.markdown(youtube_embed_html(YOUTUBE_VIDEO_ID), unsafe_allow_html=True)

st.markdown("---")
st.caption(" 이 페이지는 연세대학교 노어노문학과 25-2 러시아어 교육론 5팀의 프로젝트 결과물입니다. <br>  본 페이지의 내용, 기능 및 데이터를 학습 목적 이외의 용도로 무단 복제, 배포, 상업적 이용할 경우,  관련 법령에 따라 민사상 손해배상 청구 및 형사상 처벌을 받을 수 있습니다.")
