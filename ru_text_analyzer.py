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
from typing import Union

# ---------------------- 0. 초기 설정 및 상수 ----------------------

NEW_DEFAULT_TEXT = """Том живёт в Санкт-Петербурге уже несколько месяцев. В субботу, когда была хорошая погода, Том решил пойти в Исаакиевский собор. Том давно мечтал побывать в этом соборе. Исаакиевский собор — одно из самых высоких зданий в Санкт-Петербурге, его можно увидеть
даже издалека. Когда Том гулял по центру города, он отовсюду видел золотой купол собора. Сначала Том решил осмотреть собор снаружи. Он пришёл на Исаакиевскую площадь — отсюда открывается прекрасный вид на собор. Потом Том подошёл к собору поближе, осмотрел его спереди, сзади, 2 раза обошёл вокруг собора, потом вошёл внутрь. Внутри собор очень красивый. Том прочитал, что купол собора — третий по величине в Европе. Том поднял голову вверх и увидел, что под куполом «летает» серебряный голубь. Том посмотрел вокруг: впереди, сзади, справа, слева — везде были красивые иконы.
Потом Том решил подняться на колоннаду собора. В выходной день в соборе было много туристов: одни поднимались вверх, другие спускались вниз. Том тоже поднялся вверх. Оттуда, сверху, с высоты 43 (сорока трёх) метров, открывается прекрасный вид на центр города. Том увидел Дворцовую площадь, Петропавловскую крепость, крыши домов, а над крышами летали птицы.
Тому очень понравилась экскурсия. Он посоветовал друзьям посетить собор и обязательно подняться на колоннаду."""

DEFAULT_TEST_TEXT = "Человек идёт по улице. Это тестовая строка. Хорошо. Я часто читаю эту книгу."

mystem = Mystem()
YOUTUBE_VIDEO_ID = "wJ65i_gDfT0" 
IMAGE_FILE_PATH = "banner.png"

# --- 세션 상태 초기화 함수 (AttributeError 방지) ---
def initialize_session_state():
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
    if "input_text_area" not in st.session_state:
        st.session_state.input_text_area = DEFAULT_TEST_TEXT
    if "translated_text" not in st.session_state:
        st.session_state.translated_text = ""
    if "last_processed_text" not in st.session_state:
        st.session_state.last_processed_text = ""
    if "last_processed_query" not in st.session_state:
        st.session_state.last_processed_query = ""

# ---------------------- 0.1. 페이지 설정 및 배너 삽입 ----------------------

# 세션 상태 초기화 실행
initialize_session_state()

st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")

try:
    st.image(IMAGE_FILE_PATH, use_column_width=True)
except FileNotFoundError:
    st.warning(f"배너 이미지 파일 ({IMAGE_FILE_PATH})을 찾을 수 없습니다. GitHub 저장소에 이미지를 업로드하고 파일명을 확인해주세요.")
    st.markdown("###")


# ---------------------- 0.2. YouTube 임베드 함수 ----------------------

def youtube_embed_html(video_id: str):
    """지정된 YouTube ID로 반응형 임베드 HTML을 반환합니다."""
    embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=0&rel=0"
    
    html_code = f"""
    <div class="video-container-wrapper">
        <div class="video-responsive">
            <iframe
                src="{embed_url}"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen
                title="프로젝트 홍보 영상"
            ></iframe>
        </div>
    </div>
    """
    return html_code


# ---------------------- 품사 변환 딕셔너리 및 Mystem 함수 ----------------------
POS_MAP = {
    'S': '명사', 'V': '동사', 'A': '형용사', 'ADV': '부사', 'PR': '전치사',
    'CONJ': '접속사', 'INTJ': '감탄사', 'PART': '불변화사', 'NUM': '수사',
    'APRO': '대명사적 형용사', 'ANUM': '서수사', 'SPRO': '대명사',
    'PRICL': '동사부사',
    'COMP': '비교급', 'A=cmp': '비교급 형용사', 'ADV=cmp': '비교급 부사',
    'ADVB': '부사',
    'NONLEX': '비단어',      
    'INIT': '머리글자',      
    'P': '불변화사/전치사', 
    'ADJ': '형용사',         
    'N': '명사',             
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
        return '구 형태' 
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        analysis = mystem.analyze(word)
        if analysis and 'analysis' in analysis[0] and analysis[0]['analysis']:
            grammar_info = analysis[0]['analysis'][0]['gr']
            parts = re.split(r'[,=]', grammar_info, 1)
            pos_abbr_base = parts[0].strip()
            pos_full = grammar_info.split(',')[0].strip()
            if pos_full in POS_MAP:
                return POS_MAP[pos_full]
            return POS_MAP.get(pos_abbr_base, '품사') 
    return '품사'

# ---------------------- OCR 클라이언트 및 함수 ----------------------

# Gemini 클라이언트 로직은 아래에서 재정의됨 (Vision API 디버깅을 위해)

def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    return genai.Client(api_key=api_key) if api_key else None

@st.cache_resource(show_spinner=False)
def get_vision_client():
    client = get_gemini_client() # Gemini 클라이언트 미리 가져오기
    # Gemini 클라이언트가 없으면 디버깅 불가 (Vision API 키 설정 여부와 별개)
    if client is None:
        st.error("Vision API 초기화 전에 Gemini API 키가 설정되어야 디버깅이 가능합니다.")
        return None

    try:
        # Secrets에서 JSON 키를 불러옴
        key_json = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON") 
        
        if not key_json:
            st.warning("Secrets 변수 'GOOGLE_APPLICATION_CREDENTIALS_JSON'이 설정되지 않았습니다.")
            return None

        import google.auth
        import google.cloud.vision
        
        # 🌟🌟🌟 1. JSON 유효성 검사 및 로드 시도 (오류 포착 지점) 🌟🌟🌟
        try:
            # 유니코드 제어 문자를 강제로 무시하고 ASCII로 클린하게 만듭니다.
            cleaned_json_string = key_json.encode('ascii', 'ignore').decode('ascii')
            key_data = json.loads(cleaned_json_string)

        except Exception as json_error:
            # JSON 로드 실패 시, Gemini에게 오류 분석 요청
            error_details = f"Python Traceback: {str(json_error)}\n\n문제의 JSON 시작 부분: {key_json[:300]}"
            
            debugging_prompt = f"""
            주어진 Python Traceback과 JSON 시작 부분을 분석하여, JSON 파싱 오류(특히 'Invalid control character' 오류)가 발생한 이유와, 사용자가 Secrets 파일에 어떤 문자를 잘못 입력했는지 설명해 주세요.

            {error_details}
            """
            
            try:
                gemini_res = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=debugging_prompt
                )
                st.error("🚨 JSON 키 파싱 오류 발생 (Gemini 분석 결과)")
                st.info(gemini_res.text.strip())
            except Exception:
                st.error("🚨 JSON 키 파싱 오류 발생. Gemini 디버깅도 실패했습니다. Secrets의 문자열을 확인해주세요.")
                
            return None
        
        # 2. Credential 생성 및 클라이언트 반환
        credentials, _ = google.auth.load_credentials_from_dict(key_data)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        return client
        
    except Exception as e:
        st.error(f"Vision API 클라이언트 초기화 오류: {e}")
        return None

# 🌟 TTL=3600초 (1시간) 설정 및 타임아웃 30초 추가
@st.cache_data(show_spinner="이미지에서 텍스트 추출 중...", ttl=3600)
def detect_text_from_image(image_bytes):
    
    client = get_vision_client()
    
    if client is None:
        return "OCR API 클라이언트 초기화 실패. Secrets (GOOGLE_APPLICATION_CREDENTIALS_JSON) 설정을 확인해주세요."

    try:
        image = vision.Image(content=image_bytes)
        image_context = vision.ImageContext(language_hints=["ru"])
        
        # 🌟 타임아웃 30초 설정 추가
        response = client.text_detection(
            image=image, 
            image_context=image_context,
            timeout=30 
        )
        texts = response.text
            
        if response.error.message:
            return f"Vision API 오류: {response.error.message}"
            
        return texts if texts else "이미지에서 텍스트를 찾을 수 없습니다."

    except Exception as e:
        error_msg = str(e)
        # 🌟 오류 메시지 필터링 (InvalidCharacterError 방지)
        if "HTTPConnection" in error_msg or "ConnectTimeoutError" in error_msg:
            return "OCR 처리 중 인증/네트워크 시간 초과 오류가 발생했습니다. (GCP Secrets 및 할당량 확인 필요)"
            
        return f"OCR 처리 중 오류 발생: {error_msg}"


# ---------------------- 1. Gemini 연동 함수 (TTL 및 JSON Schema 적용) ----------------------

# (fetch_from_gemini 함수는 코드 길이 관계로 생략합니다. 이전과 동일하게 유지됩니다.)
# ---------------------- 2. 텍스트 번역 함수 (TTL 10분 설정) ----------------------
# (translate_text 함수는 코드 길이 관계로 생략합니다. 이전과 동일하게 유지됩니다.)


# --- (UI 및 나머지 코드 생략) ---

# ---------------------- 7. 텍스트 하이라이팅 및 상세 정보 레이아웃 ----------------------
# (UI 코드는 이전과 동일하게 유지됩니다.)
# ...

# ---------------------- 11. 저작권 표시 (페이지 최하단) ----------------------
# (저작권 표시는 이전과 동일하게 유지됩니다.)
