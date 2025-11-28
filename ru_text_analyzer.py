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
        elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            return "OCR API 키(GOOGLE_APPLICATION_CREDENTIALS)가 설정되지 않았습니다. Cloud Vision API 설정을 확인해주세요."

        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        texts = response.text
        
        if response.error.message:
            return f"Vision API 오류: {response.error.message}"
            
        return texts.split('\n', 1)[0] if texts else "이미지에서 텍스트를 찾을 수 없습니다."

    except Exception as e:
        return f"OCR 처리 중 오류 발생: {e}"


# ---------------------- 1. Gemini 연동 함수 ----------------------

def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    return genai.Client(api_key=api_key) if api_key else None

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word, lemma, pos):
    client = get_gemini_client()
    if not client:
        return {"ko_meanings": [f"'{word}'의 API 키 없음 (GEMINI_API_KEY 설정 필요)"], "examples": []}
    
    SYSTEM_PROMPT = "너는 러시아어-한국어 학습 도우미이다. 러시아어 단어에 대해 간단한 한국어 뜻과 예문을 최대 두 개만 제공한다. 한국어 뜻을 제공할 때 격 정보, 문법 정보 등 불필요한 부가 정보는 절대 포함하지 않는다. 만약 동사(V)이면, 불완료상(imp)과 완료상(perf) 형태를 함께 제공해야 한다. 반드시 JSON만 출력한다."
    
    if pos == '동사':
        prompt = f"""{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}
{{ "ko_meanings": ["뜻1", "뜻2"], "aspect_pair": {{"imp": "불완료상 동사", "perf": "완료상 동사"}}, "examples": [ {{"ru": "예문1", "ko": "번역1"}}, {{"ru": "예문2", "ko": "번역2"}} ] }}
"""
    else:
        prompt = f"""{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}
{{ "ko_meanings": ["뜻1", "뜻2"], "examples": [ {{"ru": "예문1", "ko": "번역1"}}, {{"ru": "예문2", "ko": "번역2"}} ] }}
"""
    
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = res.text.strip()
    
    try:
        # JSON 파싱 로직
        if text.startswith("```"):
            text = text.strip("`")
            lines = text.splitlines()
            if lines and lines[0].lower().startswith("json"):
                text = "\n".join(lines[1:])
            elif lines:
                text = "\n".join(lines)
                
        start_index = text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_text = text[start_index : end_index + 1]
        else:
            json_text = text
            
        data = json.loads(json_text)
        
        if 'examples' in data and len(data['examples']) > 2:
            data['examples'] = data['examples'][:2]
        return data
        
    except json.JSONDecodeError:
        return {"ko_meanings": ["JSON 파싱 오류"], "examples": []}

# ---------------------- 2. 텍스트 번역 함수 ----------------------

@st.cache_data(show_spinner="텍스트를 한국어로 번역하는 중...")
def translate_text(russian_text, highlight_words):
    client = get_gemini_client()
    if not client:
        return "Gemini API 키가 설정되지 않아 번역을 수행할 수 없습니다."
        
    phrases_to_highlight = ", ".join([f"'{w}'" for w in highlight_words])
    
    SYSTEM_INSTRUCTION = '''너는 번역가이다. 요청된 러시아어 텍스트를 문맥에 맞는 자연스러운 한국어로 번역하고, 절대로 다른 설명, 옵션, 질문, 부가적인 텍스트를 출력하지 않는다. 오직 최종 번역 텍스트만 출력한다.'''

    if phrases_to_highlight:
        translation_prompt = f"""
        **반드시 아래 러시아어 단어/구의 한국어 번역이 등장하면, 그 한국어 번역 단어/구를 `<PHRASE_START>`와 `<PHRASE_END>` 마크업으로 감싸야 해.**

        러시아어 텍스트: '{russian_text}'
        마크업 대상 러시아어 단어/구: {phrases_to_highlight}
        """
    else:
        translation_prompt = f"원본 러시아어 텍스트: '{russian_text}'"

    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=translation_prompt,
            config={"system_instruction": SYSTEM_INSTRUCTION}
        )
        translated = res.text.strip()
        
        # 후처리: 마크업을 HTML Span 태그로 변환
        selected_class = "word-selected"
        translated = translated.replace("<PHRASE_START>", f'<span class="{selected_class}">')
        translated = translated.replace("<PHRASE_END>", '</span>')

        return translated

    except Exception as e:
        return f"번역 오류 발생: {e}"


# ---------------------- 3. 전역 스타일 정의 ----------------------

st.markdown("""
<style>
    /* 텍스트 영역 가독성 */
    .text-container {
        line-height: 2.0;
        margin-bottom: 20px;
        font-size: 1.25em;
    }
    /* 선택/검색된 단어/구 하이라이팅 + 밑줄 */
    .word-selected {
        color: #007bff !important; 
        font-weight: bold;
        background-color: #e0f0ff; 
        padding: 2px 0px;
        border-bottom: 3px solid #007bff; 
        border-radius: 2px;
    }
    .search-link-container {
        display: flex;
        gap: 10px;
        margin-top: 15px;
        flex-wrap: wrap;
    }
</style>
""", unsafe_allow_html=True)


# 🌟 4. 버튼 클릭 시 텍스트를 로드하는 콜백 함수 정의
def load_default_text():
    st.session_state.input_text_area = NEW_DEFAULT_TEXT 
    st.session_state.translated_text = ""
    st.session_state.selected_words = []
    st.session_state.clicked_word = None
    st.session_state.word_info = {}
    st.session_state.current_search_query = ""
    st.session_state.last_processed_query = ""


# ---------------------- 4. UI 배치 및 메인 로직 ----------------------

# --- 4.1. OCR 및 텍스트 입력 섹션 ---
st.subheader("🖼️ 이미지에서 텍스트 추출(업데이트 예정)")
uploaded_file = st.file_uploader("JPG, PNG 등 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    ocr_result = detect_text_from_image(image_bytes)
    
    if ocr_result and not ocr_result.startswith(("OCR API 키", "Vision API 오류")):
        st.session_state.ocr_output_text = ocr_result
        st.session_state.input_text_area = ocr_result
        st.session_state.translated_text = ""
        st.success("이미지에서 텍스트 추출 완료!")
    else:
        st.error(ocr_result)

# 🌟 5. 텍스트 반영 버튼 추가
st.button(
    "📚 중급러시아어연습 텍스트 반영하기", 
    on_click=load_default_text, 
    help="교재 연습용 텍스트를 입력창에 반영합니다."
)

st.subheader("📝 분석 대상 텍스트") 
current_text = st.text_area(
    "러시아어 텍스트를 입력하거나 위에 업로드된 텍스트를 수정하세요", 
    value=st.session_state.input_text_area, 
    height=150, 
    key="input_text_area"
)


# 텍스트가 수정되면 상태 업데이트 및 번역/분석 상태 초기화
if current_text != st.session_state.last_processed_text:
    st.session_state.translated_text = ""
    st.session_state.selected_words = []
    st.session_state.clicked_word = None
    st.session_state.word_info = {}
    st.session_state.current_search_query = ""

# --- 4.2. 단어 검색창 및 로직 ---
st.divider()
st.subheader("🔍 단어/구 검색") 
manual_input = st.text_input("단어 또는 구를 입력하고 Enter (예: 'идёт по улице')", key="current_search_query")

if manual_input and manual_input != st.session_state.get("last_processed_query"):
    if manual_input not in st.
