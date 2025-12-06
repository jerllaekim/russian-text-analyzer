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

# ---------------------- 0. 초기 설정 및 세션 상태 ----------------------

NEW_DEFAULT_TEXT = """Том живёт в Санкт-Петербурге уже несколько месяцев. В субботу, когда была хорошая погода, Том решил пойти в Исаакиевский собор. Том давно мечтал побывать в этом соборе. Исаакиевский собор — одно из самых высоких зданий в Санкт-Петербурге, его можно увидеть
даже издалека. Когда Том гулял по центру города, он отовсюду видел золотой купол собора. Сначала Том решил осмотреть собор снаружи. Он пришёл на Исаакиевскую площадь — отсюда открывается прекрасный вид на собор. Потом Том подошёл к собору поближе, осмотрел его спереди, сзади, 2 раза обошёл вокруг собора, потом вошёл внутрь. Внутри собор очень красивый. Том прочитал, что купол собора — третий по величине в Европе. Том поднял голову вверх и увидел, что под куполом «летает» серебряный голубь. Том посмотрел вокруг: впереди, сзади, справа, слева — везде были красивые иконы.
Потом Том решил подняться на колоннаду собора. В выходной день в соборе было много туристов: одни поднимались вверх, другие спускались вниз. Том тоже поднялся вверх. Оттуда, сверху, с высоты 43 (сорока трёх) метров, открывается прекрасный вид на центр города. Том увидел Дворцовую площадь, Петропавловскую крепость, крыши домов, а над крышами летали птицы.
Тому очень понравилась экскурсия. Он посоветовал друзьям посетить собор и обязательно подняться на колоннаду."""

DEFAULT_TEST_TEXT = "Человек идёт по улице. Это тестовая строка. Хорошо. Я часто читаю эту книгу."


# ---------------------- 0.1. 페이지 설정 및 배너 삽입 ----------------------
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")

# 🌟 배너 이미지 파일 경로
IMAGE_FILE_PATH = "banner.png"

try:
    st.image(IMAGE_FILE_PATH, use_column_width=True)
except FileNotFoundError:
    st.warning(f"배너 이미지 파일 ({IMAGE_FILE_PATH})을 찾을 수 없습니다. GitHub 저장소에 이미지를 업로드하고 파일명을 확인해주세요.")
    st.markdown("###")

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

# ---------------------- 0.2. YouTube 임베드 함수 및 ID 정의 ----------------------

# 📌 🚨 중요: 여기에 홍보 영상의 YouTube ID를 넣어주세요. (문자열로, 따옴표 안에)
YOUTUBE_VIDEO_ID = "wJ65i_gDfT0" 

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
    # 공백이 포함된 경우 '구 형태'로 반환 (구 분석 기능을 위함)
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


# ---------------------- 3. 전역 스타일 정의 (폰트 및 유튜브 반응형 CSS 포함) ----------------------

st.markdown("""
<style>
    /* 폰트 적용: Nanum Gothic 웹 폰트 (UI 글씨체 변경) */
    @import url('[https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap](https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap)');
    
    html, body, .stApp {
        font-family: 'Nanum Gothic', sans-serif !important; 
    }
    
    /* YouTube 비디오를 위한 반응형 컨테이너 */
    .video-responsive {
        overflow: hidden;
        padding-bottom: 56.25%; /* 16:9 비율 (9 / 16 * 100) */
        position: relative;
        height: 0;
    }
    .video-responsive iframe {
        left: 0;
        top: 0;
        height: 100%;
        width: 100%;
        position: absolute;
    }
    .video-container-wrapper {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    
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
    /* 버튼 스타일 */
    .stButton>button {
        background-color: #f0f2f6;
        color: #333;
        border: 1px solid #ccc;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #e8e8e8;
        border-color: #aaa;
    }
    /* 기타 UI 조정 */
    .main .stImage {
        padding: 0;
        margin: 0;
    }
    .st-emotion-cache-1215r6w {
        margin-top: 0rem !important;
        padding-top: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- 4. 버튼 클릭 시 텍스트를 로드하는 콜백 함수 정의 ----------------------
def load_default_text():
    st.session_state.input_text_area = NEW_DEFAULT_TEXT
    st.session_state.translated_text = ""
    st.session_state.selected_words = []
    st.session_state.clicked_word = None
    st.session_state.word_info = {}
    st.session_state.current_search_query = ""
    st.session_state.last_processed_query = ""


# ---------------------- 5. 하이라이팅 로직 함수 정의 ----------------------
def get_highlighted_html(text_to_process, highlight_words):
    selected_class = "word-selected"
    display_html = text_to_process
    
    highlight_candidates = sorted(
        [word for word in highlight_words if word.strip()],
        key=len,
        reverse=True
    )

    for phrase in highlight_candidates:
        escaped_phrase = re.escape(phrase)
        
        if ' ' in phrase:
            # 구(Phrase) 검색
            display_html = re.sub(
                f'({escaped_phrase})',
                f'<span class="{selected_class}">\\1</span>',
                display_html
            )
        else:
            # 단어(Word) 검색 (\b는 단어 경계)
            pattern = re.compile(r'\b' + escaped_phrase + r'\b')
            display_html = pattern.sub(
                f'<span class="{selected_class}">{phrase}</span>',
                display_html
            )
    
    return f'<div class="text-container">{display_html}</div>'


# ---------------------- 6. UI 배치 및 메인 로직 ----------------------

# --- 6.1. OCR 및 텍스트 입력 섹션 ---
st.subheader("이미지에서 텍스트 추출(업데이트 예정)")
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

# 텍스트 반영 버튼 추가
st.button(
    "중급러시아어연습 텍스트 반영하기(교재 2권 44페이지)",
    on_click=load_default_text,
    help="교재 연습용 텍스트를 입력창에 반영합니다."
)

st.subheader("분석 대상 텍스트")
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


# --- 6.2. 단어 검색창 및 로직 ---
st.divider()
st.subheader("🔍 단어/구 검색")
manual_input = st.text_input("단어 또는 구를 입력하고 Enter (예: 'идёт по улице')", key="current_search_query")

if manual_input and manual_input != st.session_state.get("last_processed_query"):
    if manual_input not in st.session_state.selected_words:
        st.session_state.selected_words.append(manual_input)
    
    st.session_state.clicked_word = manual_input
    
    with st.spinner(f"'{manual_input}'에 대한 정보 분석 중..."):
        clean_input = manual_input
        lemma = lemmatize_ru(clean_input)
        pos = get_pos_ru(clean_input)
        try:
            info = fetch_from_gemini(clean_input, lemma, pos)
            # 기본형(lemma) 기준으로 정보 저장. 단, 현재 검색어(token)가 다르면 업데이트
            if lemma not in st.session_state.word_info or st.session_state.word_info.get(lemma, {}).get('loaded_token') != clean_input:
                st.session_state.word_info[lemma] = {**info, "loaded_token": clean_input, "pos": pos}
        except Exception as e:
            st.error(f"Gemini 오류: {e}")
            
    st.session_state.last_processed_query = manual_input

st.markdown("---")


# ---------------------- 7. 텍스트 하이라이팅 및 상세 정보 레이아웃 ----------------------

# 레이아웃: 왼쪽(원문), 오른쪽(상세정보 + 영상)
left, right = st.columns([2, 1])


with left:
    st.subheader("러시아어 텍스트 원문")
    
    # --- TTS 버튼 및 강세 링크 ---
    col_tts, col_accent = st.columns([1, 2])
    
    with col_tts:
        ELEVENLABS_URL = "https://elevenlabs.io/"
        st.markdown(
            f"[▶️ 텍스트 음성 듣기 (ElevenLabs)]({ELEVENLABS_URL})",
            unsafe_allow_html=False
        )

    with col_accent:
        ACCENT_ONLINE_URL = "[https://russiangram.com/](https://russiangram.com/)"
        
        st.markdown(
            f"🔊 [강세 표시 사이트로 이동 (russiangram.com)]({ACCENT_ONLINE_URL})",
            unsafe_allow_html=False
        )
        st.info("⬆️ 음성 듣기 및 강세 확인을 위해 외부 사이트 링크를 사용합니다. 새 탭으로 열립니다.")


    # 러시아어 텍스트 하이라이팅 출력 (current_text 사용)
    ru_html = get_highlighted_html(current_text, st.session_state.selected_words)
    st.markdown(ru_html, unsafe_allow_html=True)
    
    st.markdown("---")


    # 초기화 버튼
    def reset_all_state():
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.session_state.current_search_query = ""
        st.session_state.input_text_area = DEFAULT_TEST_TEXT
        st.session_state.ocr_output_text = ""
        st.session_state.translated_text = ""
        st.session_state.last_processed_text = ""


    st.button("선택 및 검색 초기화", key="reset_button", on_click=reset_all_state)
    

# ---------------------- 7.2. 단어 상세 정보 (right 컬럼) + 영상 삽입 ----------------------
with right:
    st.subheader("단어 상세 정보")
    
    current_token = st.session_state.clicked_word
    
    if current_token:
        clean_token = current_token
        lemma = lemmatize_ru(clean_token)
        info = st.session_state.word_info.get(lemma, {})

        if info and "ko_meanings" in info:
            pos = info.get("pos", "품사")
            aspect_pair = info.get("aspect_pair")
            
            # --- 1. 구 전체의 정보 표시 ---
            st.markdown(f"### **{clean_token}**")
            
            if pos == '동사' and aspect_pair:
                st.markdown(f"**기본형 (불완료상):** *{aspect_pair.get('imp', lemma)}*")
                st.markdown(f"**완료상:** *{aspect_pair.get('perf', '정보 없음')}*")
                st.markdown(f"**품사:** {pos}")
            elif pos == '구 형태': 
                st.markdown(f"**구(句) 형태:** *{lemma}*")
                st.markdown(f"**품사:** {pos} (개별 단어 분석을 참고하세요)")
            else:
                st.markdown(f"**기본형 (Lemma):** *{lemma}* ({pos})")
            
            st.divider()

            ko_meanings = info.get("ko_meanings", [])
            examples = info.get("examples", [])

            if ko_meanings:
                st.markdown("#### 한국어 뜻")
                for m in ko_meanings:
                    st.markdown(f"- **{m}**")

            if examples:
                st.markdown("#### 📖 예문")
                for ex in examples:
                    st.markdown(f"- {ex.get('ru', '')}")
                    st.markdown(f" → {ex.get('ko', '')}")
            else:
                if ko_meanings and ko_meanings[0].startswith(f"'{current_token}'의 API 키 없음"):
                    st.warning("API 키가 설정되지 않아 예문을 불러올 수 없습니다.")
                elif ko_meanings and ko_meanings[0] == "JSON 파싱 오류":
                    st.error("Gemini API 정보 오류.")
                else:
                    st.info("예문 정보가 없습니다.")
            
            # --- 2. 구 안에 있는 개별 단어 정보 표시 (요청 사항 반영: 간략 뜻 로드) ---
            if pos == '구 형태':
                st.markdown("---")
                st.markdown("#### 낱말(토큰) 분석")
                
                individual_words = clean_token.split() 
                
                for word in individual_words:
                    # 문장부호 제거 후 처리 (원형 추출 정확도를 높이기 위함)
                    processed_word = re.sub(r'[.,!?;:"]', '', word) 
                    
                    if not processed_word:
                        continue
                        
                    token_lemma = lemmatize_ru(processed_word)
                    token_pos = get_pos_ru(processed_word)
                    token_info = st.session_state.word_info.get(token_lemma)
                    
                    # 캐시에 정보가 없거나 구 정보만 있을 경우, Gemini API를 호출하여 뜻만 가져옴
                    if not token_info or token_info.get('pos') == '구 형태':
                        try:
                            # 기본형(lemma)만으로 API 호출하여 간략 정보를 가져옴
                            loaded_info = fetch_from_gemini(token_lemma, token_lemma, token_pos)
                            
                            if loaded_info.get("ko_meanings") and loaded_info["ko_meanings"][0] != "JSON 파싱 오류":
                                # 가져온 정보를 캐시에 저장
                                st.session_state.word_info[token_lemma] = {
                                    **loaded_info, 
                                    "loaded_token": token_lemma, 
                                    "pos": token_pos
                                }
                                token_info = st.session_state.word_info[token_lemma]
                            else:
                                st.markdown(f"**{word}** (`{token_lemma}`) → 뜻 정보 로드 실패")
                                continue
                        except Exception as e:
                            st.markdown(f"**{word}** (`{token_lemma}`) → API 오류")
                            continue

                    # *간략화된 단어 분석 결과를 표시*
                    if token_info:
                        token_pos = token_info.get("pos", "품사")
                        token_meanings = token_info.get("ko_meanings", [])
                        
                        display_meaning = "; ".join(token_meanings[:1])
                        
                        # 요청하신 간략한 출력 형식
                        st.markdown(f"**{word}** (`{token_lemma}` - {token_pos}) → **{display_meaning}**")
                    
            # --- 3. 외부 검색 링크 ---
            st.markdown("---")
            encoded_query = urllib.parse.quote(clean_token)
            
            multitran_url = f"[https://www.multitran.com/m.exe?s=](https://www.multitran.com/m.exe?s=){encoded_query}&l1=1&l2=2"
            corpus_url = f"[http://search.ruscorpora.ru/search.xml?text=](http://search.ruscorpora.ru/search.xml?text=){encoded_query}&env=alpha&mode=main&sort=gr_tagging&lang=ru&nodia=1"
            
            st.markdown("#### 🌐 외부 검색")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"[Multitran 검색]({multitran_url})")
            
            with col2:
                st.markdown(f"[국립 코퍼스 검색]({corpus_url})")
            
        else:
            st.warning("단어 정보를 불러오는 중이거나 오류가 발생했습니다.")
            
    else:
        st.info("검색창에 단어를 입력하면 여기에 상세 정보가 표시됩니다.")
        
    # --- 10. 홍보 영상 삽입 (단어 상세 정보 섹션의 맨 아래) ---
    st.markdown("---")
    st.subheader("🎬 프로젝트 홍보 영상")
    if YOUTUBE_VIDEO_ID:
        video_html = youtube_embed_html(YOUTUBE_VIDEO_ID) 
        st.markdown(video_html, unsafe_allow_html=True)
        st.caption(f"YouTube 영상 ID: {YOUTUBE_VIDEO_ID}") 
    else:
        st.warning("홍보 영상을 표시하려면 YOUTUBE_VIDEO_ID를 설정해주세요.")


# ---------------------- 8. 하단: 누적 목록 + CSV ----------------------
st.divider()
# 문구 수정 반영
st.subheader("단어 목록 (기본형 기준)")

selected = st.session_state.selected_words
word_info = st.session_state.word_info

if word_info:
    rows = []
    processed_lemmas = set()
    
    for tok in selected:
        clean_tok = tok
        lemma = lemmatize_ru(clean_tok)
        if lemma not in processed_lemmas and lemma in word_info:
            info = word_info[lemma]
            if info.get("ko_meanings") and info["ko_meanings"][0] != "JSON 파싱 오류":
                pos = info.get("pos", "품사")
                
                if pos == '동사' and info.get("aspect_pair"):
                    imp = info['aspect_pair'].get('imp', lemma)
                    perf = info['aspect_pair'].get('perf', '정보 없음')
                    base_form = f"{imp} / {perf}"
                else:
                    base_form = lemma

                short = "; ".join(info["ko_meanings"][:2])
                short = f"({pos}) {short}"

                rows.append({"기본형": base_form, "대표 뜻": short})
                processed_lemmas.add(lemma)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)
    else:
        st.info("선택된 단어의 정보가 로드 중이거나, 표시할 정보가 없습니다.")


# ---------------------- 9. 하단: 한국어 번역본 ----------------------
st.divider()
st.subheader("한국어 번역본")

if st.session_state.translated_text == "" or current_text != st.session_state.last_processed_text:
    st.session_state.translated_text = translate_text(
        current_text,
        st.session_state.selected_words
    )
    st.session_state.last_processed_text = current_text

translated_text = st.session_state.translated_text

if translated_text.startswith("Gemini API 키가 설정되지"):
    st.error(translated_text)
elif translated_text.startswith("번역 오류 발생"):
    st.error(translated_text)
else:
    st.markdown(f'<div class="text-container" style="color: #333; font-weight: 500;">{translated_text}</div>', unsafe_allow_html=True)


# ---------------------- 11. 저작권 표시 (페이지 최하단) ----------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.75em; color: #888;">
    이 페이지는 연세대학교 노어노문학과 25-2 러시아어 교육론 5팀의 프로젝트 결과물입니다.
    <br>
    본 페이지의 내용, 기능 및 데이터를 학습 목적 이외의 용도로 무단 복제, 배포, 상업적 이용할 경우,
    관련 법령에 따라 민사상 손해배상 청구 및 형사상 처벌을 받을 수 있습니다.
</div>
""", unsafe_allow_html=True)
