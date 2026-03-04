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

def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    return genai.Client(api_key=api_key) if api_key else None

@st.cache_resource(show_spinner=False)
def get_vision_client():
    try:
        # Secrets에서 JSON 키를 불러옴
        key_json = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON") 
        
        if not key_json:
            st.warning("Secrets 변수 'GOOGLE_APPLICATION_CREDENTIALS_JSON'이 설정되지 않았습니다.")
            return None

        import google.auth
        import google.cloud.vision
        
        # 🌟🌟🌟 1. JSON 유효성 검사 및 로드 시도 (Gemini 디버깅 제거, 원본 오류 포착) 🌟🌟🌟
        try:
            # 유니코드 제어 문자를 강제로 무시하고 ASCII로 클린하게 만듭니다. (최대한 오류 회피)
            cleaned_json_string = key_json.encode('ascii', 'ignore').decode('ascii')
            key_data = json.loads(cleaned_json_string)
        except Exception as json_error:
            # JSON 로드 실패 시, Python의 원본 오류를 출력
            st.error("🚨 Secrets JSON 파싱 오류 발생: 유효하지 않은 문자 포함")
            st.code(f"Secrets Value Start:\n{key_json[:300]}...\n\nJSON Error: {str(json_error)}", language="python")
            return None
        
        # 2. Credential 생성 및 클라이언트 반환
        credentials, _ = google.auth.load_credentials_from_dict(key_data)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        return client
        
    except Exception as e:
        st.error(f"Vision API 클라이언트 초기화 오류: {e}")
        return None

# 🌟 TTL=3600초 (1시간) 설정 및 타임아웃 30초 추ㅊ
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

def get_word_info_schema(is_verb: bool):
    """Gemini 응답의 JSON 스키마를 정의합니다."""
    schema = {
        "type": "object",
        "properties": {
            "ko_meanings": {"type": "array", "items": {"type": "string"}, "description": "단어의 한국어 뜻 목록"},
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ru": {"type": "string", "description": "러시아어 예문"},
                        "ko": {"type": "string", "description": "한국어 번역"}
                    },
                    "required": ["ru", "ko"]
                },
                "description": "최대 두 개의 예문과 그 번역"
            }
        },
        "required": ["ko_meanings", "examples"]
    }

    if is_verb:
        schema['properties']['aspect_pair'] = {
            "type": "object",
            "properties": {
                "imp": {"type": "string", "description": "불완료상 동사"},
                "perf": {"type": "string", "description": "완료상 동사"}
            },
            "required": ["imp", "perf"],
            "description": "동사의 상(aspect) 쌍"
        }
        schema['required'].append('aspect_pair')
    
    return schema

# 🌟 TTL=300초 (5분) 설정: 캐시를 사용하여 API 호출 횟수 관리
@st.cache_data(show_spinner=False, ttl=60 * 5) 
def fetch_from_gemini(word, lemma, pos):
    client = get_gemini_client()
    if not client:
        return {"ko_meanings": [f"'{word}'의 API 키 없음 (GEMINI_API_KEY 설정 필요)"], "examples": []}
    
    is_verb = (pos == '동사')
    
    # JSON Schema와 system_instruction 설정하여 JSON 출력을 강제함
    config = {
        "system_instruction": "너는 러시아어-한국어 학습 도우미이다. 요청된 단어에 대한 정보를 제공하며, 절대로 부가적인 설명을 넣지 말고 오직 요청된 JSON 형식의 데이터만 출력한다. 한국어 뜻은 간단해야 한다.",
        "response_mime_type": "application/json",
        "response_schema": get_word_info_schema(is_verb),
    }
    
    prompt = f"러시아어 단어: {word}. 기본형: {lemma}. 품사: {pos}. 정보를 요청합니다."

    try:
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config
        )
        
        data = json.loads(res.text) 
        
        if 'examples' in data and len(data['examples']) > 2:
            data['examples'] = data['examples'][:2]
            
        return data
    
    except Exception as e:
        error_msg = str(e)
        if "RESOURCE_EXHAUSTED" in error_msg:
             return {"ko_meanings": [f"API 할당량 초과 오류: {error_msg.split(',')[0]}..."], "examples": []}
        return {"ko_meanings": [f"API 호출 또는 JSON 파싱 오류: {error_msg}"], "examples": []}


# ---------------------- 2. 텍스트 번역 함수 (TTL 10분 설정) ----------------------

# 🌟 TTL=600초 (10분) 설정: 캐시를 사용하여 API 호출 횟수 관리
@st.cache_data(show_spinner="텍스트를 한국어로 번역하는 중...", ttl=60 * 10)
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
    /* 폰트 적용: Nanum Gothic 웹 폰트 (UI 글씨체 변경) */
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
    
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
    
    # OCR 결과 출력 로직
    if ocr_result and not ocr_result.startswith(("OCR API 클라이언트 초기화 실패", "Vision API 오류", "OCR 처리 중 오류 발생")):
        st.session_state.ocr_output_text = ocr_result
        st.session_state.input_text_area = ocr_result
        st.session_state.translated_text = ""
        st.success("이미지에서 텍스트 추출 완료!")
    else:
        st.error(ocr_result)
# ---------------------- [추가] 6.0. 러시아어 괄호 텍스트 엑셀 변환 기능 ----------------------

def to_plural_nominative(word):
    parsed = morph.parse(word)[0] # 기존 코드의 mystem 대신 pymorphy2 사용 권장 (복수주격 변환용)
    if "plur" in parsed.tag and "nomn" not in parsed.tag:
        for form in parsed.lexeme:
            if "plur" in form.tag and "nomn" in form.tag:
                return form.word
    return word

def save_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed Data')
        ws = writer.sheets['Processed Data']
        ws.column_dimensions['A'].width = 100
        ws.column_dimensions['B'].width = 50
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=1):
            for cell in row:
                cell.alignment = cell.alignment.copy(wrap_text=True)
    return output.getvalue()

# UI 부분
with st.expander("📂 괄호 텍스트 분석 및 엑셀 다운로드 (교재 정리용)"):
    st.write("텍스트 파일(.txt)을 업로드하면 괄호 안의 단어를 분석하여 엑셀 파일로 만들어드립니다.")
    excel_upload = st.file_uploader("분석할 TXT 파일을 선택하세요", type=["txt"], key="excel_uploader")
    
    if excel_upload:
        from io import BytesIO
        import pymorphy2
        
        # pymorphy2는 루프 밖에서 선언하거나 캐싱하는 것이 좋지만, 
        # 일단 기능 통합을 위해 내부에서 선언합니다.
        morph_py = pymorphy2.MorphAnalyzer()
        
        content = excel_upload.getvalue().decode("utf-8")
        processed_data = []
        
        # 분석 로직 실행
        with st.spinner('엑셀 파일 생성 중...'):
            lines = content.splitlines()
            for line in lines:
                if not line.strip(): continue
                sentences = re.split(r'(?<!\w\.\w.)(?<![А-Яа-я]\.)(?<![A-Za-z]\.)(?<!\.\d)[.!?]\s+', line.strip())
                for sentence in sentences:
                    if not sentence: continue
                    original_bracket_contents = []

                    def lemmatize_brackets(match):
                        original_text = match.group(1)
                        words = original_text.split()
                        # russian_prepositions는 상단에 정의되어 있어야 함
                        filtered_words = [w for w in words if w.lower() not in russian_prepositions]
                        if not filtered_words: return ""
                        
                        lemmatized_words = []
                        for word in filtered_words:
                            p = morph_py.parse(word)[0]
                            lex = p.normal_form
                            if "PRTF" in p.tag: lex = f"{lex} (형동사)"
                            if "GRND" in p.tag: lex = f"{lex} (부동사)"
                            if "plur" in p.tag: # 복수 주격 변환
                                for form in p.lexeme:
                                    if "plur" in form.tag and "nomn" in form.tag:
                                        lex = form.word
                            lemmatized_words.append(lex)
                        original_bracket_contents.append(original_text)
                        return f"({', '.join(lemmatized_words)})"

                    proc_sent = re.sub(r'\((.*?)\)', lemmatize_brackets, sentence)
                    if original_bracket_contents:
                        processed_data.append({
                            "Sentence": proc_sent.strip(),
                            "Original Words": ", ".join(original_bracket_contents),
                        })
            
            if processed_data:
                df_excel = pd.DataFrame(processed_data)
                excel_bytes = save_to_excel(df_excel)
                st.success("변환 완료!")
                st.download_button(
                    label="📥 분석된 엑셀 파일 다운로드",
                    data=excel_bytes,
                    file_name=f"analysis_{excel_upload.name.replace('.txt', '')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.warning("괄호()가 포함된 문장을 찾지 못했습니다.")

# ----------------------------------------------------------------------------------

st.subheader("분석 대상 텍스트") # 이 줄이 원래 코드의 295번 줄입니다.
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
st.subheader("단어/구 검색")
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
        ACCENT_ONLINE_URL = "https://russiangram.com/"
        
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
    

# ---------------------- 7.2. 단어 상세 정보 (right 컬럼) ----------------------
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
                if ko_meanings and ko_meanings[0].startswith("'{current_token}'의 API 키 없음"):
                    st.warning("API 키가 설정되지 않아 예문을 불러올 수 없습니다.")
                elif ko_meanings and ko_meanings[0].startswith(("API 할당량 초과 오류", "API 호출 또는 JSON 파싱 오류")):
                    st.error(f"Gemini API 오류: {ko_meanings[0]}")
                else:
                    st.info("예문 정보가 없습니다.")
            
            # --- 2. 구 안에 있는 개별 단어 정보 표시 (요청 사항 반영: 간략 뜻 로드) ---
            if pos == '구 형태':
                st.markdown("---")
                st.markdown("#### 낱말(토큰) 분석")
                
                individual_words = clean_token.split() 
                
                for word in individual_words:
                    processed_word = re.sub(r'[.,!?;:"]', '', word) 
                    
                    if not processed_word:
                        continue
                        
                    token_lemma = lemmatize_ru(processed_word)
                    token_pos = get_pos_ru(processed_word)
                    token_info = st.session_state.word_info.get(token_lemma)
                    
                    if not token_info or token_info.get('pos') == '구 형태':
                        try:
                            loaded_info = fetch_from_gemini(token_lemma, token_lemma, token_pos)
                            
                            if loaded_info.get("ko_meanings") and not loaded_info["ko_meanings"][0].startswith(("API 할당량 초과 오류", "API 호출 또는 JSON 파싱 오류")):
                                st.session_state.word_info[token_lemma] = {
                                    **loaded_info, 
                                    "loaded_token": token_lemma, 
                                    "pos": token_pos
                                }
                                token_info = st.session_state.word_info[token_lemma]
                            else:
                                st.markdown(f"**{word}** (`{token_lemma}`) → 뜻 정보 로드 실패 또는 오류")
                                continue
                        except Exception as e:
                            st.markdown(f"**{word}** (`{token_lemma}`) → API 호출 오류")
                            continue

                    if token_info:
                        token_pos = token_info.get("pos", "품사")
                        token_meanings = token_info.get("ko_meanings", [])
                        display_meaning = "; ".join(token_meanings[:1])
                        
                        st.markdown(f"**{word}** (`{token_lemma}` - {token_pos}) → **{display_meaning}**")
                    
            # --- 3. 외부 검색 링크 ---
            st.markdown("---")
            encoded_query = urllib.parse.quote(clean_token)
            
            multitran_url = f"https://www.multitran.com/m.exe?s={encoded_query}&l1=1&l2=2"
            corpus_url = f"http://search.ruscorpora.ru/search.xml?text={encoded_query}&env=alpha&mode=main&sort=gr_tagging&lang=ru&nodia=1"
            
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


# ---------------------- 8. 하단: 누적 목록 + CSV ----------------------
st.divider()
st.subheader("단어 목록 (기본형 기준)")

selected = st.session_state.selected_words
word_info = st.session_state.word_info

# 🌟 중요: rows를 조건문 밖에서 빈 리스트로 먼저 초기화합니다.
rows = []

if word_info and selected:
    processed_lemmas = set()
    
    for tok in selected:
        clean_tok = tok
        lemma = lemmatize_ru(clean_tok)
        if lemma not in processed_lemmas and lemma in word_info:
            info = word_info[lemma]
            # API 오류가 없는 정상적인 데이터만 리스트에 추가
            if info.get("ko_meanings") and not info["ko_meanings"][0].startswith(("API 할당량 초과 오류", "API 호출 또는 JSON 파싱 오류")):
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

# 데이터프레임 표시
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)
    
    # --- 8.5. Quizlet 연동 섹션 ---
    st.markdown("#### 🎓 Quizlet으로 단어장 만들기")
    
    # Quizlet용 텍스트 생성
    quizlet_text = ""
    for row in rows:
        quizlet_text += f"{row['기본형']}\t{row['대표 뜻']}\n"
    
    col_copy, col_link = st.columns([2, 1])
    with col_copy:
        st.text_area("아래 텍스트를 복사해서 Quizlet '가져오기'에 붙여넣으세요:", 
                     value=quizlet_text, height=100)
    with col_link:
        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button("🚀 Quizlet 사이트로 이동", "https://quizlet.com/create-set", use_container_width=True)
else:
    st.info("검색창에 단어를 입력하여 분석하면 여기에 목록이 만들어집니다.")


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


# ---------------------- 10. 홍보 영상 삽입 (페이지 맨 아래로 이동) ----------------------

st.divider()

_, col_video = st.columns([1, 1])

with col_video:
    st.subheader("🎬 프로젝트 홍보 영상")
    if YOUTUBE_VIDEO_ID:
        video_html = youtube_embed_html(YOUTUBE_VIDEO_ID) 
        st.markdown(video_html, unsafe_allow_html=True)
        st.caption(f"YouTube 영상 ID: {YOUTUBE_VIDEO_ID}") 
    else:
        st.warning("홍보 영상을 표시하려면 YOUTUBE_VIDEO_ID를 설정해주세요.")


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
