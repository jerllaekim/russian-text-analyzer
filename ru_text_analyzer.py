import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai
from google.cloud import vision # Vision API 라이브러리 추가
import io

# ---------------------- 0. 초기 설정 및 세션 상태 ----------------------
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("🇷🇺 러시아어 텍스트 분석기 (OCR 통합)")

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
if "display_text" not in st.session_state:
    st.session_state.display_text = "Человек идёт по улице. Это тестовая строка. Хорошо. Я часто читаю эту книгу."


mystem = Mystem()

# ---------------------- 품사 변환 딕셔너리 (생략) ----------------------
# (이 부분은 원본 코드와 동일하므로 생략합니다.)
POS_MAP = {
    'S': '명사', 'V': '동사', 'A': '형용사', 'ADV': '부사', 'PR': '전치사',
    'CONJ': '접속사', 'INTJ': '감탄사', 'PART': '불변화사', 'NUM': '수사',
    'APRO': '대명사적 형용사', 'ANUM': '서수사', 'SPRO': '대명사',
}

# ---------------------- OCR 함수 (새로 추가) ----------------------

@st.cache_data(show_spinner="이미지에서 러시아어 텍스트를 추출하는 중...")
def detect_text_from_image(image_bytes):
    """Google Cloud Vision API를 사용하여 이미지에서 텍스트를 감지하고 추출합니다."""
    
    # 서비스 계정 키 설정 (Streamlit Secrets 또는 환경 변수 사용 권장)
    try:
        if st.secrets.get("GCP_SA_KEY"):
            # secrets에서 키 파일을 임시 저장
            with open("temp_sa_key.json", "w") as f:
                json.dump(st.secrets["GCP_SA_KEY"], f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_sa_key.json"
        elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
             # 환경 변수 또는 직접 파일 경로 설정이 필요함을 안내
            return "OCR API 키(GOOGLE_APPLICATION_CREDENTIALS)가 설정되지 않았습니다. Cloud Vision API 설정을 확인해주세요."

        # Vision API 클라이언트 초기화
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        
        # 텍스트 감지 요청
        response = client.text_detection(image=image)
        texts = response.text
        
        if response.error.message:
            return f"Vision API 오류: {response.error.message}"
            
        # 첫 번째 텍스트 블록(전체 텍스트) 반환
        return texts.split('\n', 1)[0] if texts else "이미지에서 텍스트를 찾을 수 없습니다."

    except Exception as e:
        return f"OCR 처리 중 오류 발생: {e}"


# ---------------------- Mystem 및 Gemini 함수 (원래 코드에서 재사용) ----------------------
@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    # (원래 코드와 동일)
    if ' ' in word.strip():
        return word.strip()
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        lemmas = mystem.lemmatize(word)
        return (lemmas[0] if lemmas else word).strip()
    return word

@st.cache_data(show_spinner=False)
def get_pos_ru(word: str) -> str:
    # (원래 코드와 동일)
    if ' ' in word.strip():
        return '관용구'
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        analysis = mystem.analyze(word)
        if analysis and 'analysis' in analysis[0] and analysis[0]['analysis']:
            grammar_info = analysis[0]['analysis'][0]['gr']
            pos_abbr = grammar_info.split('=')[0].split(',')[0].strip()
            return POS_MAP.get(pos_abbr, '품사')
    return '품사'

SYSTEM_PROMPT = "너는 러시아어-한국어 학습 도우미이다. 러시아어 단어에 대해 간단한 한국어 뜻과 예문을 최대 두 개만 제공한다. 만약 동사(V)이면, 불완료상(imp)과 완료상(perf) 형태를 함께 제공해야 한다. 반드시 JSON만 출력한다."

def make_prompt(word, lemma, pos):
    # (원래 코드와 동일)
    if pos == '동사':
        return f"""{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}
{{ "ko_meanings": ["뜻1", "뜻2"], "aspect_pair": {{"imp": "불완료상 동사", "perf": "완료상 동사"}}, "examples": [ {{"ru": "예문1", "ko": "번역1"}}, {{"ru": "예문2", "ko": "번역2"}} ] }}
"""
    else:
        return f"""{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}
{{ "ko_meanings": ["뜻1", "뜻2"], "examples": [ {{"ru": "예문1", "ko": "번역1"}}, {{"ru": "예문2", "ko": "번역2"}} ] }}
"""

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word, lemma, pos):
    # (원래 코드와 동일, API 키 설정 부분은 st.secrets를 통해 통합 관리)
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    client = genai.Client(api_key=api_key) if api_key else None
    
    if not client:
        return {"ko_meanings": [f"'{word}'의 API 키 없음 (GEMINI_API_KEY 설정 필요)"], "examples": []}
        
    prompt = make_prompt(word, lemma, pos)
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = res.text.strip()
    
    try:
        # JSON 파싱 로직 (원래 코드와 동일)
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
        # st.error(f"Gemini 응답을 JSON으로 디코딩하는 데 실패했습니다. 원본 텍스트 시작: {text[:100]}...")
        return {"ko_meanings": ["JSON 파싱 오류"], "examples": []}


# ---------------------- 2. 전역 스타일 정의 (밑줄 강조 추가) ----------------------

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
        border-bottom: 3px solid #007bff; /* 파란색 밑줄 추가 */
        border-radius: 2px;
    }
    .word-punctuation {
        padding: 0px 0px;
        margin: 0;
        display: inline-block;
        white-space: pre;
        font-size: 1.25em;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- 3. UI 배치 및 메인 로직 ----------------------

# --- 3.1. 이미지 업로드 섹션 (새로 추가) ---
st.subheader("🖼️ 이미지에서 텍스트 추출 (OCR)")
uploaded_file = st.file_uploader("JPG, PNG 등 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 파일을 바이트로 읽기
    image_bytes = uploaded_file.getvalue()
    
    # 텍스트 감지 실행
    ocr_result = detect_text_from_image(image_bytes)
    
    # 결과를 텍스트 영역에 반영
    if ocr_result and not ocr_result.startswith(("OCR API 키", "Vision API 오류")):
        st.session_state.ocr_output_text = ocr_result
        st.session_state.display_text = ocr_result # 분석 대상 텍스트 업데이트
        st.success("이미지에서 텍스트 추출 완료!")
    else:
        st.error(ocr_result)


# --- 3.2. 텍스트 입력창 (업로드된 텍스트 포함) ---
st.subheader("📝 분석 대상 텍스트")
# display_text 상태를 사용하여 OCR 결과를 보여줍니다.
current_text = st.text_area(
    "러시아어 텍스트를 입력하거나 위에 업로드된 텍스트를 수정하세요", 
    st.session_state.display_text, 
    height=150, 
    key="input_text_area"
)

# 텍스트 영역이 수정되면, display_text 상태도 업데이트합니다.
if current_text != st.session_state.display_text:
     st.session_state.display_text = current_text
     # 텍스트가 변경되면 모든 검색 상태를 초기화하는 것이 좋습니다.
     st.session_state.selected_words = []
     st.session_state.clicked_word = None
     st.session_state.word_info = {}
     st.session_state.current_search_query = ""

# --- 3.3. 단어 검색창 ---
st.divider()
st.subheader("🔍 직접 단어 검색")
manual_input = st.text_input("단어 입력 후 Enter (구 검색 시 공백 포함 입력)", key="current_search_query")

# ---------------------- 4. 검색 처리 로직 ----------------------

if manual_input and manual_input != st.session_state.get("last_processed_query"):
    # 1. 검색된 단어를 선택 목록에 추가
    if manual_input not in st.session_state.selected_words:
        st.session_state.selected_words.append(manual_input)
    
    # 2. 상세 정보 영역에 표시될 단어 업데이트
    st.session_state.clicked_word = manual_input
    
    # ************** 정보 로드 및 저장 **************
    with st.spinner(f"'{manual_input}'에 대한 정보 분석 중..."):
        lemma = lemmatize_ru(manual_input)
        pos = get_pos_ru(manual_input) # 품사 추출
        
        try:
            info = fetch_from_gemini(manual_input, lemma, pos)
            
            # 검색된 단어의 정보를 세션 상태에 저장 (품사 정보 추가)
            if lemma not in st.session_state.word_info or st.session_state.word_info.get(lemma, {}).get('loaded_token') != manual_input:
                st.session_state.word_info[lemma] = {**info, "loaded_token": manual_input, "pos": pos}  
                
        except Exception as e:
            st.error(f"Gemini 오류: {e}")
        
    st.session_state.last_processed_query = manual_input # 처리 완료된 쿼리 기록

st.markdown("---") 


# ---------------------- 5. 텍스트 하이라이팅 및 상세 정보 레이아웃 ----------------------

left, right = st.columns([2, 1])

# --- 5.1. 텍스트 하이라이팅 (left 컬럼 - **수정된 로직**) ---
with left:
    st.subheader("입력된 텍스트 하이라이팅")
    st.info("검색창에 단어를 입력하면 텍스트에서 해당 단어/구가 하이라이트됩니다.")

    selected_class = "word-selected"
    display_text_html = st.session_state.display_text # 분석 대상 텍스트 가져오기
    
    # 1. '구' 단위의 하이라이팅 처리를 위해 검색된 단어/구 목록을 역순으로 정렬 (긴 구 우선 처리)
    highlight_candidates = sorted(
        [word for word in st.session_state.selected_words if word.strip()],
        key=len,
        reverse=True
    )

    # 2. 텍스트 내에서 검색된 구/단어를 HTML 마크업으로 대체
    for phrase in highlight_candidates:
        # re.escape를 사용하여 특수 문자를 보호
        escaped_phrase = re.escape(phrase)
        
        # 띄어쓰기가 있는 '구'는 그대로 찾고, 단일 단어는 정확히 찾기
        if ' ' in phrase:
             # 're.sub'를 사용하여 반복 대체
            display_text_html = re.sub(
                f'({escaped_phrase})', 
                f'<span class="{selected_class}">\\1</span>', 
                display_text_html
            )
        else:
            # 단어 경계(\b)를 사용하여 정확한 단어 일치
            # 이미 HTML 태그가 적용된 부분은 무시하도록 로직을 더 복잡하게 만들 필요가 있으나,
            # 단순화를 위해 일단 \b로 대체
            pattern = re.compile(r'\b' + escaped_phrase + r'\b')
            display_text_html = pattern.sub(
                f'<span class="{selected_class}">{phrase}</span>', 
                display_text_html
            )
            
    # 3. 최종 HTML 출력
    final_html = f'<div class="text-container">{display_text_html}</div>'
    st.markdown(final_html, unsafe_allow_html=True)
    
    # 초기화 버튼을 위한 콜백 함수 정의
    def reset_all_state():
        # (원래 코드와 동일)
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.session_state.current_search_query = ""
        st.session_state.display_text = "Человек идёт по улице. Это тестовая строка. Хорошо." # 기본 텍스트로 복원
        st.session_state.ocr_output_text = ""


    st.markdown("---")
    st.button("🔄 선택 및 검색 초기화", key="reset_button", on_click=reset_all_state)
    
    if st.session_state.reset_button:
        st.rerun()

# --- 5.2. 단어 상세 정보 (right 컬럼 - 원래 코드와 동일) ---
with right:
    st.subheader("단어 상세 정보")
    
    current_token = st.session_state.clicked_word
    
    if current_token:
        lemma = lemmatize_ru(current_token)
        info = st.session_state.word_info.get(lemma, {})

        if info and "ko_meanings" in info:
            pos = info.get("pos", "품사") 
            aspect_pair = info.get("aspect_pair") 
            
            st.markdown(f"### **{current_token}**")
            
            if pos == '동사' and aspect_pair:
                st.markdown(f"**기본형 (불완료상):** *{aspect_pair.get('imp', lemma)}*")
                st.markdown(f"**완료상:** *{aspect_pair.get('perf', '정보 없음')}*")
                st.markdown(f"**품사:** {pos}")
            elif pos == '관용구':
                st.markdown(f"**구(句) 형태:** *{lemma}*")
                st.markdown(f"**품사:** {pos}")
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
        else:
            st.warning("단어 정보를 불러오는 중이거나 오류가 발생했습니다.")
            
    else:
        st.info("검색창에 단어를 입력하면 여기에 상세 정보가 표시됩니다.")

# ---------------------- 6. 하단: 누적 목록 + CSV (원래 코드와 동일) ----------------------
st.divider()
st.subheader("📝 선택한 단어 모음 (기본형 기준)")

selected = st.session_state.selected_words
word_info = st.session_state.word_info

if word_info:
    rows = []
    processed_lemmas = set()
    
    for tok in selected:
        lemma = lemmatize_ru(tok)
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

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV로 저장", csv_bytes, "russian_words.csv", "text/csv")
    else:
        st.info("선택된 단어의 정보가 로드 중이거나, 표시할 정보가 없습니다.")


# ---------------------- 7. 저작권 표시 (페이지 최하단) ----------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.75em; color: #888;">
    이 페이지는 **연세대학교 노어노문학과 25-2 러시아어 교육론 5팀의 프로젝트 결과물**입니다. 
    <br>
    본 페이지의 내용, 기능 및 데이터를 **학습 목적 이외의 용도로 무단 복제, 배포, 상업적 이용**할 경우, 
    관련 법령에 따라 **민사상 손해배상 청구 및 형사상 처벌**을 받을 수 있습니다.
</div>
""", unsafe_allow_html=True)
