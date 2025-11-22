import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai

# ---------------------- 0. 초기 설정 및 세션 상태 ----------------------
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("🇷🇺 러시아어 텍스트 분석기")

# 세션 상태 초기화
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None
if "word_info" not in st.session_state:
    st.session_state.word_info = {}
# 검색어는 st.text_input의 key로만 사용합니다.

mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    """단어의 기본형(lemma)을 추출합니다."""
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        lemmas = mystem.lemmatize(word)
        return (lemmas[0] if lemmas else word).strip()
    return word

# ---------------------- 1. Gemini 연동 함수 ----------------------

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = "너는 러시아어-한국어 학습을 돕는 도우미이다. 러시아어 단어에 대해 간단한 한국어 뜻과 예문을 최대 두 개만 제공한다. 반드시 JSON만 출력한다."
def make_prompt(word, lemma):
    return f"""{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}
{{ "ko_meanings": ["뜻1", "뜻2"], "examples": [ {{"ru": "예문1", "ko": "번역1"}}, {{"ru": "예문2", "ko": "번역2"}} ] }}
"""

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word, lemma):
    if not client:
        return {"ko_meanings": [f"'{word}'의 API 키 없음 (GEMINI_API_KEY 설정 필요)"], "examples": []}
        
    prompt = make_prompt(word, lemma)
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = res.text.strip()
    
    try:
        if text.startswith("```"):
            text = text.strip("`")
            lines = text.splitlines()
            if lines and lines[0].lower().startswith("json"):
                text = "\n".join(lines[1:])
            elif lines:
                 text = "\n.join(lines)
                 
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
        st.error(f"Gemini 응답을 JSON으로 디코딩하는 데 실패했습니다. 원본 텍스트 시작: {text[:100]}...")
        return {"ko_meanings": ["JSON 파싱 오류"], "examples": []}


# ---------------------- 2. 전역 스타일 정의 ----------------------

st.markdown("""
<style>
    /* 텍스트 영역 가독성 */
    .text-container {
        line-height: 2.0;
        margin-bottom: 20px;
        font-size: 1.25em;
    }
    /* 선택/검색된 단어 하이라이팅 */
    .word-selected {
        color: #007bff !important; 
        font-weight: bold;
        background-color: #e0f0ff; /* 배경색으로 선택 상태 표시 */
        padding: 2px 0px;
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

# 3.1. 텍스트 입력창 (최상단)
st.subheader("📝 텍스트 입력")
text = st.text_area("러시아어 텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка. Хорошо.", height=150, key="input_text_area")

# 3.2. 단어 검색창 (바로 다음)
st.divider()
st.subheader("🔍 단어 검색")
manual_input = st.text_input("단어 입력 후 Enter", key="current_search_query")

# ---------------------- 4. 검색 처리 로직 ----------------------

if manual_input:
    # 1. 검색된 단어를 선택 목록에 추가
    if manual_input not in st.session_state.selected_words:
        st.session_state.selected_words.append(manual_input)
    
    # 2. 상세 정보 영역에 표시될 단어 업데이트
    st.session_state.clicked_word = manual_input
    
    # ************** 정보 로드 및 저장 **************
    lemma = lemmatize_ru(manual_input)
    
    try:
        info = fetch_from_gemini(manual_input, lemma)
        
        # 검색된 단어의 정보를 세션 상태에 저장
        if lemma not in st.session_state.word_info or st.session_state.word_info.get(lemma, {}).get('loaded_token') != manual_input:
             st.session_state.word_info[lemma] = {**info, "loaded_token": manual_input} 
        
    except Exception as e:
        st.error(f"Gemini 오류: {e}")
        # 오류 발생 시 빈 정보로 대체
        info = {"ko_meanings": [f"정보 로드 오류: {e}"], "examples": []}

    st.markdown("---") # 검색 정보와 텍스트 하이라이트 구분선


# ---------------------- 5. 텍스트 하이라이팅 및 상세 정보 레이아웃 ----------------------

tokens_with_punct = re.findall(r"(\w+|[^\s\w]+|\s+)", text, flags=re.UNICODE)

left, right = st.columns([2, 1])

# --- 5.1. 텍스트 하이라이팅 (left 컬럼) ---
with left:
    st.subheader("입력된 텍스트 하이라이팅")

    # 텍스트 하이라이팅 표시 
    html_parts = ['<div class="text-container">']

    for tok in tokens_with_punct:
        if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
            # 단어인 경우: 하이라이팅
            css = "word-selected" if tok in st.session_state.selected_words else ""
            html_parts.append(f'<span class="{css}">{tok}</span>')
        else:
            # 구두점 또는 공백
            html_parts.append(f'<span class="word-punctuation">{tok}</span>')

    html_parts.append('</div>')
    
    st.markdown("".join(html_parts), unsafe_allow_html=True)
    
    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 선택 및 검색 초기화", key="reset_button"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.session_state.current_search_query = "" # 검색창 초기화
        st.rerun()

# --- 5.2. 단어 상세 정보 (right 컬럼) ---
# 요청에 따라 이 영역만 단어 정보를 표시합니다.
with right:
    st.subheader("단어 상세 정보")
    
    current_token = st.session_state.clicked_word
    
    if current_token:
        lemma = lemmatize_ru(current_token)
        info = st.session_state.word_info.get(lemma, {})

        if info and "ko_meanings" in info:
            st.markdown(f"### **{current_token}**")
            st.markdown(f"**기본형 (Lemma):** *{lemma}*")
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
                if ko_meanings and ko_meanings[0] == "JSON 파싱 오류":
                     st.error("Gemini API 정보 오류.")
                elif ko_meanings and ko_meanings[0].startswith(f"'{current_token}'의 API 키 없음"):
                     st.warning("API 키가 설정되지 않아 예문을 불러올 수 없습니다.")
                else:
                    st.info("예문 정보가 없습니다.")
        else:
            st.warning("단어 정보를 불러오는 중이거나 오류가 발생했습니다.")
            
    else:
        st.info("검색창에 단어를 입력하면 여기에 상세 정보가 표시됩니다.")

# ---------------------- 6. 하단: 누적 목록 + CSV ----------------------
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
                short = "; ".join(info["ko_meanings"][:2])
                rows.append({"기본형": lemma, "대표 뜻": short})
                processed_lemmas.add(lemma)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV로 저장", csv_bytes, "russian_words.csv", "text/csv")
    else:
        st.info("선택된 단어의 정보가 로드 중이거나, 표시할 정보가 없습니다.")
