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

# Mystem 인스턴스
mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    """단어의 기본형(lemma)을 추출합니다."""
    # 단어만 처리 (구두점/공백 제외)
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        lemmas = mystem.lemmatize(word)
        return (lemmas[0] if lemmas else word).strip()
    return word

# ---------------------- 1. Gemini 연동 함수 ----------------------

# Streamlit secrets에서 API 키 로드 (os.getenv도 폴백으로 사용)
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = "너는 러시아어-한국어 학습을 돕는 도우미이다. 러시아어 단어에 대해 간단한 한국어 뜻과 예문을 제공한다. 반드시 JSON만 출력한다."
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
    
    # 모델 호출
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = res.text.strip()
    
    # JSON 파싱을 위해 Markdown 코드 블록 제거
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])
        elif lines:
             text = "\n".join(lines)
             
    return json.loads(text)

# ---------------------- 2. 전역 스타일 및 JavaScript 정의 ----------------------

st.markdown("""
<style>
    /* 단어 스타일 정의 */
    .word-span, .word-selected {
        cursor: pointer;
        padding: 2px 4px;
        margin: 2px 0;
        display: inline-block;
        transition: color 0.2s;
        user-select: none;
        border: none !important;
        text-decoration: none !important; 
        background-color: transparent !important; 
    }
    .word-span:hover {
        color: #007bff;
    }
    .word-selected {
        color: #007bff; 
        font-weight: bold;
    }
    .word-punctuation {
        padding: 2px 0px;
        margin: 2px 0;
        display: inline-block;
        user-select: none;
    }
</style>
""", unsafe_allow_html=True)

# 쿼리 파라미터 업데이트 JavaScript 주입
# 클릭된 단어를 URL의 'word' 파라미터로 설정하고 **페이지를 새로고침**하여 Streamlit 재실행 유도
st.markdown("""
<script>
    function setQueryParam(word) {
        const url = new URL(window.location.href);
        // 'word' 파라미터 설정
        url.searchParams.set('word', word);
        // URL 업데이트 후, 페이지를 새로고침하여 Streamlit의 Python 코드를 재실행합니다.
        window.location.href = url.toString();
    }
</script>
""", unsafe_allow_html=True)

# ---------------------- 3. 메인 로직 및 레이아웃 ----------------------

text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")
# 단어와 구두점을 모두 토큰으로 분리
tokens_with_punct = re.findall(r"(\w+|[^\s\w]+)", text, flags=re.UNICODE)

left, right = st.columns([2, 1])

# --- 3.1. 단어 목록 (left 컬럼) ---
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")

    html_all = ""
    for tok in tokens_with_punct:
        css = "word-span"
        if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
            # 단어인 경우
            if tok in st.session_state.selected_words:
                css = "word-selected"
            
            # HTML 코드를 한 줄로 포맷팅하여 안전하게 렌더링
            html_all += (
                f'<span class="{css}" onclick="setQueryParam(\'{tok}\');">'
                f'{tok}'
                f'</span> ' # 단어 뒤에 공백 추가 (띄어쓰기)
            )
        else:
            # 구두점/공백인 경우
            html_all += (
                f'<span class="word-punctuation">'
                f'{tok}'
                f'</span>'
            )
            # 공백 토큰을 따로 처리하지 않았다면, 구두점 뒤에 공백이 필요할 경우 여기서 추가해야 함.
            # (현재 정규식은 공백을 분리하지 않으므로, 원래 텍스트의 공백이 자연스럽게 포함됨)

    # 전체를 Div로 묶어 HTML 렌더링을 확실하게 합니다.
    st.markdown(f'<div style="line-height: 2.0; font-size: 1.25em;">{html_all}</div>', unsafe_allow_html=True) 
    
    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 선택 초기화"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        # 쿼리 파라미터도 완전히 초기화
        st.experimental_set_query_params() 
        st.rerun()

# --- 3.2. 쿼리 파라미터에서 클릭된 단어 읽기 및 정보 로드 ---

query_params = st.experimental_get_query_params()
clicked_word_from_url = query_params.get("word", [None])[0]

# URL에서 읽은 단어가 있고, 이전에 클릭한 단어와 다를 때만 로직 실행 (무한 루프 방지)
if clicked_word_from_url and clicked_word_from_url != st.session_state.clicked_word:
    st.session_state.clicked_word = clicked_word_from_url
    tok = clicked_word_from_url
    
    # 단어 정보 로드 로직
    if tok not in st.session_state.selected_words:
        st.session_state.selected_words.append(tok)
    
    lemma = lemmatize_ru(tok)
    
    # 현재 토큰에 대한 정보가 없거나, 다른 표제형의 정보가 로드된 경우에만 새로 로드
    if lemma not in st.session_state.word_info or st.session_state.word_info.get(lemma, {}).get('loaded_token') != tok:
        with st.spinner(f"'{tok}'의 정보를 불러오는 중... (Gemini API 호출)"):
            try:
                info = fetch_from_gemini(tok, lemma)
                st.session_state.word_info[lemma] = {**info, "loaded_token": tok} 
            except Exception as e:
                st.error(f"단어 정보 로드 오류: {e}")


# --- 3.3. 단어 상세 정보 (right 컬럼) ---
with right:
    st.subheader("단어 상세 정보")
    
    current_token = st.session_state.clicked_word
    
    if current_token:
        lemma = lemmatize_ru(current_token)
        info = st.session_state.word_info.get(lemma, {})

        if info:
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
                if ko_meanings and ko_meanings[0].startswith(f"'{current_token}'의 API 키 없음"):
                     st.warning("API 키가 설정되지 않아 예문을 불러올 수 없습니다.")
                else:
                    st.info("예문 정보가 없습니다.")
        else:
            st.warning("단어 정보를 불러오는 중이거나 오류가 발생했습니다.")
            
    else:
        st.info("왼쪽 단어 목록에서 단어를 클릭해주세요.")

# ---------------------- 4. 하단: 누적 목록 + CSV ----------------------
st.divider()
st.subheader("📝 선택한 단어 모음 (기본형 기준)")

selected = st.session_state.selected_words
word_info = st.session_state.word_info

if word_info:
    rows = []
    processed_lemmas = set()
    
    # 선택된 단어들을 순회하며 기본형을 기준으로 중복 없이 정보를 정리
    for tok in selected:
        lemma = lemmatize_ru(tok)
        if lemma not in processed_lemmas and lemma in word_info:
            info = word_info[lemma]
            # 대표 뜻은 최대 2개만 추출
            short = "; ".join(info["ko_meanings"][:2])
            rows.append({"기본형": lemma, "대표 뜻": short})
            processed_lemmas.add(lemma)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV로 저장", csv_bytes, "russian_words.csv", "text/csv")
    else:
        st.info("선택된 단어의 정보를 로드 중이거나, 표시할 정보가 없습니다.")


# ---------------------- 5. 직접 단어 검색 ----------------------
st.divider()
st.subheader("🔍 직접 단어 검색")

manual = st.text_input("단어 직접 입력", "")

if manual:
    lemma = lemmatize_ru(manual)
    st.markdown(f"**입력 단어:** **{manual}**")
    st.markdown(f"**기본형(lemma):** *{lemma}*")

    try:
        # 수동 검색은 캐시된 정보를 사용
        info = fetch_from_gemini(manual, lemma)
    except Exception as e:
        st.error(f"Gemini 오류: {e}")
        info = {}

    ko_meanings = info.get("ko_meanings", [])
    examples = info.get("examples", [])

    if ko_meanings:
        st.markdown("#### 한국어 뜻")
        for m in ko_meanings:
            st.markdown(f"- **{m}**")

    if examples:
        st.markdown("#### 📖 예문")
        for ex in examples:
            st.markdown(f"- **{ex.get('ru','')}**")
            st.markdown(f" → {ex.get('ko','')}")
