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
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = res.text.strip()
    
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])
        elif lines:
             text = "\n".join(lines)
             
    return json.loads(text)

# ---------------------- 2. 전역 스타일 정의 (버튼 위젯 인라인 강제) ----------------------

st.markdown("""
<style>
    /* 텍스트 영역 아래의 띄어쓰기 제어 */
    div.stTextArea + div.stMarkdown > div {
        line-height: 2.0;
        font-size: 1.25em;
    }

    /* 모든 st.button 컨테이너를 인라인 블록으로 강제하여 가로 나열 시도 */
    /* Streamlit의 내부 클래스(st-emotion-cache-123456 등)는 자주 바뀌지만,
       stButton 클래스와 그 내부 요소에 스타일을 적용하는 것이 최선입니다. */
    div[data-testid="stForm"] + div.stButton, 
    div.stButton {
        display: inline-flex !important; /* 가로 나열 */
        margin: 0px 0px 0px 0px !important; /* 마진 제거 */
    }

    /* 버튼 자체 스타일: 버튼 모양 완전히 제거 */
    div.stButton > button {
        padding: 2px 4px !important;
        margin: 0 !important;
        border: none !important;
        background: none !important;
        box-shadow: none !important;
        cursor: pointer;
        color: #333 !important; /* 기본 텍스트 색상 */
        font-weight: normal;
        height: auto !important;
        line-height: 1.5 !important;
        white-space: nowrap; /* 단어가 줄바꿈되지 않도록 */
    }
    
    /* 선택된(파란색) 단어 스타일 */
    .word-selected > button {
        color: #007bff !important; 
        font-weight: bold !important;
    }
    
    /* 구두점 스타일 (버튼 아님) */
    .word-punctuation {
        padding: 2px 0px;
        margin: 2px 0;
        display: inline-block;
        user-select: none;
        line-height: 1.5;
        font-size: 1.25em; /* 단어와 크기 맞추기 */
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- 3. 메인 로직 및 레이아웃 ----------------------

text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")
tokens_with_punct = re.findall(r"(\w+|[^\s\w]+)", text, flags=re.UNICODE)

left, right = st.columns([2, 1])

# --- 3.1. 단어 목록 (left 컬럼) ---
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")

    # 단어 버튼 클릭 시 실행될 콜백 함수
    def on_word_click(clicked_token):
        st.session_state.clicked_word = clicked_token
        if clicked_token not in st.session_state.selected_words:
            st.session_state.selected_words.append(clicked_token)

    # st.markdown을 사용하여 단어와 구두점을 같은 줄에 배치
    html_elements = [] 
    
    for i, tok in enumerate(tokens_with_punct):
        if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
            # 단어인 경우: HTML로 버튼 컨테이너 시작
            is_selected = tok in st.session_state.selected_words
            css_class = "word-container"
            if is_selected:
                 css_class += " word-selected"

            # 1. HTML 마크업 시작 (CSS 적용을 위한 래퍼)
            html_elements.append(f'<div class="{css_class}" style="display: inline-flex;">')
            
            # 2. 버튼 배치 및 클릭 로직 실행 (st.button은 Python 코드를 재실행시키는 핵심 위젯)
            st.button(
                tok, 
                key=f"word_{tok}_{i}", # 고유 key
                on_click=on_word_click,
                args=(tok,)
            )
                
            # 3. HTML 마크업 종료 및 띄어쓰기 추가 (다음 요소와 분리)
            html_elements.append(f'</div> ') 

        else:
            # 구두점인 경우: 마크다운으로 출력하여 단어 사이에 배치
            html_elements.append(f'<span class="word-punctuation">{tok}</span>')

    # Streamlit은 위젯(st.button)과 마크다운(st.markdown)이 섞여 있을 때 레이아웃 제어가 복잡합니다.
    # 위 코드에서 st.button이 이미 순서대로 배치되었기 때문에, 
    # 나머지 텍스트(구두점)만 마크다운으로 출력하는 방식이 가장 안정적입니다.
    # 단어 버튼은 위에 배치되었고, 구두점은 html_elements에 모였으므로,
    # 이를 다시 출력하여 버튼 사이에 구두점을 배치합니다.
    # 주의: st.button이 이미 출력되었으므로, 이 코드는 HTML 래핑 역할만 수행해야 합니다.
    st.markdown("".join(html_elements), unsafe_allow_html=True) 

    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 선택 초기화", key="reset_button"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.experimental_set_query_params() 
        st.rerun()

# --- 3.2. 단어 상세 정보 로드 ---

current_token = st.session_state.clicked_word

if current_token:
    tok = current_token
    lemma = lemmatize_ru(tok)
    
    # 정보 로드 로직
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
    
    for tok in selected:
        lemma = lemmatize_ru(tok)
        if lemma not in processed_lemmas and lemma in word_info:
            info = word_info[lemma]
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
