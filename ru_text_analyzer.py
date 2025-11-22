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

# ---------------------- 2. 전역 스타일 정의 (버튼 위젯 덮어쓰기) ----------------------

# Streamlit 버튼 위젯을 단어 스타일로 덮어씌웁니다.
st.markdown("""
<style>
    /* 텍스트 입력 영역 아래의 띄어쓰기 제어 */
    div.stTextArea + div.stMarkdown > div {
        line-height: 2.0;
        font-size: 1.25em;
    }

    /* 단어처럼 보이도록 버튼 스타일을 변경 */
    .word-container {
        display: inline-block;
        margin: 2px 0;
        user-select: none;
    }
    .word-button {
        padding: 2px 4px !important;
        margin: 0 !important;
        border: none !important;
        background: none !important;
        box-shadow: none !important;
        cursor: pointer;
        color: #333; /* 기본 텍스트 색상 */
        font-weight: normal;
        display: inline-block !important;
        line-height: 1.5; /* 줄 간격 유지 */
    }
    /* 클릭된/선택된 단어 스타일 */
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
    }
    
    /* Streamlit 버튼의 기본 간격을 없애 단어처럼 붙도록 처리 */
    .stButton > button {
        border-radius: 0px !important;
        padding: 2px 4px !important;
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

    # 단어를 저장할 임시 컨테이너
    word_elements = [] 
    
    # 텍스트 내의 모든 토큰을 순회하며 위젯 또는 구두점 삽입
    for i, tok in enumerate(tokens_with_punct):
        if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
            # 단어인 경우: 실제 st.button을 사용하여 클릭을 감지
            
            is_selected = tok in st.session_state.selected_words
            
            # CSS 클래스를 지정하기 위한 HTML 마크업 시작
            css_class = "word-container"
            if is_selected:
                 css_class += " word-selected"

            # 1. HTML 마크업 시작 (단어 컨테이너)
            word_elements.append(f'<div class="{css_class}">')
            
            # 2. 버튼 배치 (클릭 로직)
            # 버튼을 먼저 배치하고, 클릭되면 처리 함수를 호출합니다.
            
            # 콜백 함수: 버튼이 클릭될 때만 실행되며, 세션 상태를 업데이트합니다.
            def on_word_click(clicked_token):
                st.session_state.clicked_word = clicked_token
                # 단어 정보 로드 로직은 아래 3.2에서 재실행 시 처리됨
                if clicked_token not in st.session_state.selected_words:
                    st.session_state.selected_words.append(clicked_token)

            # st.button을 렌더링하고, 클릭 여부를 즉시 확인
            if st.button(
                tok, 
                key=f"word_{tok}_{i}", # 고유 key를 지정해야 모든 버튼이 작동
                help=f"클릭하여 '{tok}' 정보 보기",
                on_click=on_word_click,
                args=(tok,)
            ):
                # 버튼 클릭 시 on_click이 실행되고 Streamlit이 재실행됨
                pass 
                
            # 3. HTML 마크업 종료 및 띄어쓰기 추가
            word_elements.append(f'</div> ') # 띄어쓰기를 위해 div 밖에서 공백 추가

        else:
            # 구두점인 경우: 마크다운으로 출력 (클릭 불가)
            word_elements.append(f'<span class="word-punctuation">{tok}</span>')

    # st.markdown을 사용하여 구두점과 HTML 마크업을 함께 렌더링
    # st.markdown(word_elements[0], unsafe_allow_html=True) # 각 요소를 개별적으로 렌더링할 필요는 없음

    # Streamlit은 버튼과 마크다운을 섞어 렌더링할 때 약간의 트릭이 필요합니다.
    # 여기서는 Streamlit의 자동 렌더링을 믿고, 버튼 사이에 띄어쓰기를 위해 마크다운을 활용합니다.
    # 그러나 버튼 위젯과 마크다운을 섞을 때 레이아웃이 깨지기 쉬우므로,
    # 위에서 이미 st.button을 배치했으므로, 텍스트와 구두점을 버튼 사이에 넣어주는 방식으로 재구성합니다.
    
    # *******************************************************************
    # 🚨 주의: Streamlit은 위젯과 HTML을 섞을 때 문제가 발생하므로, 
    # 위 코드에서 st.button이 이미 순서대로 배치되었을 경우, 
    # 나머지 텍스트(구두점)만 마크다운으로 출력하는 방식이 더 안정적입니다.
    # *******************************************************************

    # 하지만 최종 사용자가 보는 화면을 위해, 현재는 st.button을 배치하는 것만으로 충분합니다.
    # st.button은 블록 레벨 요소처럼 동작하므로, CSS를 사용하여 인라인 블록으로 만들어야 합니다.
    # CSS 설정 (.word-container, .word-button)이 이 문제를 해결해 주길 기대합니다.


    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 선택 초기화"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.experimental_set_query_params() 
        st.rerun()

# --- 3.2. 단어 상세 정보 로드 (클릭 시 실행) ---

current_token = st.session_state.clicked_word

if current_token:
    tok = current_token
    lemma = lemmatize_ru(tok)
    
    # 단어 정보 로드 (세션 상태에 없거나 로드된 토큰이 다를 경우)
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
