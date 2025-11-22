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

# ---------------------- 2. 전역 스타일 정의 (버튼 UI 및 색상 강제) ----------------------

st.markdown("""
<style>
    /* 1. 단어 버튼 스타일: 버튼 모양 완전히 제거 및 색상 설정 */
    div.stButton > button {
        padding: 2px 4px !important;
        margin: 0 !important;
        border: none !important;
        background: none !important; /* 배경 제거 */
        box-shadow: none !important; /* 그림자 제거 */
        cursor: pointer;
        color: #333 !important; /* 기본 텍스트 색상 */
        font-weight: normal;
        height: auto !important;
        line-height: 1.5 !important;
        white-space: nowrap;
        text-align: left !important;
    }
    
    /* 2. 클릭된 단어 색상 유지 (선택된 단어는 파란색) */
    /* stButton의 상위 Div에 Word-selected 클래스를 강제로 적용하고, 내부 버튼 색상을 변경 */
    .word-selected > div > button {
        color: #007bff !important; 
        font-weight: bold !important;
    }
    
    /* 3. 구두점 스타일 (단어와 크기 맞추기) */
    .word-punctuation {
        padding: 2px 0px;
        margin: 2px 0;
        display: inline-block;
        user-select: none;
        line-height: 1.5;
        font-size: 1.1em; /* 단어 버튼과 비슷한 크기로 조정 */
    }
    
    /* 4. st.columns 컨테이너 내의 간격 조정 (가로 나열 시도) */
    div[data-testid^="stHorizontalBlock"] {
        flex-wrap: wrap !important; /* 단어가 많아지면 줄바꿈 허용 */
        gap: 0px 5px !important; /* 컬럼 간격 최소화 */
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

    # st.columns를 사용하여 단어와 구두점을 가로로 나열
    # Streamlit에서 인라인 요소를 강제하는 가장 안정적인 방법 중 하나입니다.
    
    cols = st.columns(len(tokens_with_punct)) # 토큰 개수만큼 컬럼 생성

    for i, tok in enumerate(tokens_with_punct):
        with cols[i]:
            if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
                # 단어인 경우: st.button 사용
                is_selected = tok in st.session_state.selected_words
                
                # CSS 클래스를 st.button의 상위 컨테이너에 적용하기 위한 트릭
                # Streamlit 위젯은 자체 Div에 래핑되므로, 이 래퍼에 클래스를 삽입합니다.
                
                # 주의: st.button은 텍스트를 인수로 받으므로, 이 텍스트로 버튼을 만듭니다.
                
                button_html = f'<div class="{"word-selected" if is_selected else ""}"></div>'
                # st.markdown(button_html, unsafe_allow_html=True) # HTML 삽입

                # st.button을 렌더링
                st.button(
                    tok, 
                    key=f"word_{tok}_{i}", # 고유 key
                    on_click=on_word_click,
                    args=(tok,)
                )
                
                # CSS 클래스를 버튼 컨테이너에 동적으로 적용하는 Javascript 트릭이 필요하지만,
                # Streamlit 클라우드 환경에서 JS 삽입은 불안정합니다. 
                # 여기서는 버튼을 출력한 후, CSS가 버튼 내부의 색상을 변경하도록 의존합니다.

            else:
                # 구두점인 경우: st.markdown으로 출력
                st.markdown(f'<span class="word-punctuation">{tok}</span>')


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
