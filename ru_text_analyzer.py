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
    
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])
        elif lines:
             text = "\n".join(lines)
             
    try:
        data = json.loads(text)
        if 'examples' in data and len(data['examples']) > 2:
            data['examples'] = data['examples'][:2]
        return data
    except json.JSONDecodeError:
        st.error(f"Gemini 응답을 JSON으로 디코딩하는 데 실패했습니다: {text[:100]}...")
        return {"ko_meanings": ["응답 오류"], "examples": []}


# ---------------------- 2. 전역 스타일 및 숨겨진 폼 처리 ----------------------

# 숨겨진 버튼을 처리하기 위한 CSS
st.markdown("""
<style>
    /* 텍스트 입력 영역 아래의 띄어쓰기 제어 */
    div.stTextArea + div.stMarkdown > div {
        line-height: 2.0;
        font-size: 1.1em;
    }
    
    /* 폼 버튼 숨기기 */
    div.word-form > button {
        display: none !important;
    }

    /* 단어 스타일 (버튼이 아닌 HTML <span>으로 완벽히 인라인 처리) */
    .word-span {
        cursor: pointer;
        padding: 0px 0px;
        margin: 0px 0px;
        display: inline-block;
        transition: color 0.2s;
        user-select: none;
        white-space: pre; /* 띄어쓰기 보존 */
    }
    
    /* 파란색 글씨화 (선택된 단어) */
    .word-selected {
        color: #007bff !important; 
        font-weight: bold;
    }
    
    /* 구두점 스타일 (단어와 크기 맞추기) */
    .word-punctuation {
        padding: 0px 0px;
        margin: 0;
        display: inline-block;
        user-select: none;
        line-height: 1.5;
        font-size: 1em;
        white-space: pre;
    }
    
    /* 전체 텍스트를 감싸는 컨테이너 스타일 */
    .text-container {
        font-size: 1.25em;
        line-height: 2.0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- 3. 메인 로직 및 레이아웃 ----------------------

text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")
# 단어, 구두점, 공백을 모두 토큰으로 분리
tokens_with_punct = re.findall(r"(\w+|[^\s\w]+|\s+)", text, flags=re.UNICODE)

left, right = st.columns([2, 1])

# --- 3.1. 단어 목록 및 클릭 처리 (left 컬럼) ---
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")

    # 클릭된 단어를 숨겨진 st.form을 통해 처리하는 트릭
    with st.form(key='word_click_form', clear_on_submit=False):
        
        html_all = ['<div class="text-container">']
        
        for i, tok in enumerate(tokens_with_punct):
            if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
                # 단어인 경우: HTML <span>으로 렌더링하고, 클릭 시 폼 제출
                is_selected = tok in st.session_state.selected_words
                css = "word-span"
                if is_selected:
                    css += " word-selected"
                
                # HTML 버튼 역할을 하는 <span> 생성.
                # 클릭 시, 숨겨진 폼의 submit 버튼을 트리거하고 클릭된 단어를 hidden input에 담아 전달합니다.
                html_all.append(
                    f'<span class="{css}" onclick="document.getElementById(\'hidden_word\').value=\'{tok}\'; document.querySelector(\'[data-testid="stForm"] button[type="submit"]\').click();">'
                    f'{tok}'
                    f'</span>'
                )

            else:
                # 구두점 또는 공백인 경우: 일반 <span>으로 렌더링 (파란색화 방지)
                html_all.append(f'<span class="word-punctuation">{tok}</span>')

        html_all.append('</div>')
        
        st.markdown("".join(html_all), unsafe_allow_html=True)
        
        # 폼 제출 시 클릭된 단어를 저장할 숨겨진 Input
        clicked_word_input = st.text_input("Hidden Clicked Word", key='hidden_word', label_visibility="collapsed")
        
        # 숨겨진 Submit 버튼. 이 버튼이 눌리면 Python 코드가 재실행됨.
        submitted = st.form_submit_button("Submit Hidden Form", type="primary")

    # 폼 제출 후 로직 처리
    if submitted and clicked_word_input:
        st.session_state.clicked_word = clicked_word_input
        # 로드할 단어를 세션 상태에 추가
        if clicked_word_input not in st.session_state.selected_words:
            st.session_state.selected_words.append(clicked_word_input)
        # st.rerun()은 submit_button이 눌리면 자동으로 발생함.

    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 선택 초기화", key="reset_button"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
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
