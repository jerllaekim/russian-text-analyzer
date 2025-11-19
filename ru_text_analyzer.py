import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai

# ---------------------- 0. 초기 설정 및 세션 상태 ----------------------
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

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
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    client = None
else:
    client = genai.Client(api_key=api_key)

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

# ---------------------- 2. 전역 스타일 정의 ----------------------

st.markdown("""
<style>
    /* 단어 스타일 정의 (동일) */
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

# ---------------------- 3. 메인 로직 및 레이아웃 ----------------------

text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")
tokens_with_punct = re.findall(r"(\w+|[^\s\w]+)", text, flags=re.UNICODE)
clickable_words = list(dict.fromkeys([t for t in tokens_with_punct if re.fullmatch(r'\w+', t, flags=re.UNICODE)]))

left, right = st.columns([2, 1])

# ----------------------------------------
# 3.1. 단어 목록 (left 컬럼)
# ----------------------------------------
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")

    html_all = ""
    for tok in tokens_with_punct:
        if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
            css = "word-span"
            if tok in st.session_state.selected_words:
                css = "word-selected"
            
            # ❗ onclick: 쿼리 파라미터를 변경하는 JavaScript 함수 호출 (버튼 제거)
            html_all += f"""
            <span class="{css}" onclick="setQueryParam('{tok}');">
                {tok}
            </span>
            """
        else:
            html_all += f"""
            <span class="word-punctuation">
                {tok}
            </span>
            """
    
    st.markdown(html_all, unsafe_allow_html=True)
    
    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 초기화"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.experimental_set_query_params(word=None) # 쿼리 파라미터 초기화
        st.rerun()

# ----------------------------------------
# 3.2. 쿼리 파라미터 업데이트 JavaScript 주입
# ----------------------------------------

# 이 JavaScript 코드가 클릭된 단어를 URL의 'word' 파라미터로 설정하여 Streamlit을 재실행합니다.
st.markdown("""
<script>
    function setQueryParam(word) {
        const url = new URL(window.location.href);
        // 'word' 파라미터 설정
        url.searchParams.set('word', word);
        // URL을 업데이트하고, Streamlit이 이를 감지하여 재실행되도록 합니다.
        window.history.pushState(null, '', url.toString());
        
        // ❗ Streamlit이 즉시 재실행되도록 강제하는 함수 호출 (이 함수는 Streamlit의 내부 JS에 포함되어 있음)
        if (window.streamlit) {
            window.streamlit.set
        } else {
             // 페이지를 새로고침하여 Streamlit 재실행 유도 (덜 부드러운 방식)
             window.location.reload();
        }
    }
    // ❗ 숨겨진 버튼 코드는 완전히 제거되었습니다.
</script>
""", unsafe_allow_html=True)


# ----------------------------------------
# 3.3. 쿼리 파라미터에서 클릭된 단어 읽기 (로직 업데이트)
# ----------------------------------------

query_params = st.experimental_get_query_params()
clicked_word_from_url = query_params.get("word", [None])[0]

# ❗ URL에서 읽은 단어가 있으면 세션 상태를 업데이트하고 정보 로드
if clicked_word_from_url and clicked_word_from_url != st.session_state.clicked_word:
    st.session_state.clicked_word = clicked_word_from_url
    tok = clicked_word_from_url
    
    # 단어 정보 로드 로직 (이전과 동일)
    if tok not in st.session_state.selected_words:
        st.session_state.selected_words.append(tok)
    
    lemma = lemmatize_ru(tok)
    if lemma not in st.session_state.word_info or st.session_state.word_info.get(lemma, {}).get('loaded_token') != tok:
        with st.spinner(f"'{tok}'의 정보를 불러오는 중..."):
            try:
                info = fetch_from_gemini(tok, lemma)
                st.session_state.word_info[lemma] = {**info, "loaded_token": tok} 
            except Exception as e:
                st.error(f"단어 정보 로드 오류: {e}")
    # st.rerun()은 st.experimental_set_query_params나 window.location.reload()에 의해 발생하므로 생략


# ----------------------------------------
# 3.4. 단어 상세 정보 (right 컬럼)
# ----------------------------------------
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
            st.warning("단어 정보를 불러오는 중이거나 오류가 발생했습니다.")
            
    else:
        st.info("왼쪽 단어 목록에서 단어를 클릭해주세요.")

# ---------------------- 4. 하단: 누적 목록 + CSV ----------------------
st.divider()
st.subheader("📝 선택한 단어 모음")

selected = st.session_state.selected_words
word_info = st.session_state.word_info

# (누적 목록 및 CSV 로직 생략 - 동일)

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
        st.download_button("💾 CSV로 저장", csv_bytes, "words.csv", "text/csv")
    else:
        st.info("선택된 단어의 정보를 로드 중이거나, 표시할 정보가 없습니다.")


# ---------------------- 5. 직접 단어 검색 ----------------------
st.divider()
st.subheader("🔍 직접 단어 검색")

manual = st.text_input("단어 직접 입력", "")

if manual:
    lemma = lemmatize_ru(manual)
    st.markdown(f"**입력 단어:** {manual}")
    st.markdown(f"**기본형(lemma):** *{lemma}*")

    try:
        info = fetch_from_gemini(manual, lemma)
    except Exception as e:
        st.error(f"Gemini 오류: {e}")
        info = {}

    ko_meanings = info.get("ko_meanings", [])
    examples = info.get("examples", [])

    if ko_meanings:
        st.markdown("**한국어 뜻:**")
        for m in ko_meanings:
            st.markdown(f"- {m}")

    if examples:
        st.markdown("### 📖 예문")
        for ex in examples:
            st.markdown(f"- **{ex.get('ru','')}**")
            st.markdown(f" → {ex.get('ko','')}")
