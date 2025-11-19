import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai

# ---------------------- 초기 설정 ----------------------
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None
if "word_info" not in st.session_state:
    st.session_state.word_info = {}

mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    lemmas = mystem.lemmatize(word)
    return (lemmas[0] if lemmas else word).strip()

# ---------------------- Gemini ----------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY가 설정되지 않음.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """
너는 러시아어-한국어 학습을 돕는 도우미이다.
러시아어 단어에 대해 간단한 한국어 뜻과 예문을 제공한다.
반드시 JSON만 출력한다.
"""

def make_prompt(word, lemma):
    return f"""
{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}

{{
  "ko_meanings": ["뜻1", "뜻2"],
  "examples": [
    {{"ru": "예문1", "ko": "번역1"}},
    {{"ru": "예문2", "ko": "번역2"}}
  ]
}}
"""

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word, lemma):
    prompt = make_prompt(word, lemma)
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = res.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])
    return json.loads(text)

# ---------------------- CSS ----------------------
st.markdown("""
<style>
.word-span {
    font-size: 0.95rem;
    margin-right: 8px;
    cursor: pointer;
    color: #333;
}
.word-span:hover {
    text-decoration: underline;
}
.word-selected {
    color: #1E88E5 !important;
    text-decoration: underline !important;
}

/* 숨겨진 버튼 */
.hidden-btn > button {
    background: none !important;
    border: none !important;
    width: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- 텍스트 입력 ----------------------
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")
tokens = list(dict.fromkeys(re.findall(r"\w+", text, flags=re.UNICODE)))

left, right = st.columns([2, 1])

# ---------------------- 왼쪽: 단어 목록 ----------------------
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")

    for tok in tokens:
        css = "word-span"
        if tok in st.session_state.selected_words:
            css = "word-span word-selected"

        st.markdown(
            f"""
            <span class="{css}" onclick="document.getElementById('btn_{tok}').click();">
                {tok}
            </span>
            """,
            unsafe_allow_html=True
        )

        # 숨겨진 버튼이 실제로 상태 변화시킴
        if st.button(" ", key=f"btn_{tok}", help="", args=None, kwargs=None):
            st.session_state.clicked_word = tok
            if tok not in st.session_state.selected_words:
                st.session_state.selected_words.append(tok)
            st.rerun()

    if st.button("🔄 초기화"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.rerun()

# ---------------------- 오른쪽: 단어 정보 ----------------------
with right:
    st.subheader("📚 단어 정보")

    cw = st.session_state.clicked_word

    if cw:
        lemma = lemmatize_ru(cw)
        st.write(f"**선택된 단어:** {cw}")
        st.write(f"**기본형(lemma):** *{lemma}*")

        try:
            info = fetch_from_gemini(cw, lemma)
        except Exception as e:
            st.error(f"Gemini 오류: {e}")
            info = {}

        ko_meanings = info.get("ko_meanings", [])
        examples = info.get("examples", [])

        if ko_meanings:
            st.session_state.word_info[lemma] = {
                "lemma": lemma,
                "ko_meanings": ko_meanings
            }

            st.markdown("**한국어 뜻:**")
            for m in ko_meanings:
                st.markdown(f"- {m}")

        if examples:
            st.markdown("### 📖 예문")
            for ex in examples:
                st.markdown(f"- **{ex.get('ru','')}**")
                st.markdown(f" → {ex.get('ko','')}")

        # 외부 링크
        mt = f"https://www.multitran.com/m.exe?l1=2&l2=5&s={lemma}"
        rnc = f"https://ruscorpora.ru/search?search={lemma}"
        st.markdown(f"[Multitran에서 검색]({mt})  \n[러시아 국립 코퍼스]({rnc})")

    else:
        st.info("왼쪽 단어를 클릭하세요.")

# ---------------------- 하단: 누적 목록 + CSV ----------------------
st.divider()
st.subheader("📝 선택한 단어 모음")

selected = st.session_state.selected_words
word_info = st.session_state.word_info

# ---- lemma / 뜻 표 ----
if word_info:
    rows = []
    for lemma, info in word_info.items():
        short = "; ".join(info["ko_meanings"][:2])
        rows.append({"lemma": lemma, "뜻": short})

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("💾 CSV로 저장", csv_bytes, "words.csv", "text/csv")

# ---------------------- 직접 단어 검색 ----------------------
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
