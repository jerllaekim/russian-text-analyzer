import os
import re
import json

import pandas as pd
import streamlit as st
from pymystem3 import Mystem
from google import genai


# ─────────────────────────────
# 기본 설정 + 전역 상태
# ─────────────────────────────
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None          # 현재 상세보기 단어(표면형)
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []          # 선택된 단어(표면형)
if "word_info" not in st.session_state:
    st.session_state.word_info = {}              # lemma -> {lemma, ko_meanings}


# ─────────────────────────────
# CSS: 버튼을 텍스트처럼 보이게 + 가로 나열
# ─────────────────────────────
st.markdown(
    """
<style>
/* 모든 버튼 컨테이너를 인라인으로 → 가로로 나열 */
div[data-testid="stButton"] {
    display: inline-block;
    margin: 0 4px 4px 0;
}

/* 버튼 자체를 "텍스트"처럼 보이게 */
div[data-testid="stButton"] > button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    cursor: pointer !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────
# 형태소 분석기
# ─────────────────────────────
mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    lemmas = mystem.lemmatize(word)
    return (lemmas[0] if lemmas else word).strip()


# ─────────────────────────────
# Gemini 설정
# ─────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit Secrets에 GEMINI_API_KEY를 넣어주세요.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
너는 러시아어-한국어 학습을 돕는 도우미이다.
러시아어 단어에 대해 간단한 한국어 뜻과 예문을 제공한다.
반드시 유효한 JSON만 출력해야 한다.
"""

def build_prompt(word: str, lemma: str) -> str:
    return f"""
{SYSTEM_INSTRUCTION}

러시아어 단어: {word}
기본형(lemma): {lemma}

다음 형식의 JSON만 출력해라:

{{
  "ko_meanings": ["뜻1", "뜻2"],
  "examples": [
    {{
      "ru": "러시아어 예문1 (단어 또는 lemma 포함)",
      "ko": "예문1의 한국어 번역"
    }},
    {{
      "ru": "러시아어 예문2 (단어 또는 lemma 포함)",
      "ko": "예문2의 한국어 번역"
    }}
  ]
}}
"""

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word: str, lemma: str):
    prompt = build_prompt(word, lemma)
    res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = res.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])
    return json.loads(text)


# ─────────────────────────────
# 텍스트 입력
# ─────────────────────────────
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")

# 원문 그대로 보여주기 (클릭 X)
st.subheader("원문 텍스트")
st.write(text)

# 단어만 추출 (소문자/대문자 포함, 구두점 제외)
tokens = re.findall(r"\w+", text, flags=re.UNICODE)
unique_tokens = sorted(set(tokens), key=lambda x: tokens.index(x))  # 등장 순서 유지


left, right = st.columns([2, 1], gap="large")


# ─────────────────────────────
# 왼쪽: 단어 목록 (텍스트처럼, 클릭하면 파란색 유지)
# ─────────────────────────────
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")
    st.caption("단어를 클릭하면 파란색으로 바뀌고, 오른쪽에 정보가 표시되며, 하단에 누적됩니다.")

    if not unique_tokens:
        st.info("텍스트에서 단어를 찾지 못했습니다.")
    else:
        for idx, tok in enumerate(unique_tokens):
            # 한 번이라도 클릭된 단어는 :blue[...] 로 표시
            if tok in st.session_state.selected_words:
                label = f":blue[{tok}]"
            else:
                label = tok

            if st.button(label, key=f"word_{idx}_{tok}"):
                st.session_state.clicked_word = tok
                if tok not in st.session_state.selected_words:
                    st.session_state.selected_words.append(tok)
        st.write("")  # 줄바꿈용

    with st.expander("초기화"):
        if st.button("🔄 선택 & 누적 데이터 초기화", key="reset_all"):
            st.session_state.clicked_word = None
