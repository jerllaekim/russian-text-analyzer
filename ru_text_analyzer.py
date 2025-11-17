import os
import re
import json

import pandas as pd
import streamlit as st
from pymystem3 import Mystem
from google import genai  # google-genai 패키지


# ─────────────────────────────
# 기본 설정 + 전역 상태
# ─────────────────────────────
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None          # 현재 상세보기 중인 단어(표면형)
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []          # 사용자가 선택한 단어(표면형) 리스트
if "word_info" not in st.session_state:
    # lemma 기준으로 뜻을 누적 저장
    # 예: {"человек": {"lemma": "человек", "ko_meanings": ["사람", "인간"]}, ...}
    st.session_state.word_info = {}


# ─────────────────────────────
# CSS: 단어 버튼은 텍스트처럼, 칩/다른 버튼은 별도
# ─────────────────────────────
st.markdown(
    """
<style>
/* ✅ 단어용 버튼: 텍스트처럼 보이게 */
div.word-btn-normal > button,
div.word-btn-selected > button {
    border: none;
    background: transparent;
    padding: 0 2px 2px 0;
    margin: 0;
    min-width: 0;
    font-size: 1rem;
}

/* 처음 상태: 검은 글씨 */
div.word-btn-normal > button {
    color: #000000;
}
div.word-btn-normal > button:hover {
    text-decoration: underline;
}

/* 선택된 단어: 파란색 + 조금 두껍게 */
div.word-btn-selected > button {
    color: #1E88E5;
    font-weight: 600;
}
div.word-btn-selected > button:hover {
    text-decoration: underline;
}

/* 🔹 선택 단어 칩 */
div.selected-word-chip > button {
    border-radius: 999px;
    padding: 2px 10px;
    margin: 3px;
    border: 1px solid #1E88E5;
    background-color: rgba(30, 136, 229, 0.06);
    color: #1E88E5;
}

/* 🔹 현재 선택된 단어 칩(✅) */
div.selected-word-chip-active > button {
    border-radius: 999px;
    padding: 2px 10px;
    margin: 3px;
    border: 1px solid #1E88E5;
    background-color: rgba(30, 136, 229, 0.18);
    color: #1E88E5;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────
# 형태소 분석기 (lemma)
# ─────────────────────────────
mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    lemmas = mystem.lemmatize(word)
    return (lemmas[0] if lemmas else word).strip()


# ─────────────────────────────
# Gemini API 설정
# ─────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit Secrets에 GEMINI_API_KEY를 넣어주세요.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
너는 러시아어-한국어 학습을
