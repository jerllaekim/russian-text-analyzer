import streamlit as st
import re
import os
import json
from pymystem3 import Mystem
from google import genai
import pandas as pd

# 초기 상태
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None
if "word_info" not in st.session_state:
    st.session_state.word_info = {}

st.set_page_config(layout="wide")
st.title("러시아어 텍스트 분석기")


# ----- CSS: span 클릭 스타일 -----
st.markdown("""
<style>
.word-span {
    font-size: 0.95rem;
    margin-right: 6px;
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
</style>
""", unsafe_allow_html=True)


# ----- 텍스트 입력 -----
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")

tokens = re.findall(r"\w+", text, flags=re.UNICODE)
unique_tokens = list(dict.fromkeys(tokens))  # 순서 유지


left, right = st.columns([2, 1])


# ----- 왼쪽: 단어 목록 -----
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")

    for tok in unique_tokens:

        # 이미 선택된 단어는 파란색 스타일
        css_class = "word-span"
        if tok in st.session_state.selected_words:
            css_class = "word-span word-selected"

        # span 클릭 → hidden button 클릭 유도
        html = f"""
        <span class="{css_class}" onclick="document.getElementById('btn_{tok}').click();">
            {tok}
        </span>
        """
        st.markdown(html, unsafe_allow_html=True)

        # 진짜 동작하는 것은 이 숨겨진 버튼
        if st.button("", key=f"btn_{tok}", help="", args=(tok,), kwargs=None):
            st.session_state.clicked_word = tok
            if tok not in st.session_state.selected_words:
                st.session_state.selected_words.append(tok)

    st.write("")

    if st.button("🔄 초기화"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.rerun()


# ----- 오른쪽: 단어 정보 -----
with right:
    st.subheader("📚 단어 정보")

    cw = st.session_state.clicked_word
    if cw:
        st.write(f"**선택된 단어:** {cw}")
    else:
        st.info("왼쪽 단어를 클릭하세요.")
