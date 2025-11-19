import streamlit as st
import re

# 상태값
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None

st.markdown("""
<style>
.word-btn {
    border: none !important;
    background: none !important;
    padding: 0 !important;
    margin-right: 8px !important;
    font-size: 0.95rem !important;
    color: #333 !important;
    cursor: pointer !important;
}
.word-btn:hover {
    text-decoration: underline;
}
.word-selected {
    color: #1E88E5 !important;
    text-decoration: underline !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- 단어 표시 ----------------
def word_button(word: str):
    selected = word in st.session_state.selected_words
    css = "word-btn word-selected" if selected else "word-btn"

    # HTML 텍스트처럼 보이는 버튼 생성
    clicked = st.button(
        f"<span class='{css}'>{word}</span>",
        key=f"w_{word}",
        help="",
        use_container_width=False
    )

    # 클릭 시 상태 업데이트
    if clicked:
        if word not in st.session_state.selected_words:
            st.session_state.selected_words.append(word)
        st.session_state.clicked_word = word


# ---------------- 렌더링 ----------------
st.subheader("단어 목록 (텍스트에서 추출)")

text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")
tokens = re.findall(r"\w+", text)
tokens = list(dict.fromkeys(tokens))

for w in tokens:
    word_button(w)
