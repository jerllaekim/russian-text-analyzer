import streamlit as st
import re

# 상태 초기화
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None

st.set_page_config(layout="wide")
st.title("러시아어 텍스트 분석기")

# ------------------ CSS: 버튼을 텍스트로 보이게 ------------------
st.markdown("""
<style>
/* 버튼 컨테이너를 inline-block으로 만들어 가로로 나열 */
div[data-testid="stButton"] {
    display: inline-block;
    margin-right: 6px;
}

/* 버튼 자체를 투명하게 → 텍스트처럼 보임 */
div[data-testid="stButton"] > button {
    border: none !important;
    background: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    font-size: 0.95rem !important;
    color: #333 !important;
}

/* 파란색 텍스트 */
.word-selected {
    color: #1E88E5 !important;
    text-decoration: underline !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------ 텍스트 입력 ------------------
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")
tokens = re.findall(r"\w+", text)
tokens = list(dict.fromkeys(tokens))  # 중복 제거 + 순서 유지

st.subheader("단어 목록 (텍스트에서 추출)")

# ------------------ 단어 목록 버튼 ------------------
for tok in tokens:
    # 이미 선택된 단어는 파란색 텍스트로 렌더링
    label = f'<span class="word-selected">{tok}</span>' if tok in st.session_state.selected_words else tok

    if st.button(label, key=f"w_{tok}"):
        st.session_state.clicked_word = tok
        if tok not in st.session_state.selected_words:
            st.session_state.selected_words.append(tok)

