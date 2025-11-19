import streamlit as st
import re

# 상태 초기화
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None

st.set_page_config(layout="wide")
st.title("러시아어 텍스트 분석기")


# --- CSS ---
st.markdown("""
<style>
.word {
    display: inline-block;
    margin-right: 6px;
    font-size: 0.95rem;
    cursor: pointer;
    color: #333;
}

.word:hover {
    text-decoration: underline;
}

.word.selected {
    color: #1E88E5;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)


# --- JS + HTML 클릭 이벤트 처리 ---
click_js = """
<script>
function selectWord(word) {
    window.parent.postMessage({type: 'word_click', word: word}, "*");
}
</script>
"""

st.markdown(click_js, unsafe_allow_html=True)



# --- 메시지 리스너 ---
# Streamlit >= 1.32 에서 작동하는 방법
msg = st.experimental_get_query_params().get("word_click_signal", None)
if msg:
    w = msg[0]
    if w not in st.session_state.selected_words:
        st.session_state.selected_words.append(w)
    st.session_state.clicked_word = w


# --- 텍스트 입력 ---
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")
tokens = re.findall(r"\w+", text, flags=re.UNICODE)
tokens = list(dict.fromkeys(tokens))


# --- 단어 목록 렌더링 (HTML로 직접) ---
html_words = ""
for tok in tokens:
    cls = "word"
    if tok in st.session_state.selected_words:
        cls = "word selected"
    html_words += f'<span class="{cls}" onclick="selectWord(\'{tok}\')">{tok}</span> '

# Streamlit rerun 유도용 hidden param
st.experimental_set_query_params(word_click_signal=st.session_state.clicked_word or "")

st.markdown("### 단어 목록 (텍스트에서 추출)")
st.markdown(html_words, unsafe_allow_html=True)
