import streamlit as st
import streamlit.components.v1 as components
import json

# 상태 초기화
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []

st.title("단어 클릭 테스트 - 완전 커스텀 방식")

text = "Человек идёт по улице Это тестовая строка".split()

# HTML 생성
html_words = ""
for w in text:
    color = "#1E88E5" if w in st.session_state.selected_words else "#333"
    html_words += f"""
    <span 
        style="margin-right:8px; cursor:pointer; color:{color}; font-size:18px;"
        onclick="parent.postMessage({{'clicked':'{w}'}}, '*')"
    >{w}</span>
    """

# HTML 렌더링
components.html(
    f"""
    <html>
    <body>{html_words}</body>
    <script>
    window.addEventListener('message', (event) => {{
        if (event.data.clicked) {{
            const params = new URLSearchParams(window.location.search);
            params.set('clicked_word', event.data.clicked);
            window.location.search = params.toString();
        }}
    }});
    </script>
    </html>
    """,
    height=120,
)

# Python 측 이벤트 처리
clicked = st.experimental_get_query_params().get("clicked_word", [""])[0]
if clicked:
    if clicked not in st.session_state.selected_words:
        st.session_state.selected_words.append(clicked)
    st.experimental_set_query_params()  # 쿼리 초기화

st.write("### 선택된 단어:", st.session_state.selected_words)
