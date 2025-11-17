# ─────────────────────────────
# URL 쿼리 파라미터 기반 클릭 처리 (수정됨)
# ─────────────────────────────
params = st.query_params
clicked_from_url = None
if "w" in params:
    clicked_from_url = params["w"]  # [0] 제거


# ─────────────────────────────
# 텍스트 입력
# ─────────────────────────────
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")

left, right = st.columns([2, 1], gap="large")


# ─────────────────────────────
# 왼쪽 영역 — 원문 텍스트 (인라인 하이퍼링크) (수정됨)
# ─────────────────────────────
with left:
    st.subheader("텍스트 분석 결과")
    st.caption("단어를 클릭하면 오른쪽에 기본형, 뜻, 예문이 표시되고, 아래 ‘선택한 단어 모음’에 누적됩니다.")

    # 클릭된 단어가 있으면 상태에 반영 + 누적
    if clicked_from_url:
        st.session_state.clicked_word = clicked_from_url
        if clicked_from_url not in st.session_state.selected_words:
            st.session_state.selected_words.append(clicked_from_url)

    # 텍스트를 word / non-word 단위로 split
    segments = re.split(r'(\w+)', text, flags=re.UNICODE)

    html_parts = []
    for seg in segments:
        if not seg:
            continue
        if re.fullmatch(r'\w+', seg, flags=re.UNICODE):
            word = seg
            # 아직 선택되지 않은 단어는 검은색, 선택된 단어는 파란색
            if word in st.session_state.selected_words:
                color = "#1E88E5"
                font_weight = "600"
            else:
                color = "#000000"
                font_weight = "400"
            href = f"?w={urllib.parse.quote_plus(word)}"
            
            # ⭐️ [수정] target="_self" 를 추가하여 현재 탭에서 열리도록 강제
            html_parts.append(
                f'<a href="{href}" target="_self" style="color:{color}; font-weight:{font_weight}; text-decoration:none;">'
                f'{html.escape(word)}</a>'
            )
        else:
            html_parts.append(html.escape(seg))

    html_text = "".join(html_parts)
    st.markdown(html_text, unsafe_allow_html=True)

    with st.expander("초기화"):
        if st.button("🔄 선택 & 누적 데이터 초기화"):
            st.session_state.clicked_word = None
            st.session_state.selected_words = []
            st.session_state.word_info = {}
            st.query_params.clear()
            st.rerun()
