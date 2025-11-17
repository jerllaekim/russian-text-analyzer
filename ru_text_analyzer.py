import re
import streamlit as st
from pymystem3 import Mystem

st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

# 세션 상태 초기화
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None
if "user_glosses" not in st.session_state:
    st.session_state.user_glosses = {}  # {단어: 한국어 뜻}

mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    lemmas = mystem.lemmatize(word)
    return (lemmas[0] if lemmas else word).strip()

# 텍스트 입력
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")

# 단어 / 문장부호 분리
tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("텍스트 분석 결과")
    st.caption("단어 버튼을 클릭하면 오른쪽에 기본형이 표시됩니다.")

    for i, tok in enumerate(tokens):
        if re.match(r"\w+", tok, flags=re.UNICODE):
            if st.button(tok, key=f"tok_{i}"):
                st.session_state.clicked_word = tok
        else:
            st.write(tok)

    with st.expander("초기화"):
        if st.button("🔄 선택 및 뜻 초기화"):
            st.session_state.clicked_word = None
            st.session_state.user_glosses = {}
            st.rerun()

with col_right:
    st.subheader("📚 단어 정보")

    cw = st.session_state.clicked_word
    if cw:
        lemma = lemmatize_ru(cw)
        st.markdown(f"**선택된 단어:** {cw}")
        st.markdown(f"**기본형(lemma):** *{lemma}*")

        # 이전에 적어둔 뜻 있으면 불러오기
        prev_gloss = st.session_state.user_glosses.get(cw, "")
        gloss = st.text_input("한국어 뜻 (직접 입력)", value=prev_gloss, key=f"gloss_{cw}")
        st.session_state.user_glosses[cw] = gloss

        # 외부 사전 링크 (원한다면 PPT에서 “확장 가능 기능”으로 설명 가능)
        mt_url = f"https://www.multitran.com/m.exe?l1=2&l2=5&s={lemma}"
        yd_url = f"https://translate.yandex.com/?source_lang=ru&target_lang=ko&text={lemma}"
        st.markdown(f"[Multitran에서 보기]({mt_url})  \n[Yandex Translate에서 보기]({yd_url})")
    else:
        st.info("왼쪽에서 단어를 클릭하면 여기 기본형과 한국어 뜻 입력 칸이 나타납니다.")

st.divider()
st.subheader("🔍 직접 단어 검색")

manual = st.text_input("텍스트와 상관없이, 직접 단어를 입력해 분석할 수도 있습니다.", "")
if manual:
    lemma = lemmatize_ru(manual)
    st.markdown(f"**입력 단어:** {manual}")
    st.markdown(f"**기본형(lemma):** *{lemma}*")

