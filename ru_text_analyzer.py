import os
import re
import json
import streamlit as st
from pymystem3 import Mystem
from google import genai  # google-genai 패키지

# ───────────────── 기본 설정 ─────────────────

st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

# 세션 상태 초기화
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None

# 형태소 분석기
mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    lemmas = mystem.lemmatize(word)
    return (lemmas[0] if lemmas else word).strip()

# ───────────────── Gemini 설정 ─────────────────

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit Secrets에 API 키를 추가하세요.")
    st.stop()

client = genai.Client(api_key=api_key)

GEMINI_SYSTEM_PROMPT = """
너는 러시아어-한국어 학습을 돕는 도우미이다.
러시아어 단어에 대해 한국어 의미와 예문을 제공한다.
반드시 유효한 JSON만 출력해야 한다.
"""

def build_user_prompt(word: str, lemma: str) -> str:
    return f"""
러시아어 단어: {word}
기본형(lemma): {lemma}

다음 정보를 JSON 형식으로 만들어라:

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

요구사항:
- "ko_meanings"에는 너무 장황하지 않은 한국어 뜻 1~3개를 넣어라.
- "examples"에는 자연스러운 문장 2개를 넣어라.
- 모든 예문은 B1~B2 수준의 자연스러운 러시아어여야 한다.
- 각 예문에는 반드시 이 단어(또는 어형 변화된 형태)를 포함해야 한다.
- 반드시 JSON만 출력하고, 그 외의 텍스트(설명, 말머리, 주석 등)는 출력하지 마라.
"""

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word: str, lemma: str):
    """Gemini로부터 뜻 + 예문 JSON 받아오기."""
    prompt = build_user_prompt(word, lemma)
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            {"role": "system", "parts": [GEMINI_SYSTEM_PROMPT]},
            {"role": "user", "parts": [prompt]},
        ],
    )

    text = response.text.strip()
    # 혹시 코드블록(````json`)으로 감싸져 오면 제거
    if text.startswith("```"):
        text = text.strip("`")
        # 맨 첫 줄에 json 같은 언어 태그가 붙어있을 수 있음
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])
    data = json.loads(text)
    return data

# ───────────────── UI ─────────────────

text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")

tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("텍스트 분석 결과")
    st.caption("단어 버튼을 클릭하면 오른쪽에 기본형, 한국어 뜻, 예문이 표시됩니다.")

    for i, tok in enumerate(tokens):
        if re.match(r"\w+", tok, flags=re.UNICODE):
            if st.button(tok, key=f"tok_{i}"):
                st.session_state.clicked_word = tok
        else:
            st.write(tok)

    with st.expander("초기화"):
        if st.button("🔄 선택 초기화"):
            st.session_state.clicked_word = None
            st.rerun()

with col_right:
    st.subheader("📚 단어 정보")

    cw = st.session_state.clicked_word
    if cw:
        lemma = lemmatize_ru(cw)
        st.markdown(f"**선택된 단어:** {cw}")
        st.markdown(f"**기본형(lemma):** *{lemma}*")

        try:
            info = fetch_from_gemini(cw, lemma)
            ko_meanings = info.get("ko_meanings", [])
            examples = info.get("examples", [])
        except Exception as e:
            st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
            ko_meanings = []
            examples = []

        if ko_meanings:
            st.markdown("**한국어 뜻:**")
            for m in ko_meanings:
                st.markdown(f"- {m}")
        else:
            st.write("한국어 뜻을 가져올 수 없습니다.")

        if examples:
            st.markdown("### 📖 예문")
            for ex in examples:
                ru = ex.get("ru", "")
                ko = ex.get("ko", "")
                if ru:
                    st.markdown(f"- **{ru}**")
                if ko:
                    st.markdown(f" → {ko}")
        else:
            st.write("예문을 가져올 수 없습니다.")
    else:
        st.info("왼쪽에서 단어를 클릭하면 여기 기본형, 뜻, 예문이 나타납니다.")

st.divider()
st.subheader("🔍 직접 단어 검색")

manual = st.text_input("텍스트와 상관없이, 직접 단어를 입력해 분석할 수도 있습니다.", "")
if manual:
    lemma = lemmatize_ru(manual)
    st.markdown(f"**입력 단어:** {manual}")
    st.markdown(f"**기본형(lemma):** *{lemma}*")
    try:
        info = fetch_from_gemini(manual, lemma)
        ko_meanings = info.get("ko_meanings", [])
        examples = info.get("examples", [])
    except Exception as e:
        st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
        ko_meanings = []
        examples = []

    if ko_meanings:
        st.markdown("**한국어 뜻:**")
        for m in ko_meanings:
            st.markdown(f"- {m}")
    if examples:
        st.markdown("### 📖 예문")
        for ex in examples:
            ru = ex.get("ru", "")
            ko = ex.get("ko", "")
            if ru:
                st.markdown(f"- **{ru}**")
            if ko:
                st.markdown(f" → {ko}")
