import os
import re
import json
import streamlit as st
from pymystem3 import Mystem
from google import genai  # google-genai 패키지


# ─────────────────────────────
# 기본 설정
# ─────────────────────────────
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None


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
    st.error("GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit Secrets에 API 키를 넣어주세요.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
너는 러시아어-한국어 학습을 돕는 도우미이다.
러시아어 단어에 대해 간단한 한국어 뜻과 예문을 제공한다.
반드시 유효한 JSON만 출력해야 한다.
"""


def build_prompt(word: str, lemma: str) -> str:
    """
    Gemini에 요청할 전체 prompt를 하나의 문자열로 만든다.
    (google-genai는 system/user 메시지 분리 필요 없음 → 문자열 하나면 됨)
    """
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

요구사항:
- "ko_meanings"에는 너무 길지 않은 한국어 뜻 1~3개를 넣어라.
- "examples"에는 자연스러운 문장 2개를 넣어라.
- 각 예문에는 반드시 이 단어(또는 형태 변화된 형태)를 포함해야 한다.
- 반드시 JSON만 출력하고, 그 외의 텍스트는 출력하지 마라.
"""


@st.cache_data(show_spinner=False)
def fetch_from_gemini(word: str, lemma: str):
    """
    Gemini API 호출 → JSON 파싱 → Python dict로 반환.
    """
    prompt = build_prompt(word, lemma)

    # google-genai의 최신 구조: contents에 문자열 하나만 전달하면 됨
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    text = response.text.strip()

    # 혹시 ```json … ``` 이런 코드블록으로 오면 제거
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])

    data = json.loads(text)
    return data


# ─────────────────────────────
# UI 시작
# ─────────────────────────────
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")

# 단어 / 문장부호 분리
tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

left, right = st.columns([2, 1], gap="large")


# ─────────────────────────────
# 왼쪽 영역 — 텍스트 분석
# ─────────────────────────────
with left:
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


# ─────────────────────────────
# 오른쪽 영역 — 단어 정보 표시
# ─────────────────────────────
with right:
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

        # 한국어 뜻
        if ko_meanings:
            st.markdown("**한국어 뜻:**")
            for m in ko_meanings:
                st.markdown(f"- {m}")

        # 예문
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
        st.info("왼쪽 단어를 클릭하면 여기 정보가 나타납니다.")


# ─────────────────────────────
# 하단 — 직접 검색
# ─────────────────────────────
st.divider()
st.subheader("🔍 직접 단어 검색")

manual = st.text_input("직접 단어를 입력하여 분석할 수도 있습니다.", "")

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
