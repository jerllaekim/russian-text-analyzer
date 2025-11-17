import os
import re
import json

import pandas as pd
import streamlit as st
from pymystem3 import Mystem
from google import genai


# ─────────────────────────────
# 기본 설정 + 전역 상태
# ─────────────────────────────
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None          # 현재 상세보기 단어(표면형)
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []          # 선택된 단어(표면형) 리스트
if "word_info" not in st.session_state:
    st.session_state.word_info = {}              # lemma -> {lemma, ko_meanings}


# ─────────────────────────────
# CSS: 단어 버튼을 텍스트처럼 보이게
# ─────────────────────────────
st.markdown(
    """
<style>
/* 단어 버튼용 래퍼 - 일반(검정) */
div.word-btn-normal > button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 4px 2px 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    color: #000000 !important;
    font-size: 1rem !important;
}

/* 단어 버튼용 래퍼 - 선택됨(파랑) */
div.word-btn-selected > button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 4px 2px 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    color: #1E88E5 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* 호버 시 밑줄만 */
div.word-btn-normal > button:hover,
div.word-btn-selected > button:hover {
    text-decoration: underline;
}

/* 선택한 단어 모음(칩 느낌) */
div.selected-chip > button {
    border-radius: 999px !important;
    padding: 2px 10px !important;
    margin: 3px !important;
    border: 1px solid #1E88E5 !important;
    background-color: rgba(30, 136, 229, 0.06) !important;
    color: #1E88E5 !important;
}
div.selected-chip-active > button {
    border-radius: 999px !important;
    padding: 2px 10px !important;
    margin: 3px !important;
    border: 1px solid #1E88E5 !important;
    background-color: rgba(30, 136, 229, 0.18) !important;
    color: #1E88E5 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────
# 형태소 분석기
# ─────────────────────────────
mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    lemmas = mystem.lemmatize(word)
    return (lemmas[0] if lemmas else word).strip()


# ─────────────────────────────
# Gemini 설정
# ─────────────────────────────
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY가 설정되어 있지 않습니다. Streamlit Secrets에 GEMINI_API_KEY를 넣어주세요.")
    st.stop()

client = genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
너는 러시아어-한국어 학습을 돕는 도우미이다.
러시아어 단어에 대해 간단한 한국어 뜻과 예문을 제공한다.
반드시 유효한 JSON만 출력해야 한다.
"""

def build_prompt(word: str, lemma: str) -> str:
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
"""

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word: str, lemma: str):
    prompt = build_prompt(word, lemma)
    res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = res.text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])
    return json.loads(text)


# ─────────────────────────────
# 텍스트 입력
# ─────────────────────────────
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")

# 단어 / 문장부호 분리
tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

left, right = st.columns([2, 1], gap="large")


# ─────────────────────────────
# 왼쪽: 텍스트 (텍스트 느낌 버튼, 가로 배열)
# ─────────────────────────────
with left:
    st.subheader("텍스트 분석 결과")
    st.caption("단어(검은 글씨)를 클릭하면 파란색으로 바뀌고, 오른쪽/하단에 정보가 표시됩니다.")

    # 핵심: 한 줄에 5개씩만 → 각 칸이 넓어서 글자가 세로로 안 쪼개짐
    row_size = 5
    for start in range(0, len(tokens), row_size):
        row_tokens = tokens[start:start + row_size]
        cols = st.columns(row_size)
        for j, tok in enumerate(row_tokens):
            col = cols[j]
            with col:
                if re.match(r"\w+", tok, flags=re.UNICODE):
                    # 이미 선택된 단어면 파란색, 아니면 검정
                    wrapper_class = "word-btn-selected" if tok in st.session_state.selected_words else "word-btn-normal"
                    st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
                    if st.button(tok, key=f"tok_{start}_{j}_{tok}"):
                        st.session_state.clicked_word = tok
                        if tok not in st.session_state.selected_words:
                            st.session_state.selected_words.append(tok)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # 문장부호는 그냥 출력
                    st.write(tok)

    with st.expander("초기화"):
        if st.button("🔄 선택 & 누적 데이터 초기화"):
            st.session_state.clicked_word = None
            st.session_state.selected_words = []
            st.session_state.word_info = {}
            st.rerun()


# ─────────────────────────────
# 오른쪽: 단어 정보
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

        # word_info에 누적
        if ko_meanings:
            st.session_state.word_info[lemma] = {
                "lemma": lemma,
                "ko_meanings": ko_meanings,
            }

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

        st.markdown("### 🔗 외부 사전 / 코퍼스 검색")
        lemma_for_link = lemma or cw
        mt_url = f"https://www.multitran.com/m.exe?l1=2&l2=5&s={lemma_for_link}"
        rnc_url = f"https://ruscorpora.ru/search?search={lemma_for_link}"
        st.markdown(f"[Multitran에서 검색]({mt_url})  \n[러시아 국립 코퍼스에서 검색]({rnc_url})")
    else:
        st.info("왼쪽 텍스트에서 단어를 클릭하면 여기 정보가 나타납니다.")


# ─────────────────────────────
# 하단: 선택한 단어 모음 + lemma/뜻 표 + CSV
# ─────────────────────────────
st.divider()
st.subheader("📝 선택한 단어 모음")

selected = st.session_state.selected_words
cw = st.session_state.clicked_word
word_info = st.session_state.word_info

if not selected and not word_info:
    st.caption("아직 클릭해서 누적된 단어가 없습니다. 위 텍스트에서 단어를 클릭해보세요.")
else:
    # 칩 형태로 표시 (그냥 시각용)
    if selected:
        cols = st.columns(min(4, len(selected)))
        for idx, w in enumerate(selected):
            col = cols[idx % len(cols)]
            with col:
                if w == cw:
                    st.markdown("<div class='selected-chip-active'>", unsafe_allow_html=True)
                    st.button(f"✅ {w}", key=f"chip_{w}_active")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='selected-chip'>", unsafe_allow_html=True)
                    st.button(w, key=f"chip_{w}")
                    st.markdown("</div>", unsafe_allow_html=True)

    # lemma / 한국어 뜻 표 + CSV
    if word_info:
        rows = []
        for lemma, info in word_info.items():
            meanings = info.get("ko_meanings", [])
            short_kr = "; ".join(meanings[:2])
            rows.append({"lemma": lemma, "한국어 뜻": short_kr})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="💾 CSV로 저장하기",
            data=csv_bytes,
            file_name="russian_words.csv",
            mime="text/csv",
        )


# ─────────────────────────────
# 맨 아래: 직접 단어 검색
# ─────────────────────────────
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
        st.session_state.word_info[lemma] = {
            "lemma": lemma,
            "ko_meanings": ko_meanings,
        }

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
