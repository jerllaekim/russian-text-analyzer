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
    st.session_state.selected_words = []          # 선택된 단어(표면형)
if "word_info" not in st.session_state:
    st.session_state.word_info = {}              # lemma -> {lemma, ko_meanings}


# ─────────────────────────────
# CSS
#  - 체크박스 네모/토글 숨기기
#  - 단어는 가로로 나열
#  - 선택된(체크된) 단어만 파란색 + 밑줄
# ─────────────────────────────
st.markdown(
    """
<style>
/* 체크박스 컨테이너: 가로로 나열 & 줄바꿈은 자연스럽게 */
div[data-testid="stCheckbox"] {
    display: inline-flex;
    align-items: center;
    margin: 0 6px 4px 0;
}

/* 라벨 전체 */
div[data-testid="stCheckbox"] > label {
    display: inline-flex;
    align-items: center;
    cursor: pointer;
}

/* 실제 체크박스 input은 완전히 숨김 */
div[data-testid="stCheckbox"] input[type="checkbox"] {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
    pointer-events: none;
}

/* 기본 단어 텍스트 스타일 */
div[data-testid="stCheckbox"] span {
    color: #333333;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* 선택된(체크된) 단어만 파란색 + 밑줄
   Streamlit이 내부에 role="checkbox" 요소를 두는데
   그게 aria-checked="true"일 때, 바로 뒤의 span(텍스트)을 스타일링 */
div[data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] + span {
    color: #1E88E5;
    text-decoration: underline;
}

/* 하단 선택 단어 모음 칩 스타일 (기존 느낌 유지) */
.selected-chip button {
    border-radius: 999px !important;
    padding: 2px 10px !important;
    margin: 3px !important;
    border: 1px solid #1E88E5 !important;
    background-color: rgba(30, 136, 229, 0.06) !important;
    color: #1E88E5 !important;
    font-size: 0.9rem !important;
}
.selected-chip-active button {
    border-radius: 999px !important;
    padding: 2px 10px !important;
    margin: 3px !important;
    border: 1px solid #1E88E5 !important;
    background-color: rgba(30, 136, 229, 0.18) !important;
    color: #1E88E5 !important;
    font-size: 0.9rem !important;
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

# 원문 그대로 보여주기 (클릭 X)
st.subheader("원문 텍스트")
st.write(text)

# 단어만 추출 (소문자/대문자 포함, 구두점 제외)
tokens = re.findall(r"\w+", text, flags=re.UNICODE)
unique_tokens = sorted(set(tokens), key=lambda x: tokens.index(x))  # 등장 순서 유지


left, right = st.columns([2, 1], gap="large")


# ─────────────────────────────
# 왼쪽: 단어 목록
#   - 체크박스로 멀티 선택
#   - 기본 UI는 숨기고 단어만 보이게
#   - 선택된 단어는 파란색으로 유지
# ─────────────────────────────
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")
    st.caption("단어를 클릭하면 파란색으로 바뀌고, 오른쪽에 정보가 표시되며, 하단에 누적됩니다.")

    prev_selected = list(st.session_state.selected_words)

    if not unique_tokens:
        st.info("텍스트에서 단어를 찾지 못했습니다.")
    else:
        checkbox_values = {}
        for tok in unique_tokens:
            key = f"chk_{tok}"
            default_checked = tok in prev_selected
            # 체크박스를 쓰지만, CSS로 네모를 숨겨서 '단어만 보이는' 형태
            checked = st.checkbox(tok, key=key, value=default_checked)
            checkbox_values[tok] = checked

        # 현재 선택된 단어 리스트
        curr_selected = [tok for tok, v in checkbox_values.items() if v]

        # 새로 추가된 단어, 해제된 단어 계산
        added = [tok for tok in curr_selected if tok not in prev_selected]
        removed = [tok for tok in prev_selected if tok not in curr_selected]

        # clicked_word 업데이트
        if added:
            # 방금 새로 선택된 단어를 현재 클릭 단어로
            st.session_state.clicked_word = added[-1]
        else:
            cw = st.session_state.clicked_word
            # 기존 클릭 단어가 선택 해제되었으면 마지막 선택 단어로 이동
            if cw and cw not in curr_selected:
                if curr_selected:
                    st.session_state.clicked_word = curr_selected[-1]
                else:
                    st.session_state.clicked_word = None

        st.session_state.selected_words = curr_selected

    with st.expander("초기화"):
        if st.button("🔄 선택 & 누적 데이터 초기화", key="reset_all"):
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
        st.info("왼쪽에서 단어를 하나 이상 클릭(선택)하면 여기 정보가 나타납니다.")


# ─────────────────────────────
# 하단: 선택한 단어 모음 + lemma/뜻 표 + CSV
# ─────────────────────────────
st.divider()
st.subheader("📝 선택한 단어 모음")

selected = st.session_state.selected_words
cw = st.session_state.clicked_word
word_info = st.session_state.word_info

if not selected and not word_info:
    st.caption("아직 클릭해서 누적된 단어가 없습니다. 위 단어 목록에서 단어를 선택해보세요.")
else:
    if selected:
        st.caption("아래 단어를 클릭하면 다시 상세 정보를 볼 수 있습니다.")
        cols = st.columns(min(4, len(selected)))
        for idx, w in enumerate(selected):
            col = cols[idx % len(cols)]
            with col:
                if w == cw:
                    st.markdown("<div class='selected-chip-active'>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='selected-chip'>", unsafe_allow_html=True)
                if st.button(w, key=f"selectedchip_{w}"):
                    st.session_state.clicked_word = w
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

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
