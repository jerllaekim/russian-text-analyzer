import os
import re
import json
import html
import urllib.parse

import pandas as pd
import streamlit as st
from pymystem3 import Mystem
from google import genai  # google-genai 패키지


# ─────────────────────────────
# 기본 설정 + 전역 상태
# ─────────────────────────────
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기")

if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None          # 현재 상세보기 중인 단어(표면형)
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []          # 사용자가 선택한 단어(표면형) 리스트
if "word_info" not in st.session_state:
    # lemma 기준으로 뜻을 누적 저장
    # 예: {"человек": {"lemma": "человек", "ko_meanings": ["사람", "인간"]}, ...}
    st.session_state.word_info = {}


# 선택 단어 칩용 CSS
st.markdown(
    """
<style>
div.selected-word-chip > button {
    border-radius: 999px;
    padding: 2px 10px;
    margin: 3px;
    border: 1px solid #1E88E5;
    background-color: rgba(30, 136, 229, 0.06);
}
div.selected-word-chip-active > button {
    border-radius: 999px;
    padding: 2px 10px;
    margin: 3px;
    border: 1px solid #1E88E5;
    background-color: rgba(30, 136, 229, 0.18);
}
</style>
""",
    unsafe_allow_html=True,
)


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

요구사항:
- "ko_meanings"에는 너무 길지 않은 한국어 뜻 1~3개를 넣어라.
- "examples"에는 자연스러운 문장 2개를 넣어라.
- 각 예문에는 반드시 이 단어(또는 형태 변화된 형태)를 포함해야 한다.
- 반드시 JSON만 출력하고, 그 외의 텍스트는 출력하지 마라.
"""


@st.cache_data(show_spinner=False)
def fetch_from_gemini(word: str, lemma: str):
    prompt = build_prompt(word, lemma)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    text = response.text.strip()

    # ```json ... ``` 로 감싸져 오는 경우 제거
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])

    data = json.loads(text)
    return data


# ─────────────────────────────
# URL 쿼리 파라미터 기반 클릭 처리
# ─────────────────────────────
params = st.query_params
clicked_from_url = None
if "w" in params: # 'and params["w"]' 체크는 불필요합니다.
    clicked_from_url = params["w"]  # [0]을 제거하고 값 자체를 사용합니다.


# ─────────────────────────────
# 텍스트 입력
# ─────────────────────────────
text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка.")

left, right = st.columns([2, 1], gap="large")


# ─────────────────────────────
# 왼쪽 영역 — 원문 텍스트 (인라인 하이퍼링크)
# ─────────────────────────────
# ─────────────────────────────
# 왼쪽 영역 — 원문 텍스트 (인라인 하이퍼링크)
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
            
            # ⭐️ [핵심] target="_self" 가 있는지 다시 확인해 주세요.
            # 이것이 없으면 새 탭이 열리고 모든 기록이 초기화됩니다.
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


# ─────────────────────────────
# 오른쪽 영역 — 현재 선택 단어 상세
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

        # word_info 세션에 lemma 기준으로 누적 저장
        if ko_meanings:
            st.session_state.word_info[lemma] = {
                "lemma": lemma,
                "ko_meanings": ko_meanings,
            }

        # 한국어 뜻 표시
        if ko_meanings:
            st.markdown("**한국어 뜻:**")
            for m in ko_meanings:
                st.markdown(f"- {m}")
        else:
            st.write("한국어 뜻을 가져올 수 없습니다.")

        # 예문 표시
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

        # 🔎 외부 사전 / 코퍼스 링크
        st.markdown("### 🔗 외부 사전 / 코퍼스 검색")
        lemma_for_link = lemma or cw
        mt_url = f"https://www.multitran.com/m.exe?l1=2&l2=5&s={lemma_for_link}"
        rnc_url = f"https://ruscorpora.ru/search?search={lemma_for_link}"
        st.markdown(f"[Multitran에서 검색]({mt_url})  \n[러시아 국립 코퍼스에서 검색]({rnc_url})")

    else:
        st.info("왼쪽 텍스트에서 단어를 클릭하면 여기 정보가 나타납니다.")


# ─────────────────────────────
# 하단 — 선택한 단어 모음 (칩 + 표 + CSV)
# ─────────────────────────────
st.divider()
st.subheader("📝 선택한 단어 모음")

selected = st.session_state.selected_words
cw = st.session_state.clicked_word

word_info = st.session_state.word_info

if not selected and not word_info:
    st.caption("아직 클릭해서 누적된 단어가 없습니다. 위 텍스트에서 단어를 클릭해보세요.")
else:
    # 1) 칩 형태로 선택 단어들
    if selected:
        cols = st.columns(min(4, len(selected)))
        for idx, w in enumerate(selected):
            col = cols[idx % len(cols)]
            with col:
                if w == cw:
                    st.markdown('<div class="selected-word-chip-active">', unsafe_allow_html=True)
                    label = f"✅ {w}"
                    if st.button(label, key=f"sel_{w}_active"):
                        st.session_state.clicked_word = w
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="selected-word-chip">', unsafe_allow_html=True)
                    label = w
                    if st.button(label, key=f"sel_{w}"):
                        st.session_state.clicked_word = w
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # 2) lemma / 한국어 뜻 요약 표 + CSV (같은 섹션 안에 통합)
    if word_info:
        rows = []
        for lemma, info in word_info.items():
            meanings = info.get("ko_meanings", [])
            short_kr = "; ".join(meanings[:2])  # 한두 개만
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
# 맨 아래 — 직접 단어 검색
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

    # 직접 검색으로 가져온 것도 word_info에 누적
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
