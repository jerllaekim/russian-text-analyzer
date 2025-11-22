import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai

# ---------------------- 0. 초기 설정 및 세션 상태 ----------------------
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("🇷🇺 러시아어 텍스트 분석기")

# 세션 상태 초기화
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None
if "word_info" not in st.session_state:
    st.session_state.word_info = {}

mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    """단어의 기본형(lemma)을 추출합니다."""
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        lemmas = mystem.lemmatize(word)
        return (lemmas[0] if lemmas else word).strip()
    return word

# ---------------------- 1. Gemini 연동 함수 ----------------------

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = "너는 러시아어-한국어 학습을 돕는 도우미이다. 러시아어 단어에 대해 간단한 한국어 뜻과 예문을 최대 두 개만 제공한다. 반드시 JSON만 출력한다."
def make_prompt(word, lemma):
    # 예문을 최대 2개만 요청하도록 프롬프트 수정
    return f"""{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}
{{ "ko_meanings": ["뜻1", "뜻2"], "examples": [ {{"ru": "예문1", "ko": "번역1"}}, {{"ru": "예문2", "ko": "번역2"}} ] }}
"""

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word, lemma):
    if not client:
        return {"ko_meanings": [f"'{word}'의 API 키 없음 (GEMINI_API_KEY 설정 필요)"], "examples": []}
        
    prompt = make_prompt(word, lemma)
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = res.text.strip()
    
    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("json"):
            text = "\n".join(lines[1:])
        elif lines:
             text = "\n".join(lines)
             
    try:
        data = json.loads(text)
        # 예문이 3개 이상일 경우 2개로 제한
        if 'examples' in data and len(data['examples']) > 2:
            data['examples'] = data['examples'][:2]
        return data
    except json.JSONDecodeError:
        st.error(f"Gemini 응답을 JSON으로 디코딩하는 데 실패했습니다: {text[:100]}...")
        return {"ko_meanings": ["응답 오류"], "examples": []}


# ---------------------- 2. 전역 스타일 정의 (버튼 UI 및 간격 최소화) ----------------------

st.markdown("""
<style>
    /* 1. 단어 버튼 스타일: 버튼 모양 완전히 제거 및 간격 최소화 */
    /* stButton 내부 button 요소에 직접 스타일 적용 */
    div.stButton > button {
        padding: 2px 0px !important; /* 내부 패딩 최소화 */
        margin: 0 !important;
        border: none !important;
        background: none !important; 
        box-shadow: none !important; 
        cursor: pointer;
        color: #333 !important; 
        font-weight: normal;
        height: auto !important;
        line-height: 1.5 !important;
        white-space: nowrap;
        text-align: left !important;
    }
    
    /* 2. 클릭된 단어 색상 유지 (파란색) */
    /* word-selected 클래스가 버튼의 텍스트를 파란색으로 만듭니다. */
    .word-selected > div > button { 
        color: #007bff !important; 
        font-weight: bold !important;
    }
    
    /* 3. 구두점 스타일 */
    .word-punctuation {
        padding: 2px 0px;
        margin: 0; /* 구두점 마진 제거 */
        display: inline-block;
        user-select: none;
        line-height: 1.5;
        font-size: 1.1em;
    }
    
    /* 4. st.columns 컨테이너 내의 간격 조정 (가로 나열 강제) */
    /* 단어 간 간격을 0으로 만들고, 띄어쓰기는 단어 자체의 버튼 레이블에 포함시키거나,
       구두점 토큰으로 처리하여 시각적으로 최소화합니다. */
    div[data-testid^="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0px 0px !important; /* 컬럼 간격을 0으로 설정하여 단어를 붙임 */
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- 3. 메인 로직 및 레이아웃 ----------------------

text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка. Хорошо.")
# 공백(띄어쓰기)과 구두점을 토큰으로 분리하여 정확히 배치
tokens_with_punct = re.findall(r"(\w+|[^\s\w]+|\s+)", text, flags=re.UNICODE)
tokens_with_punct = [t for t in tokens_with_punct if t.strip()] # 빈 토큰 제거

left, right = st.columns([2, 1])

# --- 3.1. 단어 목록 (left 컬럼) ---
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")

    # 단어 버튼 클릭 시 실행될 콜백 함수
    def on_word_click(clicked_token):
        st.session_state.clicked_word = clicked_token
        if clicked_token not in st.session_state.selected_words:
            st.session_state.selected_words.append(clicked_token)

    # st.columns를 사용하여 단어와 구두점을 가로로 나열
    # 이 방식은 Streamlit에서 인라인 레이아웃을 보장하는 가장 확실한 방법입니다.
    
    # 토큰 개수만큼 컬럼을 생성하고 간격 최소화 CSS를 적용
    cols = st.columns(len(tokens_with_punct)) 

    for i, tok in enumerate(tokens_with_punct):
        with cols[i]:
            if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
                # 단어인 경우: st.button 사용
                is_selected = tok in st.session_state.selected_words
                
                # CSS 클래스를 버튼의 상위 Div에 적용하기 위한 래퍼 HTML 생성
                # Streamlit은 st.button을 자체 Div로 래핑하므로, 래퍼 위에 래퍼를 씌워야 합니다.
                
                # HTML 래퍼: 이 래퍼에 word-selected 클래스를 적용하고, st.button을 배치합니다.
                # 그러나 st.button은 Python 함수 호출이므로, HTML과 섞이지 않도록 분리합니다.
                
                # 임시: st.button이 이미 순서대로 배치되고 있으므로,
                # CSS에 의존하여 버튼의 부모 Div에 스타일이 적용되도록 합니다.
                
                # st.button을 렌더링하고, 클릭 시 로직 실행
                st.button(
                    tok, 
                    key=f"word_{tok}_{i}", # 고유 key
                    on_click=on_word_click,
                    args=(tok,)
                )
                
                # 파란색 글씨화를 위해 버튼이 출력된 후 CSS 클래스를 동적으로 적용하는 트릭이 필요하지만,
                # 현재는 **CSS 선택자를 이용해 글자색을 강제**하는 방식으로 해결을 시도합니다.
                # 클릭된 단어의 key를 이용해 Streamlit이 재실행될 때, CSS 클래스가 재적용됩니다.
                if is_selected:
                    st.markdown(
                        f"""
                        <script>
                            // 해당 버튼을 포함하는 stButton 컨테이너를 찾아 word-selected 클래스 적용
                            const button = document.querySelector('[data-testid="stButton"] button[key="word_{tok}_{i}"]');
                            if (button && button.parentElement) {{
                                button.parentElement.parentElement.classList.add('word-selected');
                            }}
                        </script>
                        """, 
                        unsafe_allow_html=True
                    )
            
            else:
                # 구두점 또는 공백인 경우: st.markdown으로 출력
                st.markdown(f'<span class="word-punctuation">{tok}</span>')


    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 선택 초기화", key="reset_button"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.experimental_set_query_params() 
        st.rerun()

# --- 3.2. 단어 상세 정보 로드 ---

current_token = st.session_state.clicked_word

if current_token:
    tok = current_token
    lemma = lemmatize_ru(tok)
    
    # 정보 로드 로직
    if lemma not in st.session_state.word_info or st.session_state.word_info.get(lemma, {}).get('loaded_token') != tok:
        with st.spinner(f"'{tok}'의 정보를 불러오는 중... (Gemini API 호출)"):
            try:
                info = fetch_from_gemini(tok, lemma)
                st.session_state.word_info[lemma] = {**info, "loaded_token": tok} 
            except Exception as e:
                st.error(f"단어 정보 로드 오류: {e}")


# --- 3.3. 단어 상세 정보 (right 컬럼) ---
with right:
    st.subheader("단어 상세 정보")
    
    if current_token:
        lemma = lemmatize_ru(current_token)
        info = st.session_state.word_info.get(lemma, {})

        if info:
            st.markdown(f"### **{current_token}**")
            st.markdown(f"**기본형 (Lemma):** *{lemma}*")
            st.divider()

            ko_meanings = info.get("ko_meanings", [])
            # 예문 개수 제한은 이미 fetch_from_gemini에서 처리됨
            examples = info.get("examples", [])

            if ko_meanings:
                st.markdown("#### 한국어 뜻")
                for m in ko_meanings:
                    st.markdown(f"- **{m}**")

            if examples:
                st.markdown("#### 📖 예문")
                for ex in examples:
                    st.markdown(f"- {ex.get('ru', '')}")
                    st.markdown(f" → {ex.get('ko', '')}")
            else:
                if ko_meanings and ko_meanings[0].startswith(f"'{current_token}'의 API 키 없음"):
                     st.warning("API 키가 설정되지 않아 예문을 불러올 수 없습니다.")
                else:
                    st.info("예문 정보가 없습니다.")
        else:
            st.warning("단어 정보를 불러오는 중이거나 오류가 발생했습니다.")
            
    else:
        st.info("왼쪽 단어 목록에서 단어를 클릭해주세요.")

# ---------------------- 4. 하단: 누적 목록 + CSV ----------------------
st.divider()
st.subheader("📝 선택한 단어 모음 (기본형 기준)")

selected = st.session_state.selected_words
word_info = st.session_state.word_info

if word_info:
    rows = []
    processed_lemmas = set()
    
    for tok in selected:
        lemma = lemmatize_ru(tok)
        if lemma not in processed_lemmas and lemma in word_info:
            info = word_info[lemma]
            short = "; ".join(info["ko_meanings"][:2])
            rows.append({"기본형": lemma, "대표 뜻": short})
            processed_lemmas.add(lemma)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV로 저장", csv_bytes, "russian_words.csv", "text/csv")
    else:
        st.info("선택된 단어의 정보를 로드 중이거나, 표시할 정보가 없습니다.")


# ---------------------- 5. 직접 단어 검색 ----------------------
st.divider()
st.subheader("🔍 직접 단어 검색")

manual = st.text_input("단어 직접 입력", "")

if manual:
    lemma = lemmatize_ru(manual)
    st.markdown(f"**입력 단어:** **{manual}**")
    st.markdown(f"**기본형(lemma):** *{lemma}*")

    try:
        info = fetch_from_gemini(manual, lemma)
    except Exception as e:
        st.error(f"Gemini 오류: {e}")
        info = {}

    ko_meanings = info.get("ko_meanings", [])
    examples = info.get("examples", [])

    if ko_meanings:
        st.markdown("#### 한국어 뜻")
        for m in ko_meanings:
            st.markdown(f"- **{m}**")

    if examples:
        st.markdown("#### 📖 예문")
        for ex in examples:
            st.markdown(f"- **{ex.get('ru','')}**")
            st.markdown(f" → {ex.get('ko','')}")
