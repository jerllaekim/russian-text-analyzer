import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai

# (초기 설정 및 Gemini 연동 함수는 이전과 동일)
# ... (생략) ...
st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("🇷🇺 러시아어 텍스트 분석기")

if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None
if "word_info" not in st.session_state:
    st.session_state.word_info = {}
if "manual_search_word" not in st.session_state:
    st.session_state.manual_search_word = ""

mystem = Mystem()

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        lemmas = mystem.lemmatize(word)
        return (lemmas[0] if lemmas else word).strip()
    return word

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
client = genai.Client(api_key=api_key) if api_key else None

SYSTEM_PROMPT = "너는 러시아어-한국어 학습을 돕는 도우미이다. 러시아어 단어에 대해 간단한 한국어 뜻과 예문을 최대 두 개만 제공한다. 반드시 JSON만 출력한다."
def make_prompt(word, lemma):
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
    
    try:
        if text.startswith("```"):
            text = text.strip("`")
            lines = text.splitlines()
            if lines and lines[0].lower().startswith("json"):
                text = "\n".join(lines[1:])
            elif lines:
                 text = "\n".join(lines)
                 
        start_index = text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_text = text[start_index : end_index + 1]
        else:
            json_text = text
            
        data = json.loads(json_text)
        
        if 'examples' in data and len(data['examples']) > 2:
            data['examples'] = data['examples'][:2]
        return data
    
    except json.JSONDecodeError:
        st.error(f"Gemini 응답을 JSON으로 디코딩하는 데 실패했습니다. 원본 텍스트 시작: {text[:100]}...")
        return {"ko_meanings": ["JSON 파싱 오류"], "examples": []}


# ---------------------- 2. 전역 스타일 및 JavaScript 정의 (자동 선택 기능) ----------------------

# JavaScript: 단어 클릭 시, 숨겨진 입력 필드에 단어를 넣고 전체 선택 후, 검색창에 자동으로 넣습니다.
st.markdown("""
<script>
    function selectTextForCopy(word) {
        // 1. 숨겨진 복사 필드를 찾습니다. (key="hidden_copy_field"로 지정될 필드)
        const copyField = document.querySelector('[aria-label="Hidden Copy Field"]');
        
        if (copyField) {
            // 2. 값을 설정하고 전체 선택합니다.
            copyField.value = word;
            copyField.select(); // 텍스트를 선택 상태로 만듭니다.
            
            // 3. (선택 사항) 사용자에게 Ctrl+C를 누르도록 알림
            alert(`'${word}'가 선택되었습니다. Ctrl+C (Cmd+C)를 눌러 복사 후, 위 검색창에 붙여넣으세요.`);
        }
        
        // 4. 자동 검색 필드에 값 입력 시도 (이전 자동 검색 로직)
        const inputField = document.querySelector('[aria-label="단어 직접 입력"]');
        if (inputField) {
            inputField.value = word;
            const event = new Event('input', { bubbles: true });
            inputField.dispatchEvent(event);
        }
    }
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* 1. 복사/검색 자동 입력을 위한 숨겨진 필드 */
    /* stTextInput의 컨테이너를 숨깁니다. */
    div[data-testid="stTextInput"]:has(input[aria-label="Hidden Copy Field"]) {
        display: none;
    }

    /* 2. (나머지 CSS는 이전과 동일) */
    .word-span {
        cursor: pointer;
        padding: 0px 0px;
        margin: 0px 0px;
        display: inline-block;
        transition: color 0.2s;
        user-select: none;
        white-space: pre; 
        font-size: 1.25em;
    }
    .word-span:hover {
        color: #007bff;
        text-decoration: underline;
    }
    
    .word-selected {
        color: #007bff !important; 
        font-weight: bold;
    }
    
    .word-punctuation {
        padding: 0px 0px;
        margin: 0;
        display: inline-block;
        user-select: none;
        white-space: pre;
        font-size: 1.25em;
    }
    
    .text-container {
        line-height: 2.0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- 3. 직접 단어 검색 (상단) 및 처리 로직 ----------------------
st.divider()
st.subheader("🔍 직접 단어 검색")

# ❗ 숨겨진 복사/선택 필드: CSS로 숨겨집니다.
st.text_input("Hidden Copy Field", key="hidden_copy_field", label_visibility="collapsed") 

# 검색 입력 필드 
manual_input = st.text_input("단어 직접 입력", key="manual_search_word")

# 검색 입력 처리 로직
if manual_input:
    if manual_input not in st.session_state.selected_words:
        st.session_state.selected_words.append(manual_input)
    
    st.session_state.clicked_word = manual_input
    
    lemma = lemmatize_ru(manual_input)
    st.markdown(f"**입력 단어:** **{manual_input}**")
    st.markdown(f"**기본형(lemma):** *{lemma}*")

    try:
        info = fetch_from_gemini(manual_input, lemma)
        
        if lemma not in st.session_state.word_info or st.session_state.word_info.get(lemma, {}).get('loaded_token') != manual_input:
             st.session_state.word_info[lemma] = {**info, "loaded_token": manual_input} 
        
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
    
    st.markdown("---")


# ---------------------- 4. 메인 텍스트 및 레이아웃 ----------------------

text = st.text_area("텍스트를 입력하세요", "Человек идёт по улице. Это тестовая строка. Хорошо.", height=150)
tokens_with_punct = re.findall(r"(\w+|[^\s\w]+|\s+)", text, flags=re.UNICODE)

left, right = st.columns([2, 1])

# --- 4.1. 단어 목록 및 클릭 처리 (left 컬럼) ---
with left:
    st.subheader("단어 목록 (텍스트에서 추출)")

    html_all = ['<div class="text-container">']
    
    for tok in tokens_with_punct:
        if re.fullmatch(r'\w+', tok, flags=re.UNICODE):
            is_selected = tok in st.session_state.selected_words
            css = "word-span"
            
            if is_selected:
                css += " word-selected"
            
            # onclick: JavaScript 함수 호출 (단어를 숨겨진 필드에 넣어 자동 선택)
            html_all.append(
                f'<span class="{css}" onclick="selectTextForCopy(\'{tok}\');">'
                f'{tok}'
                f'</span>'
            )

        else:
            # 구두점 또는 공백
            html_all.append(f'<span class="word-punctuation">{tok}</span>')

    html_all.append('</div>')
    
    st.markdown("".join(html_all), unsafe_allow_html=True)
    
    # 초기화 버튼
    st.markdown("---")
    if st.button("🔄 선택 및 검색 초기화", key="reset_button"):
        st.session_state.selected_words = []
        st.session_state.clicked_word = None
        st.session_state.word_info = {}
        st.session_state.manual_search_word = ""
        st.rerun()

# --- 4.2. 단어 상세 정보 (right 컬럼) ---
with right:
    st.subheader("단어 상세 정보")
    
    current_token = st.session_state.clicked_word
    
    if current_token:
        lemma = lemmatize_ru(current_token)
        info = st.session_state.word_info.get(lemma, {})

        if info and "ko_meanings" in info:
            st.markdown(f"### **{current_token}**")
            st.markdown(f"**기본형 (Lemma):** *{lemma}*")
            st.divider()

            ko_meanings = info.get("ko_meanings", [])
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
                if ko_meanings and ko_meanings[0] == "JSON 파싱 오류":
                     st.error("Gemini API 정보 오류.")
                elif ko_meanings and ko_meanings[0].startswith(f"'{current_token}'의 API 키 없음"):
                     st.warning("API 키가 설정되지 않아 예문을 불러올 수 없습니다.")
                else:
                    st.info("예문 정보가 없습니다.")
        else:
            st.warning("단어 정보를 불러오는 중이거나 오류가 발생했습니다.")
            
    else:
        st.info("왼쪽 텍스트에서 단어를 클릭하고 복사(Ctrl+C)하여 위 검색창을 이용해주세요.")

# ---------------------- 5. 하단: 누적 목록 + CSV ----------------------
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
            if info.get("ko_meanings") and info["ko_meanings"][0] != "JSON 파싱 오류":
                short = "; ".join(info["ko_meanings"][:2])
                rows.append({"기본형": lemma, "대표 뜻": short})
                processed_lemmas.add(lemma)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSV로 저장", csv_bytes, "russian_words.csv", "text/csv")
    else:
        st.info("선택된 단어의 정보가 로드 중이거나, 표시할 정보가 없습니다.")
