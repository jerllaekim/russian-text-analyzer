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
if "manual_search_word" not in st.session_state:
    st.session_state.manual_search_word = ""

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
    
    # --- [JSON 파싱 로직 강화] ---
    try:
        # 1. Markdown 코드 블록 제거
        if text.startswith("```"):
            text = text.strip("`")
            lines = text.splitlines()
            if lines and lines[0].lower().startswith("json"):
                text = "\n".join(lines[1:])
            elif lines:
                 text = "\n".join(lines)
                 
        # 2. JSON 문자열의 시작(첫 '{')과 끝(마지막 '}') 인덱스 찾기
        start_index = text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_text = text[start_index : end_index + 1]
        else:
            json_text = text
            
        data = json.loads(json_text)
        
        # 3. 예문 개수 제한 로직
        if 'examples' in data and len(data['examples']) > 2:
            data['examples'] = data['examples'][:2]
        return data
    
    except json.JSONDecodeError:
        st.error(f"Gemini 응답을 JSON으로 디코딩하는 데 실패했습니다. 원본 텍스트 시작: {text[:100]}...")
        return {"ko_meanings": ["JSON 파싱 오류"], "examples": []}

# ---------------------- 2. 전역 스타일 및 JavaScript 정의 ----------------------

# JavaScript: 단어 클릭 시 검색창에 텍스트를 채우고, 이벤트(input)를 발생시켜 Streamlit의 재실행을 유도합니다.
# 이 함수가 '자동 검색'을 구현하는 핵심이지만 불안정할 수 있습니다.
st.markdown("""
<script>
    function setManualSearchWordAndRerun(word) {
        // 1. '단어 직접 입력' 필드를 찾습니다. (ARIA-LABEL 기반)
        const inputField = document.querySelector('[aria-label="단어 직접 입력"]');
        if (inputField) {
            // 2. 값을 설정합니다.
            inputField.value = word;
            
            // 3. 'input' 이벤트를 강제 발생시켜 Streamlit에게 값이 변경되었음을 알립니다.
            //    이것이 Python 세션 상태를 업데이트하고 페이지를 재실행(RERUN)하도록 유도합니다.
            const event = new Event('input', { bubbles: true });
            inputField.dispatchEvent(event);
        } else {
             // 4. Input 필드를 찾지 못하면 복사만 합니다.
             navigator.clipboard.writeText(word);
             alert(`죄송합니다. 자동 검색이 작동하지 않아 '${word}'를 클립보드에 복사했습니다. 검색창에 붙여넣어 주세요.`);
        }
    }
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* (CSS 스타일은 이전과 동일하게 유지되어 파란색 글씨와 구두점 배치를 보장합니다.) */
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

# 검색 입력 필드 (key="manual_search_word"로 세션 상태에 바인딩)
manual_input = st.text_input("단어 직접 입력", key="manual_search_word")

# 검색 입력 처리 로직
if manual_input:
    # 1. 검색된 단어를 선택 목록에 추가 (파란색 글씨 유지를 위함)
    if manual_input not in st.session_state.selected_words:
        st.session_state.selected_words.append(manual_input)
    
    # 2. 상세 정보 영역에 표시될 단어 업데이트
    st.session_state.clicked_word = manual_input
    
    # ************** 검색 상세 정보 표시 **************
    lemma = lemmatize_ru(manual_input)
    st.markdown(f"**입력 단어:** **{manual_input}**")
    st.markdown(f"**기본형(lemma):** *{lemma}*")

    try:
        info = fetch_from_gemini(manual_input, lemma)
        
        # *** FIX for Issue 1: 검색된 단어의 정보를 세션 상태에 저장하여 하단 목록에 추가되도록 함 ***
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
            
            # 파란색 글씨 유지: 선택된 단어에 클래스를 직접 삽입
            if is_selected:
                css += " word-selected"
            
            # onclick: JavaScript 함수 호출 (클릭 시 자동 검색 시도)
            html_all.append(
                f'<span class="{css}" onclick="setManualSearchWordAndRerun(\'{tok}\');">'
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
                if ko_meanings and ko_meanings[0].startswith("JSON 파싱 오류"):
                     st.error("Gemini API에서 예상치 못한 형식이 반환되어 정보 표시 오류가 발생했습니다.")
                elif ko_meanings and ko_meanings[0].startswith(f"'{current_token}'의 API 키 없음"):
                     st.warning("API 키가 설정되지 않아 예문을 불러올 수 없습니다.")
                else:
                    st.info("예문 정보가 없습니다.")
        else:
            st.warning("단어 정보를 불러오는 중이거나 오류가 발생했습니다.")
            
    else:
        st.info("왼쪽 텍스트에서 단어를 클릭하여 자동 검색을 시도하거나, 위 검색창을 이용해주세요.")

# ---------------------- 5. 하단: 누적 목록 + CSV ----------------------
st.divider()
st.subheader("📝 선택한 단어 모음 (기본형 기준)")

selected = st.session_state.selected_words
word_info = st.session_state.word_info

if word_info:
    rows = []
    processed_lemmas = set()
    
    # 선택된 단어 목록을 순회하며, 해당 단어의 기본형 정보가 로드되었는지 확인
    for tok in selected:
        lemma = lemmatize_ru(tok)
        if lemma not in processed_lemmas and lemma in word_info:
            info = word_info[lemma]
            # JSON 파싱 오류 단어는 제외
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
