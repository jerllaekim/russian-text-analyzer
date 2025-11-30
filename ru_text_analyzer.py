import streamlit as st
import re
import os
import json
import pandas as pd
from pymystem3 import Mystem
from google import genai
from google.cloud import vision 
import io
import urllib.parse 
import struct # WAV 헤더 생성을 위해 추가
from base64 import b64decode # Base64 디코딩을 위해 추가

# ---------------------- 0. 초기 설정 및 세션 상태 ----------------------

NEW_DEFAULT_TEXT = """МОЙ РАБОЧИЙ ДЕНЬ
(Рассказ японского банкира)
Разрешите представитьcя. Меня зовут Такеши Осада. Я работаю в банке «Сакура». Я живу недалеко от Токио, поэтому в рабочие дни я встаю в 5 часов утра, умываюсь, одеваюсь, завтракаю и иду на станцию. На станции я покупаю свежую газету. Я еду на работу на электричке. В электричке я обычно читаю или сплю. Дорога от дома до работы занимает 2 часа.
Банк начинает работать в 8 утра, а заканчивает в 8 вечера, то есть мой рабочий день продолжается 12 часов, включая 2 перерыва. Рабочий день для женщин, конечно, меньше.
В 8:30 обычно начинается собрание, где мы обсуждаем экономическую ситуацию, курс доллара, последние экономические новости, планируем работу на день. Потом я читаю документы, решаю важные вопросы, встречаюсь с клиентами, открываю им счёт в банке, даю им кредит, разговариваю по телефону и так далее.
После работы я возвращаюсь домой. Так как я очень устаю, дома я сразу ложусь спать."""

DEFAULT_TEST_TEXT = "Человек идёт по улице. Это тестовая строка. Хорошо. Я часто читаю эту книгу."


st.set_page_config(page_title="러시아어 텍스트 분석기", layout="wide")
st.title("러시아어 텍스트 분석기") 

# --- 세션 상태 초기화 ---
if "selected_words" not in st.session_state:
    st.session_state.selected_words = []
if "clicked_word" not in st.session_state:
    st.session_state.clicked_word = None
if "word_info" not in st.session_state:
    st.session_state.word_info = {}
if "current_search_query" not in st.session_state:
    st.session_state.current_search_query = ""
if "ocr_output_text" not in st.session_state:
    st.session_state.ocr_output_text = ""
if "input_text_area" not in st.session_state:
    st.session_state.input_text_area = DEFAULT_TEST_TEXT
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "last_processed_text" not in st.session_state:
    st.session_state.last_processed_text = "" 
if "last_processed_query" not in st.session_state:
    st.session_state.last_processed_query = ""
if "tts_audio" not in st.session_state: # TTS 오디오 저장
    st.session_state.tts_audio = None
if "tts_text_key" not in st.session_state: # TTS 오디오와 텍스트 일치 확인용
    st.session_state.tts_text_key = ""


mystem = Mystem()

# ---------------------- 품사 변환 딕셔너리 및 Mystem 함수 ----------------------
POS_MAP = {
    'S': '명사', 'V': '동사', 'A': '형용사', 'ADV': '부사', 'PR': '전치사',
    'CONJ': '접속사', 'INTJ': '감탄사', 'PART': '불변화사', 'NUM': '수사',
    'APRO': '대명사적 형용사', 'ANUM': '서수사', 'SPRO': '대명사',
}

@st.cache_data(show_spinner=False)
def lemmatize_ru(word: str) -> str:
    if ' ' in word.strip():
        return word.strip()
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        lemmas = mystem.lemmatize(word)
        return (lemmas[0] if lemmas else word).strip()
    return word

@st.cache_data(show_spinner=False)
def get_pos_ru(word: str) -> str:
    if ' ' in word.strip():
        return '관용구'
    if re.fullmatch(r'\w+', word, flags=re.UNICODE):
        analysis = mystem.analyze(word)
        if analysis and 'analysis' in analysis[0] and analysis[0]['analysis']:
            grammar_info = analysis[0]['analysis'][0]['gr']
            pos_abbr = grammar_info.split('=')[0].split(',')[0].strip()
            return POS_MAP.get(pos_abbr, '품사')
    return '품사'

# ---------------------- OCR 함수 ----------------------
@st.cache_data(show_spinner="이미지에서 텍스트 추출 중")
def detect_text_from_image(image_bytes):
    try:
        if st.secrets.get("GCP_SA_KEY"):
            with open("temp_sa_key.json", "w") as f:
                json.dump(st.secrets["GCP_SA_KEY"], f)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "temp_sa_key.json"
        elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            return "OCR API 키(GOOGLE_APPLICATION_CREDENTIALS)가 설정되지 않았습니다. Cloud Vision API 설정을 확인해주세요."

        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        texts = response.text
            
        if response.error.message:
            return f"Vision API 오류: {response.error.message}"
            
        return texts.split('\n', 1)[0] if texts else "이미지에서 텍스트를 찾을 수 없습니다."

    except Exception as e:
        return f"OCR 처리 중 오류 발생: {e}"


# ---------------------- TTS WAV 변환 함수 ----------------------

def pcm_to_wav(pcm_data: bytes, sample_rate: int) -> bytes:
    """Converts raw 16-bit PCM data to a WAV file format.
    Assumes 1 channel, 16-bit depth (2 bytes per sample).
    """
    channels = 1
    sample_width = 2
    
    # RIFF header
    chunk_id = b'RIFF'
    chunk_size = 36 + len(pcm_data)
    format_type = b'WAVE'

    # fmt sub-chunk
    sub_chunk1_id = b'fmt '
    sub_chunk1_size = 16  # PCM
    audio_format = 1  # PCM
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    bits_per_sample = 16

    # data sub-chunk
    sub_chunk2_id = b'data'
    sub_chunk2_size = len(pcm_data)

    wav_header = struct.pack(
        '<4sI4s'  # RIFF header
        '4sIHHIIHH'  # fmt sub-chunk
        '4sIHHIIHH'  # fmt sub-chunk
        '4sI',  # data sub-chunk
        chunk_id, chunk_size, format_type,
        sub_chunk1_id, sub_chunk1_size, audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        sub_chunk2_id, sub_chunk2_size
    )

    return wav_header + pcm_data

# ---------------------- 1. Gemini 연동 함수 (기존) ----------------------

def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    return genai.Client(api_key=api_key) if api_key else None

@st.cache_data(show_spinner=False)
def fetch_from_gemini(word, lemma, pos):
    client = get_gemini_client()
    if not client:
        return {"ko_meanings": [f"'{word}'의 API 키 없음 (GEMINI_API_KEY 설정 필요)"], "examples": []}
    
    SYSTEM_PROMPT = "너는 러시아어-한국어 학습 도우미이다. 러시아어 단어에 대해 간단한 한국어 뜻과 예문을 최대 두 개만 제공한다. 한국어 뜻을 제공할 때 격 정보, 문법 정보 등 불필요한 부가 정보는 절대 포함하지 않는다. 만약 동사(V)이면, 불완료상(imp)과 완료상(perf) 형태를 함께 제공해야 한다. 반드시 JSON만 출력한다."
    
    if pos == '동사':
        prompt = f"""{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}
{{ "ko_meanings": ["뜻1", "뜻2"], "aspect_pair": {{"imp": "불완료상 동사", "perf": "완료상 동사"}}, "examples": [ {{"ru": "예문1", "ko": "번역1"}}, {{"ru": "예문2", "ko": "번역2"}} ] }}
"""
    else:
        prompt = f"""{SYSTEM_PROMPT}
단어: {word}
기본형: {lemma}
{{ "ko_meanings": ["뜻1", "뜻2"], "examples": [ {{"ru": "예문1", "ko": "번역1"}}, {{"ru": "예문2", "ko": "번역2"}} ] }}
"""
    
    res = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    text = res.text.strip()
    
    try:
        # JSON 파싱 로직
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
        return {"ko_meanings": ["JSON 파싱 오류"], "examples": []}


# ---------------------- 2. TTS 함수 (신규) ----------------------

@st.cache_data(show_spinner="음성 파일 생성 중...")
def fetch_tts_audio(russian_text: str) -> bytes | str:
    client = get_gemini_client()
    if not client:
        return "Gemini API 키가 설정되지 않아 TTS를 수행할 수 없습니다."
    
    # 사용할 음성 설정 (Kore는 맑고 단단한 목소리)
    TTS_VOICE = "Kore" 

    # 텍스트 길이 제한 (TTS API 효율 및 비용 고려)
    if len(russian_text) > 500:
        russian_text = russian_text[:500] + "..."
    
    TTS_PROMPT = f"Say the following Russian text clearly and professionally:\n\n{russian_text}"

    payload = {
        "contents": [{
            "parts": [{"text": TTS_PROMPT}]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": TTS_VOICE}
                }
            }
        },
    }

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts", 
            contents=payload['contents'],
            config=payload['generationConfig']
        )

        audio_part = response.candidates[0].content.parts[0]
        base64_data = audio_part.inlineData.data
        mime
