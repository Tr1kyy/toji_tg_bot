# bot_merged.py
import os
import re
import json
import logging
import asyncio
import shutil
from pathlib import Path
from threading import Thread

import httpx
import requests
import pytesseract
import cv2
from PIL import Image
from pydub import AudioSegment
from flask import Flask

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------------
# Config from env (no hardcoded keys)
# ---------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-r1")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

YANDEX_TTS_API_KEY = os.environ.get("YANDEX_TTS_API_KEY", "")
YANDEX_STT_API_KEY = os.environ.get("YANDEX_STT_API_KEY", "")
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")

# Optional: allow overriding tesseract path via env
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Bot behavior
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are Toji Fushiguro from the anime 'Jujutsu Kaisen'. Respond as this character would: laconic, confident, and pragmatic. Stay in character, use short, punchy phrases, and refer to your abilities and past experiences. Use occasional Russian swearing words and a bit of rough flirting. When replying, use Russian language only. Avoid empty answers and make sure every reply contains at least one meaningful phrase."
)

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1200"))

# ---------------------------
# Data directories (user files)
# ---------------------------
BASE_DIR = Path.cwd()
VOICE_DIR = BASE_DIR / "Voice"
IMAGES_DIR = BASE_DIR / "Images"
HISTORY_DIR = BASE_DIR / "History"
TMP_DIR = BASE_DIR / "tmp"  # временные файлы

# Ensure directories exist
for p in (VOICE_DIR, IMAGES_DIR, HISTORY_DIR, TMP_DIR):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.exception("Не удалось создать директорию %s: %s", p, e)

# ---------------------------
# In-memory user state (simple)
# ---------------------------
user_histories = {}        # {user_id: [messages...]}
user_message_counts = {}   # {user_id: int}
user_locks = {}            # {user_id: asyncio.Lock()}

def get_lock(user_id: int):
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    return user_locks[user_id]

# ---------------------------
# Utils
# ---------------------------
def process_content(content: str) -> str:
    return content.replace('<think>', '').replace('</think>', '')

def save_user_history(user_id: int, history: list) -> None:
    """Сохраняем историю в JSON в папке History"""
    filename = HISTORY_DIR / f"history_{user_id}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Не удалось сохранить историю: %s", e)

def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'[\*<>]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------------------
# OpenRouter (DeepSeek) streaming - async
# ---------------------------
async def deepseek_chat_stream(user_id: int, prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        return "Ошибка: сервер не настроен (нет API ключа)."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    lock = get_lock(user_id)
    async with lock:
        user_history = user_histories.get(user_id, [])
        user_history.append({"role": "user", "content": prompt})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + user_history
        user_histories[user_id] = user_history[-20:]

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "stream": True
    }

    full = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", API_URL, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    logger.error("OpenRouter API Error %s: %s", resp.status_code, text)
                    return f"❌ Ошибка API: {resp.status_code}"
                async for raw_line in resp.aiter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        data_part = line[len("data:"):].strip()
                    else:
                        data_part = line
                    if not data_part:
                        continue
                    if data_part == "[DONE]":
                        break
                    try:
                        data = json.loads(data_part)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        chunk = delta.get("content", "")
                        if chunk:
                            cleaned = process_content(chunk)
                            full.append(cleaned)
                    except json.JSONDecodeError:
                        continue
    except httpx.RequestError as e:
        logger.exception("Request error to OpenRouter: %s", e)
        return "❌ Ошибка запроса к модели: попробуйте позже."

    combined = ''.join(full).strip()
    async with lock:
        user_histories[user_id].append({"role": "assistant", "content": combined})
        save_user_history(user_id, user_histories[user_id])
    return combined or "Извини, я не смог найти, что сказать. Попробуй задать другой вопрос."

# ---------------------------
# Yandex TTS / STT helpers
# ---------------------------
def text_to_speech(text: str, user_id: int) -> str:
    text = clean_text_for_tts(text)
    if not YANDEX_TTS_API_KEY:
        logger.warning("YANDEX_TTS_API_KEY not set, skipping TTS")
        return ""
    url = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    headers = {"Authorization": f"Api-Key {YANDEX_TTS_API_KEY}"}
    data = {
        "text": text,
        "lang": "ru-RU",
        "voice": "ermil",
        "speed": "0.9",
        "folderId": YANDEX_FOLDER_ID,
        "format": "mp3"
    }
    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)
        if response.status_code != 200:
            logger.error("Yandex TTS API Error: %s - %s", response.status_code, response.text)
            return ""
        filename = VOICE_DIR / f"tts_{user_id}.mp3"
        with open(filename, 'wb') as f:
            f.write(response.content)
        return str(filename)
    except Exception as e:
        logger.exception("Ошибка Yandex TTS: %s", e)
        return ""

def recognize_speech_from_file(file_path: str) -> str:
    if not YANDEX_STT_API_KEY:
        logger.warning("YANDEX_STT_API_KEY not set, skipping STT")
        return ""
    try:
        audio = AudioSegment.from_file(file_path)
        ogg_file = Path(file_path).with_suffix('.ogg')
        audio.export(ogg_file, format='ogg', codec='libopus')
        with open(ogg_file, 'rb') as f:
            audio_data = f.read()
        url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
        headers = {"Authorization": f"Api-Key {YANDEX_STT_API_KEY}", "Content-Type": "application/octet-stream"}
        params = {"folderId": YANDEX_FOLDER_ID, "lang": "ru-RU"}
        response = requests.post(url, headers=headers, params=params, data=audio_data, timeout=60)
        if response.status_code != 200:
            logger.error("Yandex STT API Error: %s - %s", response.status_code, response.text)
            return ""
        result = response.json()
        text = result.get('result', '') or (result.get('chunks') and result.get('chunks')[0].get('alternatives', [{}])[0].get('text', ''))
        # cleanup exported ogg
        try:
            if ogg_file.exists():
                ogg_file.unlink()
        except Exception:
            pass
        return (text or "").strip()
    except Exception as e:
        logger.exception("Ошибка при распознавании речи: %s", e)
        return ""

# ---------------------------
# Telegram handlers
# ---------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет! Я чат-бот в образе Toji Fushiguro. Задай мне вопрос.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    # PHOTO
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
        # save into Images folder
        file_path = IMAGES_DIR / f"image_{user_id}_{photo_file.file_unique_id}.jpg"
        await photo_file.download_to_drive(custom_path=str(file_path))
        logger.info("Получено изображение: %s", file_path)
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError("opencv не смог прочитать изображение")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_image = Image.fromarray(thresh)
            custom_config = r'--oem 3 --psm 6'
            text_from_image = pytesseract.image_to_string(processed_image, lang='rus', config=custom_config).strip()
            logger.info("OCR: %s", text_from_image)
            if not text_from_image:
                await update.message.reply_text("Не удалось извлечь текст с изображения.")
                return
            full_response = await deepseek_chat_stream(user_id, f"Проанализируй изображение: {text_from_image}")
            if user_message_counts.get(user_id, 0) % 3 == 2:
                tts_filename = text_to_speech(full_response, user_id)
                if tts_filename:
                    try:
                        with open(tts_filename, 'rb') as audio_file:
                            await update.message.reply_audio(audio=audio_file, filename=os.path.basename(tts_filename))
                    finally:
                        try:
                            Path(tts_filename).unlink()
                        except Exception:
                            pass
                else:
                    await update.message.reply_text(full_response)
            else:
                await update.message.reply_text(full_response)
            user_message_counts[user_id] = user_message_counts.get(user_id, 0) + 1
        except Exception as e:
            logger.exception("Ошибка при обработке изображения: %s", e)
            await update.message.reply_text("Ошибка при анализе изображения.")
        return

    # VOICE (voice = OGG Opus voice note in telegram)
    if update.message.voice:
        voice_file = await update.message.voice.get_file()
        file_path = VOICE_DIR / f"voice_{user_id}_{voice_file.file_unique_id}.oga"
        await voice_file.download_to_drive(custom_path=str(file_path))
        logger.info("Получен voice файл: %s", file_path)
        try:
            recognized_text = await asyncio.get_event_loop().run_in_executor(None, recognize_speech_from_file, str(file_path))
            if not recognized_text:
                await update.message.reply_text("Извини, я не смог распознать твой голос.")
                return
            full_response = await deepseek_chat_stream(user_id, recognized_text)
            if user_message_counts.get(user_id, 0) % 3 == 2:
                tts_filename = text_to_speech(full_response, user_id)
                if tts_filename:
                    try:
                        with open(tts_filename, 'rb') as audio_file:
                            await update.message.reply_audio(audio=audio_file, filename=os.path.basename(tts_filename))
                    finally:
                        try:
                            Path(tts_filename).unlink()
                        except Exception:
                            pass
                else:
                    await update.message.reply_text(full_response)
            else:
                await update.message.reply_text(full_response)
            # optional: keep voice files, or delete if you prefer
            # Path(file_path).unlink(missing_ok=True)  # uncomment to delete
            user_message_counts[user_id] = user_message_counts.get(user_id, 0) + 1
        except Exception as e:
            logger.exception("Ошибка при обработке voice: %s", e)
            await update.message.reply_text("Ошибка при распознавании/обработке голосового сообщения.")
        return

    # TEXT
    if update.message.text:
        user_text = update.message.text
        try:
            full_response = await deepseek_chat_stream(user_id, user_text)
            if user_message_counts.get(user_id, 0) % 3 == 2:
                tts_filename = text_to_speech(full_response, user_id)
                if tts_filename:
                    try:
                        with open(tts_filename, 'rb') as audio_file:
                            await update.message.reply_audio(audio=audio_file, filename=os.path.basename(tts_filename))
                    finally:
                        try:
                            Path(tts_filename).unlink()
                        except Exception:
                            pass
                else:
                    await update.message.reply_text(full_response)
            else:
                await update.message.reply_text(full_response)
            user_message_counts[user_id] = user_message_counts.get(user_id, 0) + 1
        except Exception as e:
            logger.exception("Ошибка при обработке текста: %s", e)
            await update.message.reply_text("Ошибка при генерации ответа.")
        return

# ---------------------------
# Health server (Flask)
# ---------------------------
def run_health_server():
    app = Flask("health")
    @app.route("/health")
    def health():
        return "OK", 200
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)

def start_health_server_in_thread():
    t = Thread(target=run_health_server, daemon=True)
    t.start()
    logger.info("Health server started on port %s", os.environ.get("PORT", "8000"))

# ---------------------------
# Dependency check helper
# ---------------------------
def check_deps():
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg не найден в PATH. Конвертация аудио может не работать.")
    try:
        if not Path(pytesseract.pytesseract.tesseract_cmd).exists():
            logger.warning("Tesseract не найден по пути: %s", pytesseract.pytesseract.tesseract_cmd)
    except Exception:
        logger.exception("Ошибка при проверке Tesseract")

# ---------------------------
# Main
# ---------------------------
def main() -> None:
    check_deps()
    start_health_server_in_thread()

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set. Exiting.")
        return

    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.VOICE, handle_message))
        logger.info("Запускаем Telegram-бота (polling)...")
        app.run_polling()
    except Exception as e:
        logger.exception("Fatal error in main: %s", e)

if __name__ == '__main__':
    main()
