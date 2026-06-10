"""
Backend proxy pour l'application clavier IA (Android).

Expose /api/correct qui reçoit le texte tapé par l'utilisateur et
retourne les corrections grammaire/orthographe/conjugaison fournies
par Claude (claude-sonnet-4-20250514) via la clé Emergent Universal.

La clé reste côté serveur : elle n'est jamais exposée dans l'APK.
"""

from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import re
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import List

from emergentintegrations.llm.chat import LlmChat, UserMessage


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# --- Configuration -----------------------------------------------------------

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
EMERGENT_LLM_KEY = os.environ["EMERGENT_LLM_KEY"]

# Modèle demandé dans le cahier des charges. La clé Emergent route automatiquement
# vers l'API Anthropic. Si la version exacte n'est plus disponible, on peut
# basculer sur claude-sonnet-4-5-20250929 / claude-sonnet-4-6.
CLAUDE_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = (
    "Tu es un assistant de correction linguistique français. "
    "Analyse le texte fourni et retourne UNIQUEMENT un JSON avec ce format:\n"
    "{\n"
    '  "corrections": [\n'
    "    {\n"
    '      "original": "texte original",\n'
    '      "corrected": "texte corrigé",\n'
    '      "explanation": "explication courte en français"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Si le texte est correct, retourne corrections: [].\n"
    "Sois précis, pédagogique et bienveillant. "
    "Ne retourne JAMAIS de texte hors du JSON."
)

# --- App & DB ----------------------------------------------------------------

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI(title="Clavier IA - Backend de correction")
api_router = APIRouter(prefix="/api")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Modèles Pydantic --------------------------------------------------------


class CorrectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    device_id: str = Field(..., min_length=1, max_length=128)
    is_premium: bool = False


class Correction(BaseModel):
    original: str
    corrected: str
    explanation: str


class CorrectResponse(BaseModel):
    corrections: List[Correction]
    remaining_today: int
    is_premium: bool
    limit_reached: bool = False


class QuotaResponse(BaseModel):
    remaining_today: int
    used_today: int
    daily_limit: int
    is_premium: bool


class PremiumUpdate(BaseModel):
    device_id: str
    is_premium: bool


# --- Helpers quota -----------------------------------------------------------

DAILY_LIMIT = 50


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def _get_usage(device_id: str) -> dict:
    doc = await db.keyboard_usage.find_one(
        {"device_id": device_id}, {"_id": 0}
    )
    today = _today_str()
    if not doc:
        doc = {
            "device_id": device_id,
            "date": today,
            "count": 0,
            "is_premium": False,
        }
        await db.keyboard_usage.insert_one(dict(doc))
        return doc
    if doc.get("date") != today:
        # Réinitialisation quotidienne
        doc["date"] = today
        doc["count"] = 0
        await db.keyboard_usage.update_one(
            {"device_id": device_id},
            {"$set": {"date": today, "count": 0}},
        )
    return doc


async def _increment_usage(device_id: str) -> int:
    today = _today_str()
    await db.keyboard_usage.update_one(
        {"device_id": device_id},
        {"$set": {"date": today}, "$inc": {"count": 1}},
        upsert=True,
    )
    doc = await db.keyboard_usage.find_one(
        {"device_id": device_id}, {"_id": 0, "count": 1}
    )
    return int(doc.get("count", 0))


# --- Helpers Claude ----------------------------------------------------------


def _extract_json(raw: str) -> dict:
    """Extrait un objet JSON de la réponse Claude (au cas où il y aurait du texte autour)."""
    raw = raw.strip()
    if raw.startswith("```"):
        # Retire les fences markdown ```json ... ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Tentative : trouver le premier { et le dernier }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"corrections": []}


async def _call_claude(text: str) -> List[Correction]:
    session_id = f"keyboard-{uuid.uuid4()}"
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=SYSTEM_PROMPT,
    ).with_model("anthropic", CLAUDE_MODEL)

    user_message = UserMessage(text=text)
    response = await chat.send_message(user_message)

    payload = _extract_json(response if isinstance(response, str) else str(response))
    raw_corrections = payload.get("corrections", []) or []

    corrections: List[Correction] = []
    for item in raw_corrections:
        if not isinstance(item, dict):
            continue
        original = str(item.get("original", "")).strip()
        corrected = str(item.get("corrected", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        if original and corrected and original != corrected:
            corrections.append(
                Correction(
                    original=original,
                    corrected=corrected,
                    explanation=explanation or "Correction proposée.",
                )
            )
    return corrections


# --- Routes ------------------------------------------------------------------


@api_router.get("/")
async def root():
    return {"message": "Clavier IA - backend opérationnel", "model": CLAUDE_MODEL}


@api_router.get("/download/android-zip")
async def download_android_zip():
    """Route temporaire pour télécharger le code source Android. À retirer après usage."""
    zip_path = Path("/app/clavier-ia-android.zip")
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Archive introuvable.")
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename="clavier-ia-android.zip",
    )


@api_router.get("/quota/{device_id}", response_model=QuotaResponse)
async def get_quota(device_id: str):
    usage = await _get_usage(device_id)
    is_premium = bool(usage.get("is_premium", False))
    used = int(usage.get("count", 0))
    remaining = 10_000 if is_premium else max(0, DAILY_LIMIT - used)
    return QuotaResponse(
        remaining_today=remaining,
        used_today=used,
        daily_limit=DAILY_LIMIT,
        is_premium=is_premium,
    )


@api_router.post("/premium")
async def set_premium(payload: PremiumUpdate):
    await db.keyboard_usage.update_one(
        {"device_id": payload.device_id},
        {"$set": {"is_premium": payload.is_premium}},
        upsert=True,
    )
    return {"ok": True, "is_premium": payload.is_premium}


@api_router.post("/correct", response_model=CorrectResponse)
async def correct(payload: CorrectRequest):
    usage = await _get_usage(payload.device_id)
    is_premium = bool(payload.is_premium or usage.get("is_premium", False))
    used = int(usage.get("count", 0))

    if not is_premium and used >= DAILY_LIMIT:
        return CorrectResponse(
            corrections=[],
            remaining_today=0,
            is_premium=False,
            limit_reached=True,
        )

    try:
        corrections = await _call_claude(payload.text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erreur appel Claude")
        raise HTTPException(
            status_code=502,
            detail=f"Erreur lors de l'appel au service de correction : {exc}",
        )

    new_count = await _increment_usage(payload.device_id)
    remaining = 10_000 if is_premium else max(0, DAILY_LIMIT - new_count)

    return CorrectResponse(
        corrections=corrections,
        remaining_today=remaining,
        is_premium=is_premium,
        limit_reached=False,
    )


# --- Branchement -------------------------------------------------------------

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
     client.close()
