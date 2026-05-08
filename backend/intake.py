"""
Smart Data Intake — CSV upload, text paste, auto-detect text column, dedup, preview.
"""
import io
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .auth import get_current_user
from .database import Dataset, FeedbackItem, User, get_db

router = APIRouter(prefix="/intake", tags=["intake"])


# ------------------------- helpers -------------------------

LIKELY_TEXT_COLUMNS = {
    "feedback", "comment", "comments", "review", "reviews", "message",
    "text", "content", "response", "answer", "note", "notes", "description",
    "ticket", "issue", "complaint", "transcript",
}

LIKELY_LABEL_COLUMNS = {
    "user", "username", "name", "email", "customer", "user_id", "id",
    "author", "submitted_by", "from",
}


def _pick_text_column(df: pd.DataFrame) -> str:
    """Pick the column most likely to contain feedback text."""
    # 1. Prefer columns with known names
    for col in df.columns:
        if str(col).strip().lower() in LIKELY_TEXT_COLUMNS:
            return col
    # 2. Fall back to the column with the longest average string length
    best_col, best_len = None, -1.0
    for col in df.columns:
        try:
            series = df[col].dropna().astype(str)
            if series.empty:
                continue
            avg = series.str.len().mean()
            if avg > best_len:
                best_len = avg
                best_col = col
        except Exception:
            continue
    if best_col is None:
        raise HTTPException(status_code=400, detail="Could not detect a text column in the CSV.")
    return best_col


def _pick_label_column(df: pd.DataFrame, text_col: str) -> Optional[str]:
    for col in df.columns:
        if col == text_col:
            continue
        if str(col).strip().lower() in LIKELY_LABEL_COLUMNS:
            return col
    return None


def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # Collapse whitespace
    return " ".join(s.split())


# ------------------------- request/response models -------------------------

class PreviewResponse(BaseModel):
    columns: List[str]
    detected_text_column: str
    detected_label_column: Optional[str]
    preview_rows: List[dict]
    total_rows: int
    cleaned_count: int
    duplicate_count: int
    upload_token: str  # cached preview ID


class TextPasteRequest(BaseModel):
    name: str
    text: str


class SaveDatasetRequest(BaseModel):
    upload_token: str
    name: str
    text_column: str
    label_column: Optional[str] = None


class DatasetSummary(BaseModel):
    id: int
    name: str
    source_type: str
    row_count: int
    created_at: str


class FeedbackItemResponse(BaseModel):
    id: int
    content: str
    user_label: Optional[str]


# Cheap in-memory cache for upload previews (keyed by token)
# In production, use Redis or temp files — fine for a hackathon/MVP.
_PREVIEW_CACHE: dict = {}


# ------------------------- endpoints -------------------------

@router.post("/upload-csv", response_model=PreviewResponse)
async def upload_csv(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    if not file.filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a .csv, .tsv, or .txt file")

    raw = await file.read()
    # Strip a UTF-8 BOM if present (Excel adds one when saving as "CSV UTF-8")
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]

    # Try a series of (encoding, separator) combinations. latin-1 is last
    # because it never raises a decode error (it accepts any byte), so it's
    # our guaranteed fallback for unusual encodings.
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    separators = [",", "\t", ";"]
    df = None
    last_err: Exception | None = None
    for enc in encodings:
        for sep in separators:
            try:
                candidate = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
                # Reject results that didn't actually find separators (1 column with junk)
                if candidate.shape[1] >= 1 and len(candidate) > 0:
                    df = candidate
                    break
            except Exception as e:
                last_err = e
                continue
        if df is not None:
            break
    if df is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV. Try saving it as 'CSV UTF-8' from Excel. ({last_err})",
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV is empty.")

    text_col = _pick_text_column(df)
    label_col = _pick_label_column(df, text_col)

    # Clean: drop empty text, dedupe on cleaned text
    original_count = len(df)
    df[text_col] = df[text_col].astype(str).map(_clean_text)
    df = df[df[text_col].str.len() > 0]
    cleaned_count = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first")
    deduped_count = len(df)
    duplicate_count = cleaned_count - deduped_count

    # Cache the cleaned df under a token (use file content hash to be deterministic-ish)
    import uuid
    token = uuid.uuid4().hex
    _PREVIEW_CACHE[token] = {
        "df": df,
        "user_id": current_user.id,
    }

    preview_rows = df.head(5).fillna("").astype(str).to_dict(orient="records")

    return PreviewResponse(
        columns=[str(c) for c in df.columns],
        detected_text_column=str(text_col),
        detected_label_column=str(label_col) if label_col else None,
        preview_rows=preview_rows,
        total_rows=deduped_count,
        cleaned_count=original_count - cleaned_count,
        duplicate_count=duplicate_count,
        upload_token=token,
    )


@router.post("/save", response_model=DatasetSummary)
def save_dataset(req: SaveDatasetRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    cached = _PREVIEW_CACHE.get(req.upload_token)
    if not cached or cached["user_id"] != current_user.id:
        raise HTTPException(status_code=400, detail="Upload token expired or invalid. Please re-upload.")

    df: pd.DataFrame = cached["df"]
    if req.text_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.text_column}' not in dataset")

    name = req.name.strip() or "Untitled dataset"
    dataset = Dataset(
        user_id=current_user.id,
        name=name,
        source_type="csv",
        row_count=len(df),
    )
    db.add(dataset)
    db.flush()  # get dataset.id

    items = []
    label_col = req.label_column if req.label_column and req.label_column in df.columns else None
    for _, row in df.iterrows():
        content = _clean_text(row[req.text_column])
        if not content:
            continue
        label = _clean_text(row[label_col]) if label_col else None
        # Stash any other columns as metadata
        meta = {}
        for col in df.columns:
            if col in (req.text_column, label_col):
                continue
            try:
                val = row[col]
                if pd.notna(val):
                    meta[str(col)] = str(val)
            except Exception:
                pass
        items.append(FeedbackItem(
            dataset_id=dataset.id,
            content=content,
            user_label=label,
            extra_metadata=meta or None,
        ))

    db.bulk_save_objects(items)
    dataset.row_count = len(items)
    db.commit()
    db.refresh(dataset)

    # free cache
    _PREVIEW_CACHE.pop(req.upload_token, None)

    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        source_type=dataset.source_type,
        row_count=dataset.row_count,
        created_at=dataset.created_at.isoformat(),
    )


@router.post("/paste", response_model=DatasetSummary)
def paste_text(req: TextPasteRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    raw = (req.text or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="No text provided")

    # Split on blank lines first; if that yields too few items, fall back to single newlines.
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    if len(blocks) < 2:
        blocks = [b.strip() for b in raw.split("\n") if b.strip()]

    # Dedup
    seen = set()
    unique_blocks = []
    for b in blocks:
        cleaned = _clean_text(b)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_blocks.append(cleaned)

    if not unique_blocks:
        raise HTTPException(status_code=400, detail="No usable feedback found in the pasted text")

    dataset = Dataset(
        user_id=current_user.id,
        name=req.name.strip() or "Pasted feedback",
        source_type="text",
        row_count=len(unique_blocks),
    )
    db.add(dataset)
    db.flush()

    items = [FeedbackItem(dataset_id=dataset.id, content=b) for b in unique_blocks]
    db.bulk_save_objects(items)
    db.commit()
    db.refresh(dataset)

    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        source_type=dataset.source_type,
        row_count=dataset.row_count,
        created_at=dataset.created_at.isoformat(),
    )


@router.get("/datasets", response_model=List[DatasetSummary])
def list_datasets(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Dataset).filter(Dataset.user_id == current_user.id).order_by(Dataset.created_at.desc()).all()
    return [
        DatasetSummary(
            id=d.id, name=d.name, source_type=d.source_type,
            row_count=d.row_count, created_at=d.created_at.isoformat(),
        )
        for d in rows
    ]


@router.get("/datasets/{dataset_id}/items", response_model=List[FeedbackItemResponse])
def list_items(dataset_id: int, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    items = db.query(FeedbackItem).filter(FeedbackItem.dataset_id == dataset_id).limit(limit).all()
    return [FeedbackItemResponse(id=i.id, content=i.content, user_label=i.user_label) for i in items]


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.user_id == current_user.id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(dataset)
    db.commit()
    return {"ok": True}
