import os
import uuid
import hashlib
import threading
import re
import json
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Text
)
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///./media_pipeline.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# ============================================================
# DATABASE MODEL
# ============================================================

class ImageJob(Base):

    __tablename__ = "image_jobs"

    id = Column(
        String,
        primary_key=True
    )

    original_filename = Column(
        String,
        nullable=False
    )

    file_path = Column(
        String,
        nullable=False
    )

    content_type = Column(
        String
    )

    file_size = Column(
        String
    )

    status = Column(
        String,
        default="pending"
    )

    result = Column(
        Text,
        nullable=True
    )

    error_message = Column(
        Text,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )


Base.metadata.create_all(
    bind=engine
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Intelligent Media Processing Pipeline",

    description=(
        "Asynchronous image analysis backend"
    ),

    version="1.0.0"
)


# ============================================================
# FILE HASHING
# ============================================================

def calculate_sha256(
    file_path: str
):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while chunk := file.read(
            8192
        ):

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def calculate_perceptual_hash(
    file_path: str
):

    image = Image.open(
        file_path
    )

    return str(
        imagehash.phash(
            image
        )
    )


# ============================================================
# BLUR DETECTION
# ============================================================

def detect_blur(
    image
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    is_blurry = variance < 100

    confidence = min(
        abs(
            variance - 100
        ) / 100,
        1.0
    )

    return {

        "detected": bool(
            is_blurry
        ),

        "laplacian_variance": round(
            float(
                variance
            ),
            2
        ),

        "threshold": 100,

        "confidence": round(
            confidence,
            2
        )

    }


# ============================================================
# BRIGHTNESS ANALYSIS
# ============================================================

def analyze_brightness(
    image
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mean_brightness = float(
        np.mean(
            gray
        )
    )

    if mean_brightness < 50:

        issue = "very_low_light"

    elif mean_brightness < 90:

        issue = "low_light"

    elif mean_brightness > 230:

        issue = "overexposed"

    else:

        issue = "normal"

    return {

        "issue": issue,

        "mean_brightness": round(
            mean_brightness,
            2
        ),

        "confidence": 0.85

    }


# ============================================================
# DIMENSION VALIDATION
# ============================================================

def analyze_dimensions(
    image
):

    height, width = image.shape[:2]

    valid = (

        width >= 640

        and

        height >= 480

    )

    return {

        "width": width,

        "height": height,

        "valid": valid,

        "issue": (

            None

            if valid

            else "low_resolution"

        )

    }


# ============================================================
# VEHICLE NUMBER VALIDATION
# ============================================================

def validate_vehicle_number(
    text: str
):

    if not text:

        return {

            "text": None,

            "valid_format": False,

            "reason": (
                "No vehicle number detected"
            )

        }

    normalized = re.sub(

        r"[^A-Z0-9]",

        "",

        text.upper()

    )

    pattern = (

        r"^[A-Z]{2}"

        r"[0-9]{1,2}"

        r"[A-Z]{1,3}"

        r"[0-9]{4}$"

    )

    is_valid = bool(

        re.match(

            pattern,

            normalized

        )

    )

    return {

        "text": normalized,

        "valid_format": is_valid,

        "reason": (

            None

            if is_valid

            else (

                "Does not match expected "
                "Indian vehicle number format"
            )

        )

    }


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def detect_duplicate(
    current_file_path: str,
    current_phash: str
):

    db = SessionLocal()

    jobs = db.query(
        ImageJob
    ).filter(
        ImageJob.status == "completed"
    ).all()

    current_sha256 = calculate_sha256(
        current_file_path
    )

    exact_duplicate = False

    perceptual_duplicate = False

    matched_processing_id = None

    for job in jobs:

        if not job.file_path:

            continue

        if not os.path.exists(
            job.file_path
        ):

            continue

        # Exact duplicate check
        existing_sha256 = calculate_sha256(
            job.file_path
        )

        if existing_sha256 == current_sha256:

            exact_duplicate = True

            matched_processing_id = job.id

            break

        # Perceptual duplicate check
        try:

            existing_phash = calculate_perceptual_hash(
                job.file_path
            )

            hash_distance = imagehash.hex_to_hash(
                existing_phash
            ) - imagehash.hex_to_hash(
                current_phash
            )

            if hash_distance <= 5:

                perceptual_duplicate = True

                matched_processing_id = job.id

                break

        except Exception:

            continue

    db.close()

    return {

        "is_duplicate": (

            exact_duplicate

            or

            perceptual_duplicate

        ),

        "duplicate_type": (

            "exact"

            if exact_duplicate

            else (

                "perceptual"

                if perceptual_duplicate

                else None

            )

        ),

        "matched_processing_id": (
            matched_processing_id
        )

    }


# ============================================================
# SCREENSHOT HEURISTICS
# ============================================================

def analyze_screenshot(
    image
):

    height, width = image.shape[:2]

    aspect_ratio = width / height

    likelihood = 0.0

    signals = []

    if (

        aspect_ratio > 1.7

        or

        aspect_ratio < 0.6

    ):

        likelihood += 0.3

        signals.append(
            "unusual_aspect_ratio"
        )

    if width in [

        1080,

        1170,

        1284,

        1440

    ]:

        likelihood += 0.3

        signals.append(
            "common_screenshot_width"
        )

    return {

        "screenshot_likelihood": round(

            min(

                likelihood,

                1.0

            ),

            2

        ),

        "signals": signals

    }


# ============================================================
# COMPLETE IMAGE ANALYSIS
# ============================================================

def analyze_image(
    file_path: str
):

    image = cv2.imread(
        file_path
    )

    if image is None:

        raise ValueError(

            "Unable to read image file"

        )

    results = {}

    # 1. Blur Detection
    results["blur"] = detect_blur(
        image
    )

    # 2. Brightness Analysis
    results["brightness"] = analyze_brightness(
        image
    )

    # 3. Dimension Validation
    results["dimensions"] = analyze_dimensions(
        image
    )

    # 4. SHA-256 Hash
    sha256_hash = calculate_sha256(
        file_path
    )

    results["file_hash"] = sha256_hash

    # 5. Perceptual Hash
    perceptual_hash = calculate_perceptual_hash(
        file_path
    )

    results["perceptual_hash"] = perceptual_hash

    # 6. Duplicate Detection
    results["duplicate"] = detect_duplicate(

        file_path,

        perceptual_hash

    )

    # 7. OCR Placeholder
    #
    # OCR can be connected here later.
    # The result remains structured.

    results["vehicle_number"] = validate_vehicle_number(

        ""

    )

    # 8. Screenshot Heuristics
    results["screenshot_analysis"] = analyze_screenshot(

        image

    )

    return results


# ============================================================
# ASYNCHRONOUS PROCESSING
# ============================================================

def process_image_async(
    job_id: str
):

    db = SessionLocal()

    try:

        job = db.query(
            ImageJob
        ).filter(
            ImageJob.id == job_id
        ).first()

        if not job:

            return

        # Update status
        job.status = "processing"

        job.updated_at = datetime.utcnow()

        db.commit()

        # Analyze image
        results = analyze_image(

            job.file_path

        )

        # Store proper JSON
        job.result = json.dumps(

            results

        )

        job.status = "completed"

        job.updated_at = datetime.utcnow()

        db.commit()

    except Exception as error:

        job = db.query(

            ImageJob

        ).filter(

            ImageJob.id == job_id

        ).first()

        if job:

            job.status = "failed"

            job.error_message = str(

                error

            )

            job.updated_at = datetime.utcnow()

            db.commit()

    finally:

        db.close()


# ============================================================
# API 1: UPLOAD IMAGE
# ============================================================

@app.post(
    "/api/v1/images"
)
async def upload_image(

    file: UploadFile = File(...)

):

    allowed_types = [

        "image/jpeg",

        "image/png",

        "image/jpg",

        "image/webp"

    ]

    if file.content_type not in allowed_types:

        raise HTTPException(

            status_code=400,

            detail=(

                "Only image files are supported"

            )

        )

    processing_id = str(

        uuid.uuid4()

    )

    extension = Path(

        file.filename

    ).suffix.lower()

    file_path = UPLOAD_DIR / (

        f"{processing_id}"

        f"{extension}"

    )

    content = await file.read()

    with open(

        file_path,

        "wb"

    ) as output_file:

        output_file.write(

            content

        )

    db = SessionLocal()

    job = ImageJob(

        id=processing_id,

        original_filename=file.filename,

        file_path=str(file_path),

        content_type=file.content_type,

        file_size=str(

            len(content)

        ),

        status="pending"

    )

    db.add(

        job

    )

    db.commit()

    db.close()

    # Start asynchronous processing
    thread = threading.Thread(

        target=process_image_async,

        args=(processing_id,),

        daemon=True

    )

    thread.start()

    return {

        "processing_id": processing_id,

        "status": "pending",

        "message": (

            "Image uploaded successfully "

            "and queued for processing"

        )

    }


# ============================================================
# API 2: GET STATUS
# ============================================================

@app.get(

    "/api/v1/images/{processing_id}/status"

)

def get_status(

    processing_id: str

):

    db = SessionLocal()

    job = db.query(

        ImageJob

    ).filter(

        ImageJob.id == processing_id

    ).first()

    db.close()

    if not job:

        raise HTTPException(

            status_code=404,

            detail=(

                "Processing ID not found"

            )

        )

    return {

        "processing_id": processing_id,

        "status": job.status,

        "error": job.error_message

    }


# ============================================================
# API 3: GET RESULTS
# ============================================================

@app.get(

    "/api/v1/images/{processing_id}/results"

)

def get_results(

    processing_id: str

):

    db = SessionLocal()

    job = db.query(

        ImageJob

    ).filter(

        ImageJob.id == processing_id

    ).first()

    db.close()

    if not job:

        raise HTTPException(

            status_code=404,

            detail=(

                "Processing ID not found"

            )

        )

    if job.status == "pending":

        return {

            "processing_id": processing_id,

            "status": "pending",

            "message": (

                "Image is waiting for processing"

            )

        }

    if job.status == "processing":

        return {

            "processing_id": processing_id,

            "status": "processing",

            "message": (

                "Image is currently being analyzed"

            )

        }

    if job.status == "failed":

        return {

            "processing_id": processing_id,

            "status": "failed",

            "error": job.error_message

        }

    # Convert JSON string back into JSON object
    analysis = json.loads(

        job.result

    )

    return {

        "processing_id": processing_id,

        "status": job.status,

        "analysis": analysis

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")

def health_check():

    return {

        "service": (

            "Intelligent Media Processing Pipeline"

        ),

        "status": "running",

        "version": "1.0.0"

    }