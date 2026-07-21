import io
import uuid
import hashlib
import threading
import re
import json
import logging
import time

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import numpy as np
import easyocr

from PIL import Image, ExifTags

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Text,
    Integer
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATABASE_URL = (
    f"sqlite:///{BASE_DIR / 'media_pipeline.db'}"
)

MAX_FILE_SIZE = 10 * 1024 * 1024


ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}


ALLOWED_IMAGE_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP"
}


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 30
    }
)


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


Base = declarative_base()


# ============================================================
# IMAGE JOB TABLE
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
        String,
        nullable=False
    )

    file_size = Column(
        Integer,
        nullable=False
    )

    status = Column(
        String,
        default="pending",
        nullable=False
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
        default=datetime.utcnow,
        nullable=False
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


# ============================================================
# IMAGE HASH TABLE
# ============================================================

class ImageHash(Base):

    __tablename__ = "image_hashes"

    image_id = Column(
        String,
        primary_key=True
    )

    sha256 = Column(
        String,
        nullable=False,
        index=True
    )

    phash = Column(
        String,
        nullable=False
    )


Base.metadata.create_all(
    bind=engine
)


# ============================================================
# OCR MODEL
# ============================================================

logger.info(
    "Loading EasyOCR model..."
)

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False
)

logger.info(
    "EasyOCR model loaded successfully"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Intelligent Media Processing Pipeline",
    description=(
        "Asynchronous image analysis backend "
        "for vehicle image quality validation"
    ),
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ),
    name="static"
)


@app.get(
    "/",
    include_in_schema=False
)
def frontend():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Frontend file not found"
        )

    return FileResponse(
        str(index_file)
    )


# ============================================================
# SHA256 HASH
# ============================================================

def calculate_sha256(
    file_path: str
):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                8192
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def calculate_perceptual_hash(
    file_path: str
):

    with Image.open(
        file_path
    ) as image:

        return str(
            imagehash.phash(
                image.convert("RGB")
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

    threshold = 100.0

    is_blurry = (
        variance < threshold
    )

    if variance < threshold:

        confidence = min(
            (threshold - variance)
            / threshold,
            1.0
        )

    else:

        confidence = min(
            (variance - threshold)
            / threshold,
            1.0
        )

    return {

        "detected": bool(
            is_blurry
        ),

        "laplacian_variance": round(
            float(variance),
            2
        ),

        "threshold": threshold,

        "confidence": round(
            float(confidence),
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
        np.mean(gray)
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
# IMAGE DIMENSION VALIDATION
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

        "width": int(width),

        "height": int(height),

        "valid": bool(valid),

        "issue": (

            None

            if valid

            else

            "low_resolution"

        )

    }


# ============================================================
# INDIAN STATE / UNION TERRITORY CODES
# ============================================================

INDIAN_STATE_CODES = {

    "AP", "AR", "AS", "BR",
    "CG", "GA", "GJ", "HR",
    "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "PB",
    "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB",

    "AN", "CH", "DN", "DD",
    "DL", "JK", "LA", "LD",
    "PY"

}


# ============================================================
# VEHICLE NUMBER VALIDATION
# ============================================================

def is_valid_vehicle_number(
    text: str
):

    if not text:

        return False

    text = normalize_ocr_text(
        text
    )

    pattern = (
        r"^[A-Z]{2}"
        r"[0-9]{2}"
        r"[A-Z]{0,3}"
        r"[0-9]{4}$"
    )

    if not re.fullmatch(
        pattern,
        text
    ):

        return False

    state_code = text[:2]

    if state_code not in INDIAN_STATE_CODES:

        return False

    return True


# ============================================================
# OCR NORMALIZATION
# ============================================================

def normalize_ocr_text(
    text: str
):

    if not text:

        return ""

    text = str(
        text
    ).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# OCR CHARACTER CORRECTION
# ============================================================

DIGIT_TO_LETTER = {

    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
    "2": "Z"

}


LETTER_TO_DIGIT = {

    "O": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2"

}


def correct_common_ocr_errors(
    text: str
):

    text = normalize_ocr_text(
        text
    )

    if len(text) < 8:

        return text, 0

    chars = list(
        text
    )

    corrections = 0

    # First 2 characters must be letters

    for index in range(
        0,
        min(2, len(chars))
    ):

        if chars[index].isdigit():

            replacement = (
                DIGIT_TO_LETTER.get(
                    chars[index]
                )
            )

            if replacement:

                chars[index] = replacement

                corrections += 1

    # Characters 2 and 3 must be digits

    for index in range(
        2,
        min(4, len(chars))
    ):

        if chars[index].isalpha():

            replacement = (
                LETTER_TO_DIGIT.get(
                    chars[index]
                )
            )

            if replacement:

                chars[index] = replacement

                corrections += 1

    # Last 4 characters must be digits

    start = max(
        0,
        len(chars) - 4
    )

    for index in range(
        start,
        len(chars)
    ):

        if chars[index].isalpha():

            replacement = (
                LETTER_TO_DIGIT.get(
                    chars[index]
                )
            )

            if replacement:

                chars[index] = replacement

                corrections += 1

    # Middle characters must be letters

    for index in range(
        4,
        max(4, len(chars) - 4)
    ):

        if chars[index].isdigit():

            replacement = (
                DIGIT_TO_LETTER.get(
                    chars[index]
                )
            )

            if replacement:

                chars[index] = replacement

                corrections += 1

    return (
        "".join(chars),
        corrections
    )


# ============================================================
# PLATE CANDIDATE EXTRACTION
# ============================================================

def extract_plate_candidate(
    text: str
):

    if not text:

        return None

    original = normalize_ocr_text(
        text
    )

    # Direct match

    direct_pattern = (
        r"[A-Z]{2}"
        r"[0-9]{2}"
        r"[A-Z]{0,3}"
        r"[0-9]{4}"
    )

    direct_matches = re.finditer(
        direct_pattern,
        original
    )

    for match in direct_matches:

        candidate = match.group()

        if is_valid_vehicle_number(
            candidate
        ):

            return {

                "text": candidate,

                "corrections": 0

            }

    # OCR correction

    corrected, corrections = (
        correct_common_ocr_errors(
            original
        )
    )

    corrected_matches = re.finditer(
        direct_pattern,
        corrected
    )

    for match in corrected_matches:

        candidate = match.group()

        if is_valid_vehicle_number(
            candidate
        ):

            return {

                "text": candidate,

                "corrections": corrections

            }

    return None


# ============================================================
# OCR PROCESSING
# ============================================================

def run_ocr_and_find_plate(
    image,
    source_name,
    all_ocr_texts
):

    if image is None:

        return None

    if image.size == 0:

        return None

    try:

        detected_texts = (
            ocr_reader.readtext(
                image,
                detail=0,
                paragraph=False
            )
        )

    except Exception as error:

        logger.warning(
            f"OCR failed for {source_name}: {error}"
        )

        return None

    logger.info(
        f"{source_name} OCR: "
        f"{detected_texts}"
    )

    all_ocr_texts.extend(
        detected_texts
    )

    # Check individual OCR results

    for text in detected_texts:

        result = (
            extract_plate_candidate(
                text
            )
        )

        if result:

            return {

                "text": result["text"],

                "detected": True,

                "valid_format": True,

                "source": source_name,

                "raw_text": text,

                "ocr_corrections": (
                    result["corrections"]
                ),

                "reason": None

            }

    # Combine nearby OCR results

    max_group_size = 3

    for start in range(
        len(detected_texts)
    ):

        for end in range(
            start + 1,
            min(
                start + max_group_size + 1,
                len(detected_texts) + 1
            )
        ):

            combined = "".join(
                detected_texts[
                    start:end
                ]
            )

            result = (
                extract_plate_candidate(
                    combined
                )
            )

            if result:

                return {

                    "text": result["text"],

                    "detected": True,

                    "valid_format": True,

                    "source": (
                        f"{source_name}_combined"
                    ),

                    "raw_text": combined,

                    "ocr_corrections": (
                        result["corrections"]
                    ),

                    "reason": None

                }

    return None


# ============================================================
# VEHICLE NUMBER EXTRACTION
# ============================================================

def extract_vehicle_number(
    file_path: str
):

    all_ocr_texts = []

    try:

        image = cv2.imread(
            file_path
        )

        if image is None:

            raise ValueError(
                "Unable to read image"
            )

        height, width = (
            image.shape[:2]
        )

        # Full image

        result = (
            run_ocr_and_find_plate(
                image,
                "full_image_ocr",
                all_ocr_texts
            )
        )

        if result:

            return result

        # Image regions

        regions = {

            "top_region_ocr": image[
                0:int(
                    height * 0.50
                ),
                0:width
            ],

            "middle_region_ocr": image[
                int(
                    height * 0.20
                ):int(
                    height * 0.80
                ),
                0:width
            ],

            "bottom_region_ocr": image[
                int(
                    height * 0.50
                ):height,
                0:width
            ],

            "center_region_ocr": image[
                int(
                    height * 0.25
                ):int(
                    height * 0.75
                ),
                int(
                    width * 0.10
                ):int(
                    width * 0.90
                )
            ]

        }

        for name, region in (
            regions.items()
        ):

            if region.size == 0:

                continue

            enlarged = cv2.resize(
                region,
                None,
                fx=2.5,
                fy=2.5,
                interpolation=cv2.INTER_CUBIC
            )

            result = (
                run_ocr_and_find_plate(
                    enlarged,
                    name,
                    all_ocr_texts
                )
            )

            if result:

                return result

        # Grayscale + CLAHE

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(
            gray
        )

        result = (
            run_ocr_and_find_plate(
                enhanced,
                "clahe_ocr",
                all_ocr_texts
            )
        )

        if result:

            return result

        # OTSU

        _, otsu_image = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )

        result = (
            run_ocr_and_find_plate(
                otsu_image,
                "otsu_ocr",
                all_ocr_texts
            )
        )

        if result:

            return result

        # Adaptive threshold

        adaptive_image = (
            cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
        )

        result = (
            run_ocr_and_find_plate(
                adaptive_image,
                "adaptive_ocr",
                all_ocr_texts
            )
        )

        if result:

            return result

        # No valid plate

        raw_text = " | ".join(
            all_ocr_texts
        )

        plate_like_text = None

        for text in all_ocr_texts:

            normalized = (
                normalize_ocr_text(
                    text
                )
            )

            if (

                6 <= len(normalized) <= 14

                and any(
                    char.isalpha()
                    for char in normalized
                )

                and any(
                    char.isdigit()
                    for char in normalized
                )

            ):

                plate_like_text = (
                    normalized
                )

                break

        if plate_like_text:

            return {

                "text": plate_like_text,

                "detected": True,

                "valid_format": False,

                "source": "combined_ocr",

                "raw_text": raw_text,

                "ocr_corrections": 0,

                "reason": (
                    "Plate-like text detected "
                    "but format is invalid"
                )

            }

        return {

            "text": None,

            "detected": False,

            "valid_format": False,

            "source": "ocr",

            "raw_text": (
                raw_text
                if raw_text
                else None
            ),

            "ocr_corrections": 0,

            "reason": (
                "No valid Indian vehicle "
                "number detected"
            )

        }

    except Exception as error:

        logger.exception(
            "Vehicle OCR failed"
        )

        return {

            "text": None,

            "detected": False,

            "valid_format": False,

            "source": "ocr",

            "raw_text": (
                " | ".join(
                    all_ocr_texts
                )
                if all_ocr_texts
                else None
            ),

            "ocr_corrections": 0,

            "reason": str(
                error
            )

        }


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def detect_duplicate(
    job_id: str,
    current_sha256: str,
    current_phash: str
):

    db = SessionLocal()

    try:

        # Exact duplicate

        exact_match = (
            db.query(
                ImageHash
            )
            .filter(
                ImageHash.sha256
                == current_sha256,

                ImageHash.image_id
                != job_id
            )
            .first()
        )

        if exact_match:

            db.merge(
                ImageHash(
                    image_id=job_id,
                    sha256=current_sha256,
                    phash=current_phash
                )
            )

            db.commit()

            return {

                "is_duplicate": True,

                "duplicate_type": "exact",

                "matched_processing_id": (
                    exact_match.image_id
                )

            }

        # Perceptual duplicate

        current_hash = (
            imagehash.hex_to_hash(
                current_phash
            )
        )

        all_hashes = (
            db.query(
                ImageHash
            )
            .filter(
                ImageHash.image_id
                != job_id
            )
            .all()
        )

        for stored in all_hashes:

            try:

                stored_hash = (
                    imagehash.hex_to_hash(
                        stored.phash
                    )
                )

                distance = (
                    stored_hash
                    - current_hash
                )

                if distance <= 5:

                    db.merge(
                        ImageHash(
                            image_id=job_id,
                            sha256=current_sha256,
                            phash=current_phash
                        )
                    )

                    db.commit()

                    return {

                        "is_duplicate": True,

                        "duplicate_type": (
                            "perceptual"
                        ),

                        "matched_processing_id": (
                            stored.image_id
                        ),

                        "hash_distance": int(
                            distance
                        )

                    }

            except Exception as error:

                logger.warning(
                    f"Hash comparison failed: "
                    f"{error}"
                )

        # New image

        db.merge(
            ImageHash(
                image_id=job_id,
                sha256=current_sha256,
                phash=current_phash
            )
        )

        db.commit()

        return {

            "is_duplicate": False,

            "duplicate_type": None,

            "matched_processing_id": None

        }

    finally:

        db.close()


# ============================================================
# SCREENSHOT DETECTION
# ============================================================

def analyze_screenshot(
    image
):

    height, width = (
        image.shape[:2]
    )

    aspect_ratio = (
        width / height
    )

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

    common_widths = {

        1080,
        1170,
        1284,
        1440,
        1242

    }

    common_heights = {

        1920,
        2532,
        2778,
        2560,
        2688

    }

    if width in common_widths:

        likelihood += 0.3

        signals.append(
            "common_screenshot_width"
        )

    if height in common_heights:

        likelihood += 0.3

        signals.append(
            "common_screenshot_height"
        )

    return {

        "screenshot_likelihood": round(
            min(
                likelihood,
                1.0
            ),
            2
        ),

        "signals": signals,

        "note": (
            "Heuristic only. "
            "This does not prove "
            "that the image is a screenshot."
        )

    }


# ============================================================
# METADATA / EXIF ANALYSIS
# ============================================================

def analyze_metadata(
    file_path: str
):

    metadata = {}

    editing_software = None

    try:

        with Image.open(
            file_path
        ) as image:

            exif_data = (
                image.getexif()
            )

            if exif_data:

                for tag_id, value in (
                    exif_data.items()
                ):

                    tag_name = (
                        ExifTags.TAGS.get(
                            tag_id,
                            str(tag_id)
                        )
                    )

                    metadata[
                        tag_name
                    ] = str(
                        value
                    )

                    if tag_name.lower() in {

                        "software",
                        "processingsoftware"

                    }:

                        editing_software = (
                            str(
                                value
                            )
                        )

        return {

            "has_exif": bool(
                metadata
            ),

            "metadata_count": len(
                metadata
            ),

            "metadata": metadata,

            "software": editing_software

        }

    except Exception as error:

        return {

            "has_exif": False,

            "metadata_count": 0,

            "metadata": {},

            "software": None,

            "error": str(
                error
            )

        }


# ============================================================
# PHOTO-OF-PHOTO ANALYSIS
# ============================================================

def analyze_photo_of_photo(
    image
):

    signals = []

    suspicion_score = 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    height, width = (
        image.shape[:2]
    )

    border_size = min(
        10,
        height,
        width
    )

    top_mean = np.mean(
        gray[
            :border_size,
            :
        ]
    )

    bottom_mean = np.mean(
        gray[
            -border_size:,
            :
        ]
    )

    left_mean = np.mean(
        gray[
            :,
            :border_size
        ]
    )

    right_mean = np.mean(
        gray[
            :,
            -border_size:
        ]
    )

    border_values = [

        top_mean,
        bottom_mean,
        left_mean,
        right_mean

    ]

    if (

        max(
            border_values
        )
        -
        min(
            border_values
        )
        > 80

    ):

        suspicion_score += 0.2

        signals.append(
            "uneven_borders"
        )

    if (

        width < 640

        or

        height < 480

    ):

        suspicion_score += 0.2

        signals.append(
            "low_resolution"
        )

    aspect_ratio = (
        width / height
    )

    if (

        aspect_ratio > 3.0

        or

        aspect_ratio < 0.33

    ):

        suspicion_score += 0.2

        signals.append(
            "extreme_aspect_ratio"
        )

    return {

        "photo_of_photo_likelihood": round(
            min(
                suspicion_score,
                1.0
            ),
            2
        ),

        "suspicious": bool(
            suspicion_score >= 0.5
        ),

        "signals": signals,

        "note": (
            "Heuristic analysis only. "
            "It cannot definitively prove "
            "that an image is a photo of another photo."
        )

    }


# ============================================================
# EDITING ANALYSIS
# ============================================================

def analyze_editing(
    image,
    file_path: str,
    metadata
):

    signals = []

    suspicion_score = 0.0

    software = metadata.get(
        "software"
    )

    if software:

        software_lower = (
            software.lower()
        )

        editing_tools = [

            "photoshop",
            "gimp",
            "lightroom",
            "canva",
            "snapseed",
            "picsart",
            "paint.net",
            "adobe"

        ]

        for tool in editing_tools:

            if tool in software_lower:

                suspicion_score += 0.5

                signals.append(
                    "editing_software_detected"
                )

                break

    extension = (
        Path(
            file_path
        )
        .suffix
        .lower()
    )

    if extension in {

        ".jpg",
        ".jpeg"

    }:

        signals.append(
            "jpeg_compression_present"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_ratio = np.mean(
        edges > 0
    )

    if edge_ratio > 0.35:

        suspicion_score += 0.2

        signals.append(
            "high_edge_density"
        )

    return {

        "suspicious": bool(
            suspicion_score >= 0.5
        ),

        "suspicion_score": round(
            min(
                suspicion_score,
                1.0
            ),
            2
        ),

        "signals": signals,

        "note": (
            "Heuristic analysis only. "
            "Editing signals do not prove manipulation."
        )

    }


# ============================================================
# COMPLETE IMAGE ANALYSIS
# ============================================================

def analyze_image(
    file_path: str,
    job_id: str
):

    image = cv2.imread(
        file_path
    )

    if image is None:

        raise ValueError(
            "Unable to read image file"
        )

    results = {}

    # 1. Blur

    results[
        "blur"
    ] = detect_blur(
        image
    )

    # 2. Brightness

    results[
        "brightness"
    ] = analyze_brightness(
        image
    )

    # 3. Dimensions

    results[
        "dimensions"
    ] = analyze_dimensions(
        image
    )

    # 4. SHA256

    sha256 = (
        calculate_sha256(
            file_path
        )
    )

    results[
        "file_hash"
    ] = sha256

    # 5. Perceptual hash

    phash = (
        calculate_perceptual_hash(
            file_path
        )
    )

    results[
        "perceptual_hash"
    ] = phash

    # 6. Duplicate

    results[
        "duplicate"
    ] = detect_duplicate(
        job_id,
        sha256,
        phash
    )

    # 7. Vehicle number

    results[
        "vehicle_number"
    ] = extract_vehicle_number(
        file_path
    )

    # 8. Screenshot

    results[
        "screenshot_analysis"
    ] = analyze_screenshot(
        image
    )

    # 9. Metadata

    results[
        "metadata_analysis"
    ] = analyze_metadata(
        file_path
    )

    # 10. Photo of photo

    results[
        "photo_of_photo_analysis"
    ] = analyze_photo_of_photo(
        image
    )

    # 11. Editing

    results[
        "editing_analysis"
    ] = analyze_editing(
        image,
        file_path,
        results[
            "metadata_analysis"
        ]
    )

    # ========================================================
    # ISSUE COLLECTION
    # ========================================================

    issues = []

    if results[
        "blur"
    ][
        "detected"
    ]:

        issues.append(
            "blurry_image"
        )

    if results[
        "brightness"
    ][
        "issue"
    ] in {

        "low_light",
        "very_low_light"

    }:

        issues.append(
            "low_light"
        )

    if results[
        "brightness"
    ][
        "issue"
    ] == "overexposed":

        issues.append(
            "overexposed_image"
        )

    if results[
        "duplicate"
    ][
        "is_duplicate"
    ]:

        issues.append(
            "duplicate_image"
        )

    if results[
        "screenshot_analysis"
    ][
        "screenshot_likelihood"
    ] >= 0.5:

        issues.append(
            "possible_screenshot"
        )

    if results[
        "photo_of_photo_analysis"
    ][
        "suspicious"
    ]:

        issues.append(
            "possible_photo_of_photo"
        )

    if results[
        "editing_analysis"
    ][
        "suspicious"
    ]:

        issues.append(
            "possible_suspicious_editing"
        )

    vehicle_result = (
        results[
            "vehicle_number"
        ]
    )

    if (

        vehicle_result[
            "detected"
        ]

        and

        not vehicle_result[
            "valid_format"
        ]

    ):

        issues.append(
            "invalid_vehicle_number_format"
        )

    if not results[
        "dimensions"
    ][
        "valid"
    ]:

        issues.append(
            "low_resolution"
        )

    results[
        "summary"
    ] = {

        "issues_detected": issues,

        "has_issues": bool(
            issues
        ),

        "total_checks": 11

    }

    return results


# ============================================================
# BACKGROUND IMAGE PROCESSING
# ============================================================

def process_image_async(
    job_id: str,
    max_retries: int = 3
):

    for attempt in range(
        1,
        max_retries + 1
    ):

        db = SessionLocal()

        try:

            job = (
                db.query(
                    ImageJob
                )
                .filter(
                    ImageJob.id
                    == job_id
                )
                .first()
            )

            if not job:

                logger.error(
                    f"Job not found: {job_id}"
                )

                return

            job.status = (
                "processing"
            )

            job.updated_at = (
                datetime.utcnow()
            )

            db.commit()

            logger.info(
                f"Processing attempt "
                f"{attempt}: {job_id}"
            )

            results = (
                analyze_image(
                    job.file_path,
                    job_id
                )
            )

            job.result = json.dumps(
                results
            )

            job.status = (
                "completed"
            )

            job.error_message = None

            job.updated_at = (
                datetime.utcnow()
            )

            db.commit()

            logger.info(
                f"Processing completed: "
                f"{job_id}"
            )

            return

        except Exception as error:

            logger.exception(
                f"Attempt {attempt} "
                f"failed for {job_id}"
            )

            try:

                job = (
                    db.query(
                        ImageJob
                    )
                    .filter(
                        ImageJob.id
                        == job_id
                    )
                    .first()
                )

                if (

                    job

                    and

                    attempt
                    >= max_retries

                ):

                    job.status = (
                        "failed"
                    )

                    job.error_message = (
                        str(
                            error
                        )
                    )

                    job.updated_at = (
                        datetime.utcnow()
                    )

                    db.commit()

                else:

                    db.rollback()

            except Exception:

                db.rollback()

        finally:

            db.close()

        if attempt < max_retries:

            wait_seconds = (
                2 ** attempt
            )

            time.sleep(
                wait_seconds
            )


# ============================================================
# API 1: UPLOAD IMAGE
# ============================================================

@app.post(
    "/api/v1/images"
)
async def upload_image(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    if file.content_type not in (
        ALLOWED_MIME_TYPES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPEG, PNG and WEBP "
                "images are supported"
            )
        )

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    if len(content) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "File size exceeds "
                "10 MB limit"
            )
        )

    try:

        verify_image = Image.open(
            io.BytesIO(
                content
            )
        )

        detected_format = (
            verify_image.format
        )

        verify_image.verify()

        if detected_format not in (
            ALLOWED_IMAGE_FORMATS
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid image format"
                )
            )

    except HTTPException:

        raise

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is not "
                "a valid image"
            )
        )

    processing_id = (
        str(
            uuid.uuid4()
        )
    )

    extension_map = {

        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp"

    }

    extension = (
        extension_map[
            detected_format
        ]
    )

    file_path = (
        UPLOAD_DIR
        /
        f"{processing_id}{extension}"
    )

    try:

        with open(
            file_path,
            "wb"
        ) as output_file:

            output_file.write(
                content
            )

        db = SessionLocal()

        try:

            job = ImageJob(

                id=processing_id,

                original_filename=(
                    file.filename
                ),

                file_path=str(
                    file_path
                ),

                content_type=(
                    file.content_type
                ),

                file_size=len(
                    content
                ),

                status="pending"

            )

            db.add(
                job
            )

            db.commit()

        finally:

            db.close()

    except Exception:

        if file_path.exists():

            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save image"
            )
        )

    thread = threading.Thread(

        target=process_image_async,

        args=(processing_id,),

        daemon=True

    )

    thread.start()

    return {

        "processing_id": (
            processing_id
        ),

        "status": "pending",

        "message": (
            "Image uploaded successfully "
            "and queued for processing"
        )

    }


# ============================================================
# API 2: STATUS
# ============================================================

@app.get(
    "/api/v1/images/{processing_id}/status"
)
def get_status(
    processing_id: str
):

    db = SessionLocal()

    try:

        job = (
            db.query(
                ImageJob
            )
            .filter(
                ImageJob.id
                == processing_id
            )
            .first()
        )

        if not job:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Processing ID not found"
                )
            )

        return {

            "processing_id": (
                processing_id
            ),

            "status": job.status,

            "error": (
                job.error_message
            )

        }

    finally:

        db.close()


# ============================================================
# API 3: RESULTS
# ============================================================

@app.get(
    "/api/v1/images/{processing_id}/results"
)
def get_results(
    processing_id: str
):

    db = SessionLocal()

    try:

        job = (
            db.query(
                ImageJob
            )
            .filter(
                ImageJob.id
                == processing_id
            )
            .first()
        )

        if not job:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Processing ID not found"
                )
            )

        status = job.status

        result = job.result

        error = job.error_message

    finally:

        db.close()

    if status == "pending":

        return {

            "processing_id": (
                processing_id
            ),

            "status": "pending",

            "message": (
                "Image is waiting "
                "for processing"
            )

        }

    if status == "processing":

        return {

            "processing_id": (
                processing_id
            ),

            "status": "processing",

            "message": (
                "Image is currently "
                "being analyzed"
            )

        }

    if status == "failed":

        return {

            "processing_id": (
                processing_id
            ),

            "status": "failed",

            "error": error

        }

    if not result:

        return {

            "processing_id": (
                processing_id
            ),

            "status": status,

            "analysis": None

        }

    return {

        "processing_id": (
            processing_id
        ),

        "status": status,

        "analysis": json.loads(
            result
        )

    }


# ============================================================
# API 4: COMPLETE JOB DETAILS
# ============================================================

@app.get(
    "/api/v1/images/{processing_id}"
)
def get_image_details(
    processing_id: str
):

    db = SessionLocal()

    try:

        job = (
            db.query(
                ImageJob
            )
            .filter(
                ImageJob.id
                == processing_id
            )
            .first()
        )

        if not job:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Processing ID not found"
                )
            )

        return {

            "processing_id": job.id,

            "original_filename": (
                job.original_filename
            ),

            "content_type": (
                job.content_type
            ),

            "file_size": (
                job.file_size
            ),

            "status": job.status,

            "created_at": (
                job.created_at
            ),

            "updated_at": (
                job.updated_at
            ),

            "error": (
                job.error_message
            )

        }

    finally:

        db.close()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/v1/health"
)
def health_check():

    return {

        "service": (
            "Intelligent Media "
            "Processing Pipeline"
        ),

        "status": "running",

        "version": "2.0.0"

    }