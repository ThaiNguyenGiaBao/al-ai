import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont


def parse_detections(text: str) -> Dict[str, Any]:
    """
    Forgiving parser: attempt strict JSON first; else extract first [...] block and parse.
    Returns a dictionary.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        raise ValueError("Invalid JSON format")
        pass
