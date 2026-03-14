from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class AIModelInterface(ABC):
    @abstractmethod
    def generate_from_image(
        self, image_bytes: bytes, prompt
    ) -> Tuple[List[dict], Dict[str, Any]]:
        pass

    @abstractmethod
    def generate_text_content(self, source: str, prompt: str) -> Dict[str, Any]:
        pass
