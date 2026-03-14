from google import genai
from google.genai import types
import os
import json
from dotenv import load_dotenv

load_dotenv()


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_reasoning_quiz(
    source_text: str,
    num_questions: int = 5,
):
    prompt = f"""
    Bạn là một hệ thống tạo câu hỏi trắc nghiệm phục vụ học tập.

    Dựa CHỈ vào nội dung văn bản bên dưới, hãy tạo {num_questions} câu hỏi trắc nghiệm.

    YÊU CẦU QUAN TRỌNG:
    - Câu hỏi phải kiểm tra khả năng HIỂU và SUY LUẬN từ nội dung văn bản
    - Không hỏi chi tiết ghi nhớ máy móc (ví dụ: số liệu, tên riêng vụn vặt)
    - Không sử dụng kiến thức bên ngoài văn bản
    - Mỗi câu hỏi có đúng 4 lựa chọn
    - Chỉ có 1 đáp án đúng
    - Không sử dụng các cụm từ như: "văn bản", "đoạn văn", "nội dung trên", "theo đoạn trên"
    - Không thêm bất kỳ văn bản nào ngoài JSON
    - Ngôn ngữ: tiếng Việt, rõ ràng, gần gũi với người học

    MỖI CÂU HỎI PHẢI BAO GỒM:
    - Nội dung câu hỏi
    - 4 phương án trả lời
    - correct_index (giá trị từ 0 đến 3)
    - Giải thích cho TỪNG phương án:
    - Nếu là đáp án đúng: giải thích vì sao đúng dựa trên văn bản
    - Nếu là đáp án sai: giải thích ngắn gọn, thân thiện (ví dụ: "Đáp án này chưa đúng vì...", "Bạn có thể nhầm lẫn vì...")

    ĐỊNH DẠNG JSON (BẮT BUỘC, PHẢI HỢP LỆ):
    {{
    "questions": [
        {{
        "question": "Nội dung câu hỏi",
        "choices": [
            "Phương án A",
            "Phương án B",
            "Phương án C",
            "Phương án D"
        ],
        "correct_index": 0,
        "explanations": [
            "Giải thích cho phương án A",
            "Giải thích cho phương án B",
            "Giải thích cho phương án C",
            "Giải thích cho phương án D"
        ]
        }}
    ]
    }}

    LƯU Ý:
    - explanations phải có đúng 4 phần tử, theo đúng thứ tự của choices
    - Không nhắc lại nguyên văn phương án trong phần giải thích
    - Không nhắc đến chỉ số (0,1,2,3) trong nội dung giải thích

    VĂN BẢN NGUỒN:
    \"\"\"
    {source_text}
    \"\"\"
    """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2, response_mime_type="application/json"
        ),
    )

    return json.loads(response.text)


if __name__ == "__main__":
    text = """
Quang hợp là quá trình trong đó cây xanh sử dụng năng lượng ánh sáng
để tổng hợp chất dinh dưỡng từ carbon dioxide và nước.
Quá trình này không chỉ giúp cây phát triển mà còn cung cấp oxy cho môi trường.
"""

    quiz = generate_reasoning_quiz(text, num_questions=2)
    print(json.dumps(quiz, indent=2, ensure_ascii=False))
