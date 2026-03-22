# health.service.py
import json
import os

from typing import List, Tuple, Optional, Dict, Any, Union

import requests


from .schema import Detection  # relative import

from common.ai_model.implements.gemini import geminiModel


API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY or GENAI_API_KEY in your environment")


class HealthService:
    """
    Lightweight service for disease detection using a multimodal LLM (Gemini).
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        pass

    def tree_disease_diagnosis_from_url(self, image_url: str) -> Dict[str, Any]:
        """
        Download an image, run detection, annotate, upload annotated image, return presigned URL and detection list.
        """
        r = requests.get(image_url, timeout=20)
        r.raise_for_status()
        image_bytes = r.content
        return self.tree_disease_diagnosis(image_bytes)

    def tree_disease_diagnosis_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Run detection on raw image bytes.
        """
        return self.tree_disease_diagnosis(image_bytes)

    def tree_disease_diagnosis(self, image_bytes: bytes) -> Dict[str, Any]:
        PROMPT = (
            "Bạn là chuyên gia nông học và sâu bệnh cây trồng. "
            "Dựa vào ảnh được gửi kèm, hãy phân tích và trả về DUY NHẤT một đối tượng JSON (KHÔNG kèm bất kỳ văn bản giải thích nào ngoài JSON). "
            "Các đề xuất hành động phải phù hợp ở Việt Nam, sử dụng tên thân thuộc, ưu tiên thuốc sinh học và biện pháp hữu cơ."
            "Định dạng JSON phải chính xác theo mô tả sau:\n\n"
            '   - prediction: (string) dự đoán ngắn gọn (ví dụ: "Cây bị rệp trắng")\n'
            '   - severity_level: (string) một trong ["Thấp","Trung bình","Cao", "Rất cao"]\n'
            "   - possible_causes: (array of objects) các nguyên nhân khả dĩ khác, mỗi item có keys:\n"
            "       * cause: (string) mô tả nguyên nhân,\n"
            "       * confidence: (number) độ tin cậy 0..1 cho nguyên nhân này,\n"
            "       * evidence: (string) bằng chứng/quan sát ngắn (ví dụ: 'vết vàng theo gân', 'lá cuốn lại', 'độ ẩm cao')\n"
            "   - recommended_actions: (array of objects) danh sách hành động theo bước, mỗi item có keys:\n"
            '       * name: (string) mô tả ngắn gọn, khoảng 10 từ (ví dụ: "Phun thuốc sinh học Azadirachtin 3ml/l"),\n'
            '       * timing: (string) khi nào thực hiện (ví dụ: "Ngay lập tức", "Trong 3-5 ngày"),\n'
            "       * description: (string, optional) mô tả chi tiết, lưu ý an toàn/độ pha/điều kiện, ...\n"
            "       * targetValue: (number) số lần cần thực hiện tronaa tuần (ví dụ: 1, 2, 3...)\n"
            "       * numOfWeeks: (number) số tuần cần thực hiện (ví dụ: 1, 2, 3...)\n"
            "   - chemical_recommendations: (array of strings) gợi ý thuốc hóa học (tên hoạt chất hoặc thương hiệu) hoặc []\n"
            "   - biological_recommendations: (array of strings) gợi ý biện pháp sinh học/thuốc hữu cơ hoặc []\n"
            '   - monitoring_plan: (string) kế hoạch theo dõi (ví dụ: "Kiểm tra lại sau 7 ngày, tập trung vào tán lá phía dưới")\n'
            "   - preventive_measures: (array of strings) các biện pháp phòng ngừa ngắn gọn\n"
            "   - additional_notes: (string) (tùy chọn)\n\n"
            "Ví dụ JSON mẫu (bắt buộc format tương tự):\n"
            "{\n"
            '    "prediction":"Cây bị rệp trắng (whiteflies)",\n'
            '    "severity_level":"Cao",\n'
            '    "possible_causes":[\n'
            '      {"cause":"Điều kiện ấm ẩm, nhiều cỏ dại quanh gốc","confidence":0.75,"evidence":"nhiều chấm trắng tập trung mặt dưới lá"},\n'
            '      {"cause":"Thiếu thông gió do cây trồng quá dày","confidence":0.45,"evidence":"lá gần nhau, ít ánh sáng vào tán"}\n'
            "    ],\n"
            '    "recommended_actions": [\n'
            '      {"name":"Phun thuốc sinh học có hoạt chất Azadirachtin","timing":"Ngay lập tức","description":"Phun vào sáng sớm hoặc chiều mát", "targetValue": 2, "numOfWeeks": 1},\n'
            '      {"name":"Cắt bỏ lá bị nhiễm nặng và tiêu hủy","timing":"Trong 1-2 ngày","description":"Đeo găng tay, tiêu hủy xa vườn", "targetValue": 2, "numOfWeeks": 1}\n'
            "    ],\n"
            '    "chemical_recommendations":["Imidacloprid (dùng thận trọng)"],\n'
            '    "biological_recommendations":["Thả thiên địch (bọ rùa), dùng bẫy dính vàng"],\n'
            '    "monitoring_plan":"Kiểm tra lại sau 7 ngày; nếu số lượng không giảm, lặp lại phun theo hướng dẫn.",\n'
            '    "preventive_measures":["Cắt tỉa thông thoáng","Quản lý cỏ dại quanh gốc"],\n'
            '    "additional_notes":"Tránh phun lúc nắng gắt; tuân thủ liều lượng nhà sản xuất."\n'
            "}\n\n"
            "QUAN TRỌNG: Nếu không phát hiện được bất kỳ dấu hiệu nào, hãy trả:\n"
            '{"analysis_vn": {"prediction":"Không phát hiện bệnh rõ rệt","severity_level":"LOW", "possible_causes":[], "recommended_actions":[], "chemical_recommendations":[], "biological_recommendations":[], "monitoring_plan":"", "preventive_measures":[], "additional_notes":""}}\n\n'
            "KHÔNG được thêm text, chú thích hay giải thích ngoài JSON này. TẤT CẢ phần phân tích phải bằng tiếng Việt."
        )

        return geminiModel.generate_from_image(image_bytes, PROMPT)

    def quiz_generation(self, source_text: str, num_questions: int) -> Dict[str, Any]:
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
        - Nếu là đáp án đúng: giải thích vì sao đúng như một giáo viên tận tâm
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
        return geminiModel.generate_json_content(prompt)

    def extract_tree_journey(self, raw_data: str) -> Dict[str, Any]:
        raw_str = (
            json.dumps(raw_data, ensure_ascii=False)
            if isinstance(raw_data, dict)
            else raw_data
        )
        PROMPT = f"""
        Bạn là một trợ lý chuyển đổi dữ liệu.
        Đây là dữ liệu về hành trình chăm sóc cây của học sinh

        Nhiệm vụ của bạn là đọc một đối tượng JSON hành trình chăm sóc của cây (raw) và chuyển đổi nó thành một object sạch, sẵn sàng để hiển thị trên giao diện web cho trang "My Tree Journey".
        Nội dung được kể lại theo ngôi thứ nhất (học sinh kể lại hành trình chăm sóc cây), xưng mình "Mình", tuyệt đối không xưng hô khác
        MỤC TIÊU
        Chuyển đổi schema đầu vào thành JSON chỉ gồm các field cần thiết cho template web.

        QUY TẮC QUAN TRỌNG
        - Chỉ trả về JSON.
        - Không bọc đầu ra dưới dạng markdown.
        - Không thêm giải thích.
        - Chỉ trả về đúng các field được yêu cầu, không thêm field khác.
        - Giữ tiếng Việt khi tự nhiên cho tiêu đề giai đoạn và văn bản kể chuyện.
        - Viết ngôi thứ nhất, phong cách học sinh, gần gũi, mạch lạc.
        - Ưu tiên ý nghĩa thay vì bê nguyên văn bản.
        - Văn phong ấm áp, tự nhiên, trung thực với dữ liệu.

        MẪU ĐẦU RA
        Trả về object có đúng cấu trúc này:

        {{
        "heroImage": "",
        "heroAlt": "",
        "heroBadge": "",
        "heroTitle": "",
        "heroDescription": "",
        "breed": "",
        "cultivar": "",
        "farmingMethod": "",
        "growDuration": 0,
        "growthStages": [
            {{
            "index": 1,
            "stageDisplayName": "",
            "stageImage": "",
            "stageImageCaption": "",
            "studentStory": "",
            "gardenNote": "",
            "layoutClass": "",
            "contentSpacingClass": "",
            "emptySpacingClass": "",
            "cardAccentClass": "",
            "timelineDotClass": "",
            "noteMarginClass": ""
            }}
        ],
        "harvestTitle": "",
        "estimatedFruitWeight": 0,
        "fruitWeight": 0,
        "harvestBadge": "",
        "harvestQuote": "",
        "footerTitle": "",
        "footerDescription": "",
        }}

        CHỈ SINH ĐÚNG CÁC FIELD TRÊN
        Không trả về bất kỳ field nào khác ngoài mẫu đầu ra

        HƯỚNG DẪN ÁNH XẠ TRƯỜNG

        1. THÔNG TIN PHẦN ĐẦU
        - heroImage:
        Chọn ảnh ưu tiên theo thứ tự:
        1) imageKeys[0]
        2) harvestTreeImageKeys[0]
        3) harvestFruitImageKeys[0]
        - heroAlt:
        "Tree Background"
        - heroBadge:
        "Nhật ký xanh của mình"
        - heroTitle:
        "My Tree Journey: Câu chuyện về {{name}}"
        - heroDescription:
        Viết mở đầu tiếng Việt tự nhiên, dựa trên mô tả cây và vòng đời.
        Ngôi thứ nhất (học sinh kể lại hành trình chăm sóc cây), văn phong học sinh, 2–3 câu.
        Có thể nhắc thời lượng hành trình nếu có growDuration.
        - breed = breed
        - cultivar = cultivar
        - farmingMethod:
        "Hữu cơ" nếu isOrganic = true, nếu không thì "Thông thường"
        - growDuration = growDuration

        2. GIAI ĐOẠN PHÁT TRIỂN
        Sinh growthStages từ input.growthStage.
        Sắp xếp tăng dần theo afterWeeks.

        Với mỗi stage, chỉ trả về các field sau:

        - index:
        Thứ tự từ 1 sau khi đã sắp xếp

        - stageDisplayName:
        Chuyển stageName thành nhãn tiếng Việt + tiếng Anh:
        SEEDLING -> "Giai đoạn Mầm Non (Seedling)"
        SAPLING -> "Giai đoạn Cây Con (Sapling)"
        VEGETATIVE -> "Giai đoạn Phát triển (Vegetative)"
        FLOWERING -> "Giai đoạn Ra Hoa (Flowering)"
        FRUITING -> "Giai đoạn Ra Trái (Fruiting)"
        Nếu không khớp, tự đặt tên phù hợp.

        - stageImage:
        Chọn ảnh tốt nhất theo thứ tự:
        1) diseaseDiagnosises[0].imageKeys[0] nếu có ảnh bệnh rõ ràng
        2) stage.imageKeys[0]
        3) root imageKeys[0]
        4) harvestTreeImageKeys[0]

        - stageImageCaption:
        Viết caption ngắn 1–2 câu, giàu hình ảnh, đúng với đặc điểm cây ở giai đoạn đó.
        Chỉ mô tả cây trồng trong giai đoạn này, không lan man.
        Văn phong gần gũi, ấm áp.

        - studentStory:
        Dựa trên studentObservation, viết lại thành 2–3 câu kể chuyện ngôi thứ nhất.
        Giọng học sinh, tự nhiên, có quan sát và bài học rút ra.
        Nếu studentObservation ngắn, có thể diễn đạt lại phong phú hơn nhưng không sai dữ liệu.

        - gardenNote:
        Dựa trên stage.comment, viết thành 1 đoạn văn 2–4 câu.
        Ưu tiên:
        1) hành động/ghi nhận của nông dân
        2) quan sát hoặc cảm nhận của học sinh
        3) gợi ý hoặc chẩn đoán từ AI nếu thật sự quan trọng
        Bỏ qua lời chào hoặc nội dung rỗng.
        Không dùng bullet.
        Nếu stage hầu như không có nội dung hữu ích, cho chuỗi rỗng "".

        - layoutClass:
        Nếu index lẻ: "md:flex-row"
        Nếu index chẵn: "md:flex-row-reverse"

        - contentSpacingClass:
        Nếu index lẻ: "pr-0 md:pr-12"
        Nếu index chẵn: "pl-0 md:pl-12"

        - emptySpacingClass:
        Nếu index lẻ: "pl-0 md:pl-12"
        Nếu index chẵn: ""

        - cardAccentClass:
        Nếu index lẻ: "border-sage"
        Nếu index chẵn: "border-earth"

        - timelineDotClass:
        Nếu index lẻ: "bg-sage"
        Nếu index chẵn: "bg-earth"

        - noteMarginClass:
        Nếu gardenNote có nội dung: "mb-6"
        Nếu gardenNote rỗng: ""

        3. PHẦN THU HOẠCH
        - harvestTitle:
        "🎉 Harvest Day: The Grand Finale!"
        - estimatedFruitWeight = estimatedFruitWeight
        - fruitWeight = fruitWeight
        - harvestBadge:
        Nếu fruitWeight > estimatedFruitWeight -> "Vượt kế hoạch"
        Nếu fruitWeight = estimatedFruitWeight -> "Đúng kế hoạch"
        Nếu fruitWeight < estimatedFruitWeight -> "Chưa đạt kế hoạch"
        - harvestQuote:
        Viết tiếng Việt ngôi thứ nhất dựa trên harvestedAt và kết quả cuối cùng.
        Tone ví dụ:
        "Vào ngày {{dd/MM/yyyy}}, sự chờ đợi cuối cùng cũng đã kết thúc. Cầm trái ngọt trên tay, mình nhận ra bao nhiêu nỗ lực đã bỏ ra cho thực phẩm chúng ta ăn hằng ngày. Đây thực sự là một hành trình đáng nhớ."
        Ngày theo định dạng dd/MM/yyyy.

        4. CHÂN TRANG
        - footerTitle:
        "Cảm ơn bạn đã lắng nghe!"
        - footerDescription:
        Viết kết ngắn tiếng Việt, tối đa 2 câu.
        Thể hiện niềm tự hào, sẻ chia, kiên nhẫn và tôn trọng sự sống.

        QUY TẮC CHẤT LƯỢNG
        - Văn phong kể chuyện tự nhiên, ấm áp.
        - Không bịa nội dung mâu thuẫn dữ liệu gốc.
        - Có thể diễn đạt phong phú hơn nhưng phải trung thực với nguồn.
        - Nếu dữ liệu ít, tổng hợp nhẹ cho hợp lý.
        - Tránh lặp lại cùng một mẫu câu giữa các giai đoạn.
        - Caption ngắn, giàu hình ảnh.
        - gardenNote thực tiễn, giống một đoạn kể chuyện ngắn.

        ĐẦU VÀO
        Đây là schema gốc cần chuyển đổi:

        {raw_str}
        """

        return geminiModel.generate_json_content(PROMPT)
