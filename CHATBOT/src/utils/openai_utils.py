from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import LLM_MODEL_NAME
from dotenv import load_dotenv

load_dotenv()

def choose_llm_model(model_name):
    if "gpt" in model_name:
        llm = ChatOpenAI(model=model_name, temperature=0)
    elif "gemini" in model_name:
        llm = ChatGoogleGenerativeAI(model="models/" + model_name, temperature=0)
    
    return llm

llm = choose_llm_model(LLM_MODEL_NAME)

async def clarify_question(query_text):
    prompt = f"""Bạn là một trợ lý AI chuyên xử lý ngôn ngữ tự nhiên. Nhiệm vụ của bạn là chuẩn hóa câu hỏi tiếng Việt từ dạng tự nhiên, không chính thức hoặc lặp từ thành dạng câu hỏi rõ ràng, ngắn gọn, đúng ngữ pháp và giữ nguyên ý nghĩa gốc. Hãy thực hiện các bước sau:

1. Loại bỏ các từ thừa như "dạ", "ạ", "cho em hỏi", "mình", "vậy", "với" khi không cần thiết.
2. Sửa lỗi chính tả, viết hoa đầu câu và các danh từ riêng (nếu có).
3. Chuyển câu thành dạng chuẩn, mạch lạc, không lặp ý.
4. Giữ nguyên nội dung và ý nghĩa của câu hỏi.

Dưới đây là ví dụ:  
- Câu gốc: "Dạ cho e hỏi HUIT và DCT là 1 đúng không"  
- Câu chuẩn hóa: "HUIT và DCT là một đúng không?"

Bây giờ, hãy chuẩn hóa câu hỏi sau:  

"{query_text}"
Chỉ trả về câu đã chuẩn hóa và không giải thích gì thêm!
""" 
    response = await llm.ainvoke(prompt)
    return response.content

async def choose_best_match(clarified_question, retrieved_questions):
    question_list_text = "\n".join([f"ID {q['id']}: {q['question']}" for q in retrieved_questions])
#     prompt = f"""Bạn là một trợ lý AI hiểu tiếng Việt. 
# Dưới đây là một câu truy vấn gốc và danh sách các câu hỏi gần nhất. 
# Nhiệm vụ của bạn là chọn ra một câu hỏi trong danh sách có ý nghĩa gần nhất với câu truy vấn gốc. 

# Câu truy vấn gốc: '{clarified_question}'

# Danh sách câu hỏi:\n{question_list_text}

# Chỉ trả về duy nhất số nguyên là ID tương ứng của câu hỏi phù hợp nhất, không thêm bất kỳ ký tự hoặc từ nào khác như 'ID', dấu chấm hay giải thích.
# Nếu trong danh sách không có câu nào giống, trả về '0' nhé
# """
    prompt = f"""Bạn là chuyên gia AI phân tích ngữ nghĩa tiếng Việt.

Câu truy vấn: '{clarified_question}'

Danh sách câu hỏi:
{question_list_text}

TIÊU CHÍ SO KHỚP (yêu cầu đạt TẤT CẢ):
1. HOÀN TOÀN CÙNG Ý ĐỊNH: Phải giải quyết chính xác vấn đề/yêu cầu của người dùng
2. NGỮ NGHĨA TƯƠNG ĐỒNG > 90%: Nội dung và ý nghĩa phải gần như giống hệt
3. CÙNG CHỦ ĐỀ VÀ PHẠM VI: Đề cập đến cùng đối tượng và giới hạn tương tự
4. CÙNG ĐỘ PHỨC TẠP: Mức độ chi tiết và độ sâu tương đương

CHỈ trả về ID của câu hỏi nếu đạt ngưỡng tương đồng > 90%. Nếu không có, trả về '0'.
Không thêm bất kỳ ký tự hay giải thích nào.
"""
    response = await llm.ainvoke(prompt)
    return response.content
