import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Khởi tạo Trạng thái Session (MỚI) ---
# Dùng để lưu trữ lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        # SỬA LỖI: Cần truyền df[col] (Series) vào pd.to_numeric thay vì col (string)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý lỗi chia cho 0 thủ công cho mẫu số tỷ trọng
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (Phân tích một lần) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét (Chức năng 5)."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm xử lý Chat (MỚI) ---
def handle_chat_query(prompt, data_for_ai_context, api_key):
    """Xử lý câu hỏi chat, duy trì lịch sử và ngữ cảnh dữ liệu."""
    try:
        # Tạo prompt đầy đủ, bao gồm ngữ cảnh dữ liệu để AI luôn có thông tin
        system_instruction = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Hãy trả lời câu hỏi của người dùng dựa trên ngữ cảnh dữ liệu tài chính sau:
        ---
        {data_for_ai_context}
        ---
        Hãy duy trì tính chuyên nghiệp và trả lời ngắn gọn, tập trung vào dữ liệu đã cho.
        """
        
        # Chuyển đổi lịch sử st.session_state sang định dạng API
        # Lấy lịch sử chỉ 5 tin nhắn gần nhất để tránh vượt giới hạn token
        recent_messages = st.session_state.messages[-5:]
        
        # Lấy tin nhắn người dùng vừa gửi để thêm vào list contents
        user_message_part = {"role": "user", "parts": [{"text": prompt}]}
        
        # Lịch sử + tin nhắn mới nhất
        contents = [
            {"role": msg["role"], "parts": [{"text": msg["content"]}]}
            for msg in recent_messages
        ] + [user_message_part]
        
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        # Gọi API với lịch sử và system instruction
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            system_instruction=system_instruction
        )
        
        ai_response = response.text
        
        # Thêm phản hồi của AI vào lịch sử session state
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        return ai_response

    except APIError as e:
        error_message = f"Lỗi gọi Gemini API trong Chat: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        return error_message
    except Exception as e:
        error_message = f"Lỗi không xác định trong Chat: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        return error_message

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo các biến thanh toán để tránh lỗi nếu người dùng chưa tải file
thanh_toan_hien_hanh_N = "N/A" 
thanh_toan_hien_hanh_N_1 = "N/A"

if uploaded_file is not None:
    try:
        # Xóa lịch sử chat khi tải file mới để tránh nhầm lẫn ngữ cảnh
        st.session_state.messages = []
        
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, xử lý chia cho 0
                if no_ngan_han_N != 0:
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                else:
                    thanh_toan_hien_hanh_N = "Không xác định"
                    
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                else:
                    thanh_toan_hien_hanh_N_1 = "Không xác định"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else thanh_toan_hien_hanh_N_1
                    )
                with col2:
                    if isinstance(thanh_toan_hien_hanh_N, (int, float)) and isinstance(thanh_toan_hien_hanh_N_1, (int, float)):
                        delta_value = thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1
                    else:
                        delta_value = None
                        
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, (int, float)) else thanh_toan_hien_hanh_N,
                        delta=f"{delta_value:.2f}" if delta_value is not None else None
                    )
                
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số. Chỉ số sẽ hiển thị là N/A.")
                 thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5 & 6
                 thanh_toan_hien_hanh_N_1 = "N/A"
            except ZeroDivisionError:
                st.warning("Chỉ số thanh toán không thể tính do Nợ Ngắn Hạn bằng 0. Chỉ số sẽ hiển thị là 'Không xác định'.")
                thanh_toan_hien_hanh_N = "Không xác định"
                thanh_toan_hien_hanh_N_1 = "Không xác định"

            # Chuẩn bị dữ liệu để gửi cho AI (Dùng cho cả Chức năng 5 và 6)
            data_for_ai_context = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)',  
                    'Tăng trưởng Tài sản ngắn hạn (%)',  
                    'Thanh toán hiện hành (N-1)',  
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    # Đảm bảo chỉ số Tăng trưởng Tài sản ngắn hạn tồn tại trước khi truy cập
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" 
                        if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else "N/A",  
                    f"{thanh_toan_hien_hanh_N_1}",  
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False)
            
            # --- Chức năng 5: Nhận xét AI (Một lần) ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI) - Tự động")
            
            if st.button("Yêu cầu AI Phân tích Tự động"):
                api_key = st.secrets.get("GEMINI_API_KEY")  
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai_context, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

            # --- Chức năng 6: Khung Chat AI (Hỏi Đáp về Dữ liệu) - MỚI ---
            st.markdown("---")
            st.subheader("6. Khung Chat AI: Hỏi đáp chuyên sâu về Báo cáo Tài chính")
            
            # Hiển thị lịch sử chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Xử lý input mới
            chat_input_key = "chat_query_input"
            user_prompt = st.chat_input("Hỏi Gemini AI về báo cáo tài chính...", key=chat_input_key)
            
            if user_prompt:
                api_key = st.secrets.get("GEMINI_API_KEY")
                
                if not api_key:
                    st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Không thể chat.")
                else:
                    # Thêm tin nhắn của người dùng ngay lập tức
                    st.session_state.messages.append({"role": "user", "content": user_prompt})
                    
                    # Gọi hàm xử lý chat
                    with st.spinner('Gemini đang phân tích và trả lời...'):
                        handle_chat_query(user_prompt, data_for_ai_context, api_key)
                        
                        # Bắt buộc gọi rerun để Streamlit hiển thị tin nhắn mới nhất
                        st.rerun() 

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích và sử dụng khung chat.")
