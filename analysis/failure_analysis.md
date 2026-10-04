# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Lê Minh Sang  
**Mã số học viên:** 2A202602864  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ | Đánh giá Benchmark |
|--------|---------------|------------|---|---------------------|
| **Faithfulness** | 0.9375 | 0.9017 | -0.0358 | Vượt ngưỡng chuẩn (>= 0.85). Đạt tiêu chuẩn bonus. |
| **Answer Relevancy** | 0.7661 | 0.8710 | +0.1049 | Cải thiện vượt bậc (+10.5%), trả lời đúng trọng tâm. |
| **Context Precision** | 0.9250 | 0.9250 | 0.0000 | Duy trì mức cực cao, tài liệu liên quan ở top đầu. |
| **Context Recall** | 0.8250 | 0.8667 | +0.0417 | Tăng +4.17%, độ phủ thông tin ngữ cảnh mở rộng. |

> **Ghi chú:** Toàn bộ 4 chỉ số của Production RAG đều vượt ngưỡng **>= 0.75**, trong đó Faithfulness đạt **0.9017 >= 0.85**, đủ điều kiện nhận trọn vẹn điểm thưởng tối đa (+6 điểm bonus RAGAS).

---

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Dựa trên thông tin trong context:\n- **Số ngày phép năm:** Nhân viên 9 năm thâm niên được 18 ngày phép (15 + 3).\n- **Khoảng lương:** Không tìm thấy.
- **Worst metric:** Faithfulness (0.50) / Context Recall (0.50)
- **Error Tree:** Output thiếu 1 vế (khoảng lương) → Context thiếu tài liệu thang bảng lương (Salary Scale) → Retrieval bỏ sót tài liệu lương do câu hỏi kép (multi-part query: nghỉ phép + lương) → Query Embedding bị lệch ngữ nghĩa về phía chủ đề nghỉ phép.
- **Root cause:** Đây là câu hỏi phức hợp (Compound / Multi-part Query) hỏi đồng thời về 2 thực thể thuộc 2 văn bản chính sách độc lập (`quy_dinh_nghi_phep.md` và `thang_bang_luong.md`). Vector embedding của Dense Search và từ khóa BM25 bị chi phối bởi các từ khóa nổi bật "Senior", "thâm niên", "ngày phép", dẫn đến top-20 candidates đều thuộc về chính sách nghỉ phép, bỏ sót hoàn toàn văn bản thang bảng lương.
- **Suggested fix:** Áp dụng kỹ thuật **Query Decomposition / Sub-query Generation**: Sử dụng LLM phân tách câu hỏi phức thành 2 câu hỏi đơn độc lập: (1) "Nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép?" và (2) "Lương nhân viên Senior trong khoảng nào?", thực hiện truy vấn song song rồi hợp nhất context trước khi chuyển qua Cross-Encoder reranking.

---

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Theo quy định trong context, khoản tạm ứng chưa thanh toán sau 15 ngày sẽ bị tính phí phạt **2%/tháng** trên số tiền chưa hoàn ứng (15.000.000 VNĐ, tương đương 300.000 VNĐ/tháng) và được khấu trừ vào lương tháng kế tiếp. *(Context không có quy định chi tiết về công thức tính số ngày quá hạn lẻ mà chỉ nêu mức phí chung là 2%/tháng trên số tiền chưa hoàn ứng).*
- **Worst metric:** Context Recall (0.50)
- **Error Tree:** Output nêu được nguyên tắc 2%/tháng và mức trần 300k nhưng thiếu phép tính pro-rata cho 5 ngày lẻ → Context chỉ lấy được chunk nguyên tắc chung (15 ngày, 2%/tháng) mà thiếu chunk hướng dẫn tính chi tiết theo ngày lẻ → Chunking cắt ngang giữa điều khoản chính và điều khoản diễn giải.
- **Root cause:** Khi áp dụng Hierarchical Chunking với kích thước child chunk 256 tokens, điều khoản nguyên tắc tạm ứng nằm ở chunk trước, còn phụ lục/ví dụ công thức tính chia theo ngày lẻ (pro-rata) lại bị đẩy sang chunk kế tiếp. Dense retrieval chỉ match chunk chứa từ khóa "tạm ứng", "thanh toán", "phạt".
- **Suggested fix:** Triển khai **Parent-Document Retrieval (Small-to-Big Retrieval)**: Khi một child chunk được truy xuất và xếp hạng cao bởi Cross-Encoder, hệ thống sẽ thay thế child chunk bằng toàn bộ Parent Chunk (2048 tokens) hoặc mở rộng ngữ cảnh lân cận (sliding window context expansion) trước khi đưa vào LLM Context.

---

### #3
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** Theo Chính sách mật khẩu v2 (thay thế Chính sách mật khẩu v1.0), mật khẩu phải được thay đổi **mỗi 120 ngày** (theo quy định trước đó tại v1 là **mỗi 90 ngày**).
- **Worst metric:** Context Precision (0.4999)
- **Error Tree:** Output trả lời chính xác và phân định rõ 2 phiên bản → Tuy nhiên điểm Context Precision bị phạt vì trong top-3 retrieved context có cả chunk từ v1.0 (90 ngày) đứng ngang hàng với chunk từ v2.0 (120 ngày).
- **Root cause:** Trong kho dữ liệu có tồn tại tài liệu cũ đã hết hiệu lực (`chinh_sach_mat_khau_v1.0.md`) cùng với tài liệu hiện hành (`chinh_sach_mat_khau_v2.0.md`). Do câu hỏi không nêu rõ phiên bản hay mốc thời gian, cả BM25 và Dense Search đều cho điểm similarity rất cao cho cả hai phiên bản, dẫn đến việc tài liệu cũ bị đưa vào top context.
- **Suggested fix:** Thiết lập cơ chế **Metadata Pre-Filtering / Temporal Filtering**: Trích xuất metadata `version`, `effective_date`, `status` (active / superseded). Mặc định truy vấn sẽ chỉ filter các tài liệu có `status == "active"`. Trong trường hợp câu hỏi yêu cầu so sánh lịch sử thì mới truy vấn tài liệu cũ.

---

### #4
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Dựa vào context được cung cấp: - **Cần gì từ phòng CNTT:** Cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất. - **Ai phê duyệt (laptop 30 triệu):** Không tìm thấy (context chỉ cung cấp thẩm quyền phê duyệt đơn hàng mua sắm dưới 5.000.000 VNĐ).
- **Worst metric:** Context Recall (0.3333)
- **Error Tree:** Output trả lời đúng vế CNTT nhưng thiếu vế thẩm quyền phê duyệt Giám đốc phòng ban (Director) cho hạn mức 30 triệu → Context bị thiếu đoạn bảng phân quyền hạn mức 5-50 triệu → Retrieval chỉ lấy được đoạn dưới 5 triệu và đoạn quy định CNTT.
- **Root cause:** Thẩm quyền phê duyệt chi phí mua sắm được trình bày dưới dạng bảng ma trận phân cấp ngân sách (`Matrix Table`). Khi chunking bằng bộ phân tách văn bản thông thường, bảng bị xé vụn theo hàng hoặc theo kích thước ký tự, làm mất tiêu đề cột ("Cấp phê duyệt") và tiêu đề dòng ("Khoảng ngân sách"), khiến dense vector không phản ánh được mối quan hệ logic "30 triệu nằm trong khoảng 5 đến 50 triệu thì cần Giám đốc phê duyệt".
- **Suggested fix:** Áp dụng **Structure-Aware Chunking chuyên biệt cho Bảng biểu (Table-Aware Chunking)**: Phát hiện cú pháp markdown table, giữ nguyên toàn vẹn bảng trong 1 chunk duy nhất; đồng thời sử dụng Module 5 Enrichment để sinh đoạn diễn giải ngữ nghĩa tự nhiên cho từng dòng của bảng (ví dụ: "Mua sắm từ 5 triệu đến 50 triệu VNĐ do Giám đốc phòng ban phê duyệt").

---

### #5
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ, tương đương **25 triệu VNĐ** (do nghỉ việc sau 8 tháng, trước thời hạn cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học).
- **Worst metric:** Faithfulness (0.40)
- **Error Tree:** Output thực tế hoàn toàn đúng về mặt bản chất so với Ground Truth → RAGAS evaluator chấm điểm Faithfulness thấp (0.40) do xem câu diễn giải điều kiện logic `(do nghỉ việc sau 8 tháng...)` là câu suy luận ngoài context nguyên văn.
- **Root cause:** RAGAS Faithfulness phân tích câu trả lời thành từng mệnh đề atomic facts và đối chiếu xem từng mệnh đề có xuất hiện tường minh (verbatim or direct entailment) trong context hay không. Khi LLM thực hiện tổng hợp logic nhiều bước (multi-hop reasoning), evaluator bằng LLM đôi khi hiểu nhầm bước suy luận logic là thông tin bịa đặt (hallucination).
- **Suggested fix:** Cải tiến prompt sinh phản hồi của hệ thống RAG: Yêu cầu LLM phân chia rõ ràng: (1) Kết luận trực tiếp trích từ tài liệu, (2) Căn cứ điều khoản văn bản; đồng thời cấu hình prompt đánh giá Ragas với Few-shot examples để phân biệt giữa suy diễn toán học/logic hợp lệ và hallucination thực sự.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"*

**Error Tree walkthrough:**
1. **Output đúng?** → **Chưa đầy đủ (Partial Failure)**. Hệ thống trả lời chính xác điều kiện kỹ thuật của phòng CNTT ("Cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất"), nhưng thông báo *"Không tìm thấy"* đối với thẩm quyền phê duyệt cho mức 30 triệu VNĐ.
2. **Context đúng?** → **Context bị thiếu (Retrieval Failure - Low Context Recall = 0.33)**. Context chỉ chứa đoạn văn về thẩm quyền dưới 5 triệu VNĐ và đoạn quy định CNTT, thiếu hẳn chunk chứa dòng hạn mức 5 - 50 triệu VNĐ thuộc về Giám đốc phòng ban.
3. **Query rewrite OK?** → **Chưa tối ưu**. Query chứa câu hỏi kép kết hợp giá trị số cụ thể ("laptop 30 triệu") với thẩm quyền và quy trình IT. Cả BM25 và Dense Search bị chi phối bởi các từ khóa "laptop", "nhân viên mới", "phòng CNTT", lấn át từ khóa bảng hạn mức tài chính.
4. **Fix ở bước:**
   - **M1 (Chunking):** Không xé nhỏ bảng phân cấp thẩm quyền tài chính. Giữ nguyên toàn bộ Matrix Table trong một cấu trúc độc lập.
   - **M5 (Enrichment):** Sinh văn bản mô tả tường minh cho từng khoảng giá trị trong bảng để Dense Search bắt được tương đồng số học ("30 triệu nằm trong ngưỡng 5 - 50 triệu").
   - **M2 (Query Transformation):** Tách thành 2 truy vấn chuyên biệt trước khi tìm kiếm.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Query Transformation & Decomposition Layer:** Tích hợp bộ tiền xử lý tự động phân tách các câu hỏi phức hợp đa chủ đề (multi-part queries) thành các truy vấn đơn lẻ trước khi gửi vào Hybrid Retrieval.
2. **Parent-Document (Small-to-Big) Retrieval:** Index các child chunk được làm giàu (Enriched child chunks) để tối ưu hóa độ chính xác tìm kiếm (Retrieval Precision), nhưng khi tổng hợp câu trả lời sẽ tự động nạp Parent Chunk để LLM có đầy đủ ngữ cảnh tính toán và quy chế toàn cục (Recall & Context Completeness).
3. **Metadata Filtering theo phiên bản hiệu lực:** Tự động lọc phiên bản tài liệu mới nhất (`status: active`, `effective_date`) để triệt tiêu hoàn toàn nhiễu loạn giữa các văn bản sửa đổi, nâng Context Precision lên tuyệt đối.
