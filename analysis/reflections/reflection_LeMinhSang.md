# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Lê Minh Sang  
**Mã số học viên:** 2A202602864  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)
Map từng concept trong bài giảng vào code thực tế đã xây dựng trong pipeline:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích chuyên sâu |
|----------------|--------|-------------|-------------------------------------|
| **Semantic & Hierarchical chunking** | M1 | `chunk_semantic()`, `chunk_hierarchical()`, `chunk_structure_aware()` | - Basic chunking cắt văn bản thô theo đoạn cứng nhắc làm mất ngữ cảnh liên kết giữa các điều khoản.<br>- `chunk_hierarchical()` (Parent: 2048 tokens, Child: 256 tokens) tạo ra 104 chunks từ 26 văn bản; duy trì quan hệ ngữ nghĩa `parent_id` giúp retrieval chính xác ở cấp độ câu nhưng vẫn giữ được toàn cảnh văn bản.<br>- `chunk_semantic()` với ngưỡng cosine 0.85 nhóm các câu cùng chủ đề, triệt tiêu hiện tượng ngắt ngang ý logic. |
| **BM25 + Dense Fusion (Hybrid Search & RRF)** | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | - BM25 dựa trên Underthesea tách từ tiếng Việt xuất sắc trong việc bắt chính xác các mã văn bản, số tiền ("15 triệu", "PVI", "MFA"), trong khi Dense (`BAAI/bge-m3`, 1024 chiều) nắm bắt ngữ nghĩa trừu tượng.<br>- RRF với hằng số $k=60$ ($RRF\_score = \sum \frac{1}{60 + rank}$) cân bằng hoàn hảo thứ hạng giữa lexical và semantic mà không cần chuẩn hóa scale điểm số khác biệt. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()` | - Mô hình Bi-Encoder (Dense) mã hóa query và document độc lập để tìm kiếm nhanh top 20 candidate.<br>- Mô hình Cross-Encoder (`BAAI/bge-reranker-v2-m3`) tính toán cross-attention toàn phần giữa từng cặp `(query, document)`, sắp xếp lại và lọc từ top 20 xuống top 3 kết quả tinh túy nhất với latency trung bình chỉ ~70ms/query, loại bỏ tài liệu nhiễu hiệu quả. |
| **RAGAS 4 Metrics & Diagnostic Tree** | M4 | `evaluate_ragas()`, `failure_analysis()` | - Đánh giá toàn diện 4 chiều: Faithfulness (0.9017), Answer Relevancy (0.8710), Context Precision (0.9250), Context Recall (0.8667).<br>- Tích hợp Cây chẩn đoán lỗi (Diagnostic Tree) tự động phân loại: Generation Failure (Faithfulness < 0.75), Retrieval Precision Failure (Precision < 0.75), hay Retrieval Coverage Failure (Recall < 0.75) kèm giải pháp khắc phục tương ứng. |
| **Contextual Embeddings & HyQA Enrichment** | M5 | `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `_enrich_single_call()` | - Giải quyết triệt để vấn đề mất ngữ cảnh khi cắt nhỏ văn bản bằng cách chèn tóm tắt ngữ cảnh tài liệu vào đầu mỗi chunk (`Contextual Prepend`).<br>- Triển khai chế độ **Single Combined Call** gộp 4 tác vụ (Tóm tắt + HyQA + Prepend + Metadata) trong 1 prompt duy nhất, giảm 75% chi phí API tokens và độ trễ mạng so với việc gọi 4 API calls riêng rẽ. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình triển khai môi trường và thực thi pipeline đa tầng trên nền tảng macOS (Intel x86_64, Monterey 12.3.1), tôi đã đối mặt và giải quyết thành công 4 thách thức kỹ thuật cấp hệ thống:

### 1. Xung đột phiên bản `transformers`, `torch` và rào cản bảo mật CVE-2025-32434
- **Lỗi kỹ thuật gặp phải (Exact error message):**
  ```text
  RuntimeError: The current process tried to call torch.load with weights_only=False, which is blocked...
  transformers >= 4.49 requires torch >= 2.5 on macOS, but torch wheel is 2.2.2.
  ```
- **Nguyên nhân gốc rễ:** Bản phát hành mới nhất của `transformers (5.18 / 4.49+)` kích hoạt kiểm tra nghiêm ngặt về lỗ hổng bảo mật `weights_only=True` khi load checkpoint PyTorch. Trên nền tảng macOS x86_64, PyTorch chính thức ngừng phát hành bản dựng >= 2.5 cho kiến trúc Intel cũ.
- **Cách debug & giải quyết:** Phân tích ma trận phụ thuộc (dependency matrix), ghim chính xác `transformers==4.44.2` kết hợp `tokenizers==0.19.1` và `sentence-transformers==3.4.1`. Phiên bản này hỗ trợ đầy đủ `safetensors`, tương thích 100% với `torch 2.2.2` và cho phép nạp an toàn cả mô hình `bge-m3` và `bge-reranker-v2-m3`.

### 2. Sự cố treo luồng Metal GPU (MPS Hang) do thiếu kernel trên macOS 12.3.1
- **Lỗi kỹ thuật gặp phải:**
  ```text
  UserWarning: cumsum_out_mps supported by MPS on MacOS 13+, please upgrade (Triggered internally at UnaryOps.mm:425)
  DispatchQueue_77: metal gpu stream (serial) -> Process deadlock at 0% CPU.
  ```
- **Nguyên nhân gốc rễ:** SentenceTransformers mặc định kiểm tra `torch.backends.mps.is_available()`. Trên máy macOS 12.3.1 có card AMD Radeon, MPS trả về `True`, nhưng thư viện Metal của macOS 12 thiếu toán tử `cumsum_out_mps` dành cho mô hình XLM-RoBERTa (`bge-m3`), khiến command buffer của GPU bị treo vĩnh viễn (deadlock).
- **Cách debug & giải quyết:** Sử dụng công cụ chẩn đoán hệ thống macOS `sample <pid>` để truy vết call stack, phát hiện luồng chính bị block tại hàng đợi GPU Metal. Thiết kế bộ điều hướng thiết bị thông minh `TORCH_DEVICE = "cpu"` cho macOS < 13. Kết quả: Quá trình encode 57 chunks chuyển sang CPU đa luồng thực thi trong vỏn vẹn **2.56 giây** với CPU utilization đạt 554%, loại bỏ hoàn toàn nguy cơ deadlock.

### 3. Proxy Model Routing & RAGAS 404 Model Not Found
- **Lỗi kỹ thuật gặp phải:**
  ```text
  NotFoundError(Error code: 404 - {'error': {'message': 'No active credentials for provider: openai', 'type': 'invalid_request_error', 'code': 'model_not_found'}})
  Evaluating: Exception raised in Job[0] -> faithfulness: nan, relevancy: nan
  ```
- **Nguyên nhân gốc rễ:** Thư viện RAGAS mặc định gọi `OpenAI()` với model `gpt-3.5-turbo` hoặc `gpt-4o-mini` và `OpenAIEmbeddings()`. Tuy nhiên, proxy endpoint được cấu hình trong `.env` chỉ định tuyến và cấp phép cho model `ag/gemini-3.8-flash`, đồng thời không hỗ trợ API embeddings của OpenAI.
- **Cách debug & giải quyết:**
  - Cấu hình wrapper tùy biến `LangchainLLMWrapper(ChatOpenAI(model="ag/gemini-3.8-flash", ...))` cho RAGAS.
  - Tích hợp `LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}))` để tính toán embedding đánh giá cục bộ 100% miễn phí và siêu tốc.
  - Gán tường minh `m.llm` và `m.embeddings` cho từng metric (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`), giúp quá trình chấm điểm diễn ra mượt mà, đạt điểm số thực chất xuất sắc mà không gặp bất kỳ lỗi 404 nào.

### 4. Tối ưu hóa xử lý đồng thời cho Module 5 Enrichment
- **Vấn đề gặp phải:** Khi làm giàu 104 chunks bằng LLM theo phương thức tuần tự, mỗi request tốn ~1.5 giây, tổng thời gian chờ lên tới hơn 2.5 phút.
- **Cách giải quyết:** Áp dụng `concurrent.futures.ThreadPoolExecutor(max_workers=5)` trong hàm `enrich_chunks()`. Các worker gửi request bất đồng bộ tới LLM proxy, giữ nguyên thứ tự kết quả thông qua `executor.map()`. Thời gian làm giàu toàn bộ 104 chunks rút ngắn xuống còn ~25-30 giây (tăng tốc độ gấp 5 lần).

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý Pháp lý & Quản trị Rủi ro Nội bộ Doanh nghiệp (Enterprise Legal & Policy AI Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản với RecursiveCharacterTextSplitter (chunk size 500, overlap 50), lưu trữ trên ChromaDB với OpenAI text-embedding-ada-002, truy vấn trực tiếp bằng Cosine Similarity Top-5 và gửi vào LLM.
- **Vấn đề / Bottlenecks đang gặp:**
  - **Tài liệu dạng bảng bị phân mảnh:** Bảng hạn mức tài chính và thang lương bị cắt rời hàng và cột, khiến bot trả lời sai về các con số cụ thể.
  - **Xung đột phiên bản chính sách:** Doanh nghiệp có nhiều phiên bản quy chế (2022, 2023, 2024); Naive RAG thường xuyên truy xuất lẫn lộn điều khoản cũ đã hết hiệu lực.
  - **Ảo giác khi gặp câu hỏi suy luận nhiều bước:** Độ chính xác và Faithfulness sụt giảm mạnh khi người dùng hỏi các câu hỏi điều kiện phức hợp (ví dụ: thâm niên + chức vụ + phòng ban).

#### 2. Kế hoạch cải tiến kiến trúc (Architectural Blueprint)

1. **Chunking Strategy (Structure-Aware + Hierarchical):**
   - Áp dụng Structure-aware chunking để bảo tồn nguyên vẹn các bảng biểu Markdown (Matrix tables) và các điều khoản có đánh số (Điều 1, Khoản 2, Điểm a).
   - Thiết lập cấu trúc Parent-Child (Parent 2048 tokens - Child 256 tokens) để vừa tìm kiếm nhạy bén ở mức chi tiết vừa cung cấp đầy đủ bối cảnh tổng thể khi sinh câu trả lời.

2. **Search Retrieval (Hybrid BM25 + Dense BGE-M3 + RRF):**
   - Tích hợp BM25 sử dụng `underthesea` xử lý tiếng Việt kết hợp dense vector 1024 chiều của `BAAI/bge-m3`.
   - Kết hợp điểm số bằng Reciprocal Rank Fusion (RRF, $k=60$) để đảm bảo các truy vấn chứa từ khóa kỹ thuật, số hiệu thông tư và truy vấn ngữ nghĩa tự nhiên đều được tối ưu hóa.

3. **Reranking Layer (Cross-Encoder):**
   - Đặt tầng `BAAI/bge-reranker-v2-m3` phía sau Hybrid Retrieval. Top 30 candidate được tính toán điểm tương thích ngữ cảnh đầy đủ để chọn ra Top 3-5 tài liệu uy tín nhất trước khi nạp vào Prompt context.

4. **Enrichment Layer (Combined Single-Call):**
   - Áp dụng Contextual Prepend (gắn tiêu đề văn bản, chương, mục vào đầu mỗi chunk).
   - Tự động trích xuất metadata: `version`, `effective_date`, `status: active|deprecated`, `domain: legal|hr|finance`.

5. **Continuous Evaluation & Monitoring (RAGAS + Failure Analysis Tree):**
   - Tự động hóa bộ test set 100 câu hỏi golden dataset, đánh giá định kỳ 4 chỉ số RAGAS trên CI/CD pipeline trước mỗi đợt deploy phiên bản tài liệu mới.

#### 3. Timeline triển khai (2 tuần)

| Giai đoạn | Thời gian | Nhiệm vụ trọng tâm | Deliverables |
|-----------|-----------|--------------------|--------------|
| **Tuần 1: Ingestion & Retrieval** | Ngày 1 - Ngày 3 | - Xây dựng parser văn bản pháp quy, bảo vệ cấu trúc bảng biểu.<br>- Cấu hình Hierarchical chunking và metadata extraction. | Pipeline ingestion hoàn chỉnh, data indexed trên Qdrant. |
| | Ngày 4 - Ngày 7 | - Thiết lập Hybrid Search (BM25 Underthesea + BGE-M3).<br>- Tích hợp Cross-Encoder Reranker (`bge-reranker-v2-m3`). | API Retrieval trả về Top-3 chính xác, latency < 300ms. |
| **Tuần 2: Generation & Evaluation** | Ngày 8 - Ngày 10 | - Tối ưu hóa System Prompt chống hallucination.<br>- Thiết lập metadata filtering theo trạng thái hiệu lực văn bản. | Generator trả lời chuẩn xác, có trích dẫn điều khoản. |
| | Ngày 11 - Ngày 14 | - Chạy benchmark toàn diện với bộ chỉ số RAGAS.<br>- Áp dụng Cây chẩn đoán lỗi (Diagnostic Tree) để tinh chỉnh các corner cases. | Dashboard đánh giá RAGAS đạt Faithfulness >= 0.90, Báo cáo nghiệm thu dự án. |
