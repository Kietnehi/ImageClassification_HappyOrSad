# ImageClassification_HappyOrSad

## Mô tả dự án

Dự án **ImageClassification_HappyOrSad** sử dụng mô hình học sâu (Deep Learning) để phân loại cảm xúc từ các hình ảnh. Cụ thể, mô hình nhận diện hai loại cảm xúc: **Vui** (Happy) và **Buồn** (Sad). Khi người dùng tải lên một bức ảnh, mô hình sẽ phân tích và dự đoán cảm xúc của bức ảnh đó, đồng thời cung cấp tỉ lệ xác suất cho kết quả dự đoán.

## Mục tiêu

Mục tiêu của dự án là xây dựng một ứng dụng web cho phép người dùng tải lên ảnh và nhận kết quả dự đoán cảm xúc từ ảnh đó. Mô hình học sâu được huấn luyện sử dụng một bộ dữ liệu hình ảnh, và sau đó triển khai vào một ứng dụng web đơn giản bằng **Streamlit**.

## Các tính năng

- **Tải ảnh lên**: Người dùng có thể tải lên ảnh bất kỳ dưới các định dạng như `.jpg`, `.png`, `.jpeg`.
- **Dự đoán cảm xúc**: Sau khi tải lên ảnh, mô hình sẽ phân tích và xác định liệu bức ảnh có cảm xúc "Vui" hay "Buồn".
- **Hiển thị kết quả**: Kết quả dự đoán sẽ được hiển thị cùng với tỉ lệ xác suất của mỗi cảm xúc.

## Cách hoạt động

1. **Đầu vào**: Người dùng tải lên ảnh thông qua giao diện web.
2. **Tiền xử lý**: Ảnh được chuẩn hóa và thay đổi kích thước thành một kích thước cố định để phù hợp với mô hình.
3. **Dự đoán**: Mô hình học sâu (CNN) phân loại ảnh thành hai nhóm: **Vui** hoặc **Buồn**.
4. **Hiển thị kết quả**: Kết quả dự đoán được hiển thị trên giao diện web với tỉ lệ xác suất.
## Một số hình ảnh trước và sau khi triển khai web bằng sử dụng thư viện Streamlit
### Ảnh 1
![image](https://github.com/user-attachments/assets/d608e64b-b665-4bf4-83a9-03ea59ea378b)
*Tổng số Parameters bằng cách sử dụng Summary*
### Ảnh 2
![image](https://github.com/user-attachments/assets/8f6a9144-8d0d-4779-a650-e3963db08950)
*Training 50 Epochs*
### Ảnh 3
![image](https://github.com/user-attachments/assets/479256bc-f5b4-4d75-a1c5-c0deda47de78)
*Lossing During Training*
### Ảnh 4
![image](https://github.com/user-attachments/assets/2a035eda-86d7-4b2f-9e97-fef212053550)
*Accuracy During Training*
### Ảnh 5
![image](https://github.com/user-attachments/assets/caf8c7c3-3fc6-41a4-834c-6e67d64645b9)
*Trang web để bỏ ảnh và Predict*





## Các bước triển khai

### Yêu cầu hệ thống

- Python 3.7+
- TensorFlow/Keras
- Streamlit
- OpenCV
- PIL (Pillow)
- Matplotlib
- NumPy

### Cài đặt môi trường

Để cài đặt môi trường và các thư viện cần thiết, bạn có thể sử dụng **pip** hoặc **conda**:

```bash
# Cài đặt các thư viện yêu cầu
pip install -r requirements.txt

Với cách này, người dùng có thể dễ dàng cài đặt tất cả các thư viện cần thiết cho dự án của bạn chỉ bằng một lệnh.
