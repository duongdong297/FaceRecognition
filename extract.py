import zipfile
import os

def extract_zips_with_original_folder_name(zip_folder_path, extract_to_path):
    """
    Giải nén tất cả các file .zip từ zip_folder_path vào extract_to_path,
    mỗi file zip sẽ được giải nén vào một thư mục con có tên trùng với tên file zip (bỏ phần .zip).

    Args:
        zip_folder_path (str): Đường dẫn đến thư mục chứa các file .zip.
        extract_to_path (str): Đường dẫn đến thư mục đích để giải nén.
    """
    # Tạo thư mục nếu nó chưa tồn tại
    os.makedirs(extract_to_path, exist_ok=True)
    print(f"Đã tạo thư mục đích: {extract_to_path}")

    # Lặp qua tất cả các file trong thư mục zip_folder_path
    for zip_filename in os.listdir(zip_folder_path):
        if zip_filename.lower().endswith('.zip'):
            zip_filepath = os.path.join(zip_folder_path, zip_filename)
            # Lấy tên file zip không có phần mở rộng .zip
            folder_name = os.path.splitext(zip_filename)[0]
            
            # Tạo đường dẫn đầy đủ cho thư mục giải nén
            current_extract_path = os.path.join(extract_to_path, folder_name)
            
            # Tạo thư mục giải nén cho file zip này
            os.makedirs(current_extract_path, exist_ok=True)
            print(f"Đang giải nén '{zip_filename}' vào thư mục '{current_extract_path}'...")

            try:
                with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                    zip_ref.extractall(current_extract_path)
                print(f"Giải nén thành công '{zip_filename}'.")
            except zipfile.BadZipFile:
                print(f"Lỗi: File '{zip_filename}' không phải là file zip hợp lệ hoặc đã bị hỏng.")
            except Exception as e:
                print(f"Đã xảy ra lỗi khi giải nén '{zip_filename}': {e}")
    print("Hoàn tất quá trình giải nén.")


# Thay đổi đường dẫn này thành thư mục chứa các file .zip của bạn
source_zip_folder = 'D:\Video_obs\\buoi 2' 
# Thay đổi đường dẫn này thành thư mục bạn muốn lưu kết quả giải nén
destination_folder = 'D:\Video_obs\\buoi 2\extract'

extract_zips_with_original_folder_name(source_zip_folder, destination_folder)