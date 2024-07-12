import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pandas as pd
from tqdm import tqdm

# 加载预处理后的数据
file_path = 'filtered_movies.csv'
df = pd.read_csv(file_path)

# 创建保存海报的目录
poster_dir = 'posters'
os.makedirs(poster_dir, exist_ok=True)

# 基础URL
base_url = "https://image.tmdb.org/t/p/original"

# 过滤出有有效poster_path且vote_average大于8的记录
df = df[df['poster_path'].notna() & (df['vote_average'] > 8)]

# 日志列表
download_log = []


# 定义函数：下载和检查图像
def download_poster(row):
    poster_url = base_url + row['poster_path']
    title = row['title']
    save_path = os.path.join(poster_dir, f"{sanitize_filename(title)}.jpg")

    # 如果文件已经存在，则跳过下载
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}, skipping download.")
        download_log.append(f"File already exists: {save_path}, skipping download.")
        return row

    try:
        # 下载图像
        urllib.request.urlretrieve(poster_url, save_path)

        # 检查文件大小是否为零
        if os.path.getsize(save_path) == 0:
            print(f"Downloaded empty file for {title}, skipping.")
            download_log.append(f"Downloaded empty file for {title}, skipping.")
            os.remove(save_path)
            return None

        # 打开并检查图像有效性
        with Image.open(save_path) as img:
            if img.mode not in ['RGB', 'RGBA'] or img.size == (0, 0):
                print(f"Invalid image for {title}, skipping.")
                download_log.append(f"Invalid image for {title}, skipping.")
                os.remove(save_path)
                return None
            # 如果需要调整大小

            img.save(save_path)

        print(f"Downloaded and resized: {save_path}")
        download_log.append(f"Success: {title} - {poster_url}")
        return row  # 返回下载成功的电影信息

    except Exception as e:
        print(f"Failed to download {poster_url}: {e}")
        download_log.append(f"Error: {title} - {poster_url} - {e}")
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                pass
        return None


# 定义函数：清理文件名
def sanitize_filename(filename):
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()


# 并行下载海报
with ThreadPoolExecutor(max_workers=100) as executor:
    results = list(tqdm(executor.map(download_poster, [row for _, row in df.iterrows()]), total=len(df),
                        desc="Downloading posters"))

# 过滤出下载成功的记录
successful_downloads = [result for result in results if result is not None]


# 创建新的DataFrame保存下载成功的电影信息
success_df = pd.DataFrame(successful_downloads)

# 保存下载成功的电影信息到新的CSV文件
success_file_path = 'successful_movies.csv'
success_df.to_csv(success_file_path, index=False)

# 保存下载日志到文件
log_file_path = 'download_log.txt'
with open(log_file_path, 'w') as log_file:
    for log in download_log:
        log_file.write(log + "\n")

print(f"下载成功的电影信息已保存到 {success_file_path}")
print(f"下载日志已保存到 {log_file_path}")
