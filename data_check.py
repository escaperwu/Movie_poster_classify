# 文件路径
input_file_path = 'C:\\Users\\29235\\Downloads\\Poster_Classifier\\movies.csv'
output_file_path = 'C:\\Users\\29235\\Downloads\\Poster_Classifier\\filtered_movies.csv'

# 导入所需的库
import pandas as pd

# 读取CSV文件
data = pd.read_csv(input_file_path)

# 提取所需的列
filtered_data = data[['genres', 'poster_path', 'title', 'vote_average']]

# 保存到新的CSV文件
filtered_data.to_csv(output_file_path, index=False)

print(f"提取的列已保存到 {output_file_path}")
