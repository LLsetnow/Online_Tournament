import argparse

# 用于将 保存 训练集/测试集 地址的txt文件格式由
# xxx.jpg xxx.xml  -> xxx.jpg 只保留jpg,舍去xml

# 设置命令行参数
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('-i', '--input', type=str, help='Input file path')
parser.add_argument('-o', '--output', type=str, help='Output file path')

# 解析命令行参数
args = parser.parse_args()

# 打开并读取输入文件
with open(args.input, 'r') as file:
    lines = file.readlines()

# 处理每一行，只保留images路径
processed_lines = [line.split(' ')[0] + '\n' for line in lines]

# 将处理后的数据写入到输出文件
with open(args.output, 'w') as outfile:
    outfile.writelines(processed_lines)

print(f"成功将图像路径文件保存至{args.output}")
