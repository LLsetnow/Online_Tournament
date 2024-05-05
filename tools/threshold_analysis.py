import argparse
# 该程序用于将 置信度阈值枚举数据 与 数据集实例个数对比 从而计算出适合的置信度区间，并给出推荐置信度（中间值）
# 传入 数据集实例统计
# 传出 推荐阈值范围 阈值
import json
import os.path


def set_threshold_list():
    return {
        'bomb': 0,
        'bridge': 0,
        'safety': 0,
        'cone': 0,
        'crosswalk': 0,
        'danger': 0,
        'evil': 0,
        'block': 0,
        'patient': 0,
        'prop': 0,
        'spy': 0,
        'thief': 0,
        'tumble': 0
    }


my_threshold = {}

# 寻找数量相同（最接近）的置信度区间
def find_threshold_range(count, threshold_data, label, threshold_list):
    start_num = None  # 阈值起点的数量
    start_threshold, end_threshold = None, None

    # 对阈值数据进行排序，确保按照阈值顺序处理
    sorted_thresholds = sorted(threshold_data.items(), key=lambda x: float(x[0]))

    for threshold, num in sorted_thresholds:
        if num >= count:
            if start_num != num:
                start_threshold = threshold  # 设置起始阈值
                start_num = num  # =更新阈值起点的数量
            end_threshold = threshold  # 更新结束阈值
        elif start_threshold is not None:
            # 如果找到起始阈值且当前数值小于count，则终止循环
            break

    # 计算推荐阈值
    if start_threshold and end_threshold:
        recommend_threshold = (float(start_threshold) + float(end_threshold)) / 2
        threshold_list[label] = f"{recommend_threshold:.2f}"
        return f"{start_threshold}-{end_threshold} 推荐阈值：{recommend_threshold:.2f}"
    else:
        return None


def analyze_data(count_file, threshold_file, threshold_analysis, output):
    # 读取数据
    with open(count_file, 'r') as f:
        counts = json.load(f)
    with open(threshold_file, 'r') as f:
        threshold = json.load(f)

    analysis = {'train': {}, 'val': {}, 'all': {}}
    my_threshold = {}
    for item in ['train', 'val', 'all']:
        threshold_list = set_threshold_list()
        for label, count in counts[item].items():
            threshold_range = find_threshold_range(count, threshold[item][label], label, threshold_list)
            if threshold_range:
                analysis[item][label] = threshold_range
        my_threshold[item] = threshold_list


    # 写入分析结果
    with open(threshold_analysis, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=4)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(my_threshold, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='Process some location')

    # 添加参数
    parser.add_argument('--model', dest='model', default='model', help='input your model path', type=str)
    parser.add_argument('--dataset', dest='dataset', help='input your dataset path', type=str)

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行提供的路径
    model_folder = args.model  # 模型路径
    dataset_folder = args.dataset  # 数据集路径

    # 文件路径
    data_count_file = os.path.join(dataset_folder, "count.json")

    threshold_file = os.path.join(model_folder, 'analyse')
    threshold_file = os.path.join(threshold_file, "threshold.json")

    threshold_analysis = os.path.join(model_folder, 'analyse')
    threshold_analysis = os.path.join(threshold_analysis, "threshold_analysis.json")

    my_threshold = os.path.join(model_folder, "analyse")
    my_threshold = os.path.join(my_threshold, "my_threshold.json")

    # 调用函数，传入文件路径
    analyze_data(data_count_file, threshold_file, threshold_analysis, my_threshold)
    print(f"已将置信度阈值分析结果保存至 {my_threshold}")
