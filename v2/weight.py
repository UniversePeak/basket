"""# 构建分类器
## 训练集数据结构

```
squat_dataset/
  up/
    image_001.jpg
    image_002.jpg
    ...
  down/
    image_001.jpg
    image_002.jpg
    ...
  ...
```
载入数据集
"""
import utils
import os

# 指定训练集路径
bootstrap_images_in_folder = 'squat_dataset'

# 指定输出文件夹，用于存放处理后的图像和CSV文件
bootstrap_images_out_folder = 'squat_images_out'
bootstrap_csvs_out_folder = 'squat_csvs_out'

# 初始化 BootstrapHelper 类，用于处理训练数据
bootstrap_helper = utils.BootstrapHelper(
    images_in_folder=bootstrap_images_in_folder,
    images_out_folder=bootstrap_images_out_folder,
    csvs_out_folder=bootstrap_csvs_out_folder,
)

# 检查每个动作有多少张图像，打印统计信息
bootstrap_helper.print_images_in_statistics()

# 提取特征，并将特征保存到CSV文件中，per_pose_class_limit参数控制每个姿势类别最多包含多少个样本
bootstrap_helper.bootstrap(per_pose_class_limit=None)

# 检查每个动作有多少张图像提取了特征
bootstrap_helper.print_images_out_statistics()

# 初始引导程序后，未检测到姿势的图像仍会保存在文件夹中（但不在 CSV 中）以用于调试目的。让我们删除它们。
# 确保图像文件和CSV文件的内容一致，删除没有姿态特征的图像
for pose_class_name in bootstrap_helper._pose_class_names:
    csv_path = os.path.join(bootstrap_csvs_out_folder, f'{pose_class_name}.csv')
    print(f"Checking contents of {csv_path}:")
    with open(csv_path, 'r') as csv_file:
        for line in csv_file:
            print(line.strip())
    print("---")
bootstrap_helper.align_images_and_csvs(print_removed_items=True)
bootstrap_helper.print_images_out_statistics()

# 对齐 CSV 与过滤后的图像。
# 再次确保图像文件和CSV文件的内容一致，删除没有姿态特征的图像
for pose_class_name in bootstrap_helper._pose_class_names:
    csv_path = os.path.join(bootstrap_csvs_out_folder, f'{pose_class_name}.csv')
    print(f"Checking contents of {csv_path}:")
    with open(csv_path, 'r') as csv_file:
        for line in csv_file:
            print(line.strip())
    print("---")
bootstrap_helper.align_images_and_csvs(print_removed_items=True)
bootstrap_helper.print_images_out_statistics()

# 将姿态特征转换为嵌入向量
pose_embedder = utils.FullBodyPoseEmbedder()

# 初始化姿态分类器
pose_classifier = utils.PoseClassifier(
    pose_samples_folder=bootstrap_csvs_out_folder,# 姿态特征CSV文件所在的文件夹
    pose_embedder=pose_embedder,# 姿态嵌入器
    top_n_by_max_distance=30,# 基于最大距离选择的前N个样本
    top_n_by_mean_distance=10)# 基于平均距离选择的前N个样本

outliers = pose_classifier.find_pose_sample_outliers()
print('Number of outliers: ', len(outliers))

# 查看所有异常数据点
bootstrap_helper.analyze_outliers(outliers)

# 移除异常数据点
bootstrap_helper.remove_outliers(outliers)

# 重新整理二分类数据
bootstrap_helper.align_images_and_csvs(print_removed_items=False)
bootstrap_helper.print_images_out_statistics()



