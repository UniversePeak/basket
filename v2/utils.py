import io
from PIL import ImageFont
from PIL import ImageDraw
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import tqdm
import csv
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

def show_image(img, figsize=(10, 10)):
  """Shows output PIL image."""
  plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.show()

"""# 人体姿态编码"""

class FullBodyPoseEmbedder(object):
  """将3D姿态关键点转换为3D嵌入向量的类。"""

  def __init__(self, torso_size_multiplier=2.5):
    # 初始化FullBodyPoseEmbedder对象。
    # torso_size_multiplier (float): 用于计算最小身体尺寸的躯干尺寸乘数，默认为2.5

    # 用于获取最小身体尺寸的躯干乘数
    self._torso_size_multiplier = torso_size_multiplier


    # 预测结果中出现的关键点名称列表
    self._landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

  def __call__(self, landmarks):
    """规范化姿态关键点并转换为嵌入向量。

    参数:
        landmarks (np.ndarray): 形状为(N, 3)的NumPy数组，包含3D关键点坐标。

        返回:
        np.ndarray: 形状为(M, 3)的NumPy数组，其中M是在_get_pose_distance_embedding中定义的成对距离的数量。
    """
    assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

    # 获取姿态关键点
    landmarks = np.copy(landmarks)

    # 规范化关键点
    landmarks = self._normalize_pose_landmarks(landmarks)

    # 获取嵌入向量
    embedding = self._get_pose_distance_embedding(landmarks)

    return embedding

  def _normalize_pose_landmarks(self, landmarks):
    """规范化关键点的平移和缩放。
    参数:
        landmarks (np.ndarray): 原始的3D关键点坐标
    返回:
        np.ndarray: 规范化后的3D关键点坐标
    """
    landmarks = np.copy(landmarks)

    # 规范化平移
    pose_center = self._get_pose_center(landmarks)
    landmarks -= pose_center

    # 规范化缩放
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # 乘以100不是必需的，但可以使调试更容易
    landmarks *= 100

    return landmarks

  def _get_pose_center(self, landmarks):
    """计算姿态中心点（两髋关节的中点）。
     参数:
        landmarks (np.ndarray): 3D关键点坐标
      返回:
        np.ndarray: 姿态中心点的坐标
    """
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """ 计算姿态尺寸。
        它是以下两个值中的较大者：
        * 躯干尺寸乘以 `torso_size_multiplier`
        * 从姿态中心到任何姿态关键点的最大距离
        参数:
        landmarks (np.ndarray): 3D关键点坐标
        torso_size_multiplier (float): 躯干尺寸乘数
        返回:
        float: 计算得到的姿态尺寸
    """
    # 这种方法仅使用2D关键点来计算姿态尺寸
    landmarks = landmarks[:, :2]

    # 计算髋部中心
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # 计算肩部中心
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # 计算躯干尺寸作为最小身体尺寸
    torso_size = np.linalg.norm(shoulders - hips)

    # 计算到姿态中心的最大距离
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)

  def _get_pose_distance_embedding(self, landmarks):
    """ 将姿态关键点转换为3D嵌入向量。
        我们使用几个成对的3D距离来形成姿态嵌入。所有距离都包括带符号的X和Y分量。
        我们使用不同类型的对来覆盖不同的姿态类别。可以根据需要删除一些或添加新的。
        参数:
        landmarks (np.ndarray): 形状为(N, 3)的NumPy数组，包含3D关键点坐标。
        返回:
        np.ndarray: 形状为(M, 3)的NumPy数组，其中M是成对距离的数量。
    """
    embedding = np.array([
        # 单关节距离

        self._get_distance(
            self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
            self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

        self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

        self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

        # 双关节距离

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

        # 四关节距离

        self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        # 五关节距离

        self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
        self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

        self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        # 跨体距离

        self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
        self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

        self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
        self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

        # 身体弯曲方向

        # self._get_distance(
        #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
        #     landmarks[self._landmark_names.index('left_hip')]),
        # self._get_distance(
        #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
        #     landmarks[self._landmark_names.index('right_hip')]),
    ])

    return embedding

  def _get_average_by_names(self, landmarks, name_from, name_to):
    """
    计算两个指定点的平均位置。
    参数:
    landmarks: 包含所有点坐标的列表或数组
    name_from: 第一个点的名称
    name_to: 第二个点的名称
    返回:
    两个点的平均位置（中点）
    """
    # 获取第一个姿态关键点的坐标
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    # 获取第二个姿态关键点的坐标
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    # 计算并返回两个姿态关键点的平均位置
    return (lmk_from + lmk_to) * 0.5

  def _get_distance_by_names(self, landmarks, name_from, name_to):
    # 计算两个指定姿态关键点之间的距离。
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    return self._get_distance(lmk_from, lmk_to)

  def _get_distance(self, lmk_from, lmk_to):
    # 返回两个姿态关键点之间的距离（向量差）
    return lmk_to - lmk_from

"""# 人体姿态分类"""

class PoseSample(object):

  def __init__(self, name, landmarks, class_name, embedding):
    """
    初始化 PoseSample 对象。
    参数:
    name: 样本的名称
    landmarks: 姿态的姿态关键点坐标
    class_name: 姿态的类别名称
    embedding: 姿态的嵌入表示
    """
    self.name = name
    self.landmarks = landmarks
    self.class_name = class_name

    self.embedding = embedding

class PoseSampleOutlier(object):
  """
  异常清除。
  """

  def __init__(self, sample, detected_class, all_classes):
    """
    初始化 PoseSampleOutlier 对象。
    参数:
    sample: 异常的姿态样本
    detected_class: 检测到的姿态类别
    all_classes: 所有可能的姿态类别
    """
    self.sample = sample
    self.detected_class = detected_class
    self.all_classes = all_classes

class PoseClassifier(object):
  """
  姿态分类器类，用于对姿态姿态关键点进行分类。
  """

  def __init__(self,
               pose_samples_folder,
               pose_embedder,
               file_extension='csv',
               file_separator=',',
               n_landmarks=33,
               n_dimensions=3,
               top_n_by_max_distance=30,
               top_n_by_mean_distance=10,
               axes_weights=(1., 1., 0.2)):
    """
    初始化 PoseClassifier 对象。
    参数:
    pose_samples_folder: 包含姿态样本的文件夹路径
    pose_embedder: 用于生成姿态嵌入的函数或对象
    file_extension: 样本文件的扩展名，默认为 'csv'
    file_separator: CSV 文件的分隔符，默认为 ','
    n_landmarks: 姿态关键点的数量，默认为 33
    n_dimensions: 每个姿态关键点的维度，默认为 3 (x, y, z)
    top_n_by_max_distance: 基于最大距离选择的样本数量，默认为 30
    top_n_by_mean_distance: 基于平均距离选择的样本数量，默认为 10
    axes_weights: 各轴的权重，默认为 (1., 1., 0.2)
    """
    self._pose_embedder = pose_embedder
    self._n_landmarks = n_landmarks
    self._n_dimensions = n_dimensions
    self._top_n_by_max_distance = top_n_by_max_distance
    self._top_n_by_mean_distance = top_n_by_mean_distance
    self._axes_weights = axes_weights

    self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                 file_extension,
                                                 file_separator,
                                                 n_landmarks,
                                                 n_dimensions,
                                                 pose_embedder)

  def _load_pose_samples(self,
                         pose_samples_folder,
                         file_extension,
                         file_separator,
                         n_landmarks,
                         n_dimensions,
                         pose_embedder):
    """从给定文件夹加载姿态样本。
    Required folder structure:
      neutral_standing.csv
      pushups_down.csv
      pushups_up.csv
      squats_down.csv
      ...

    Required CSV structure:
      sample_00001,x1,y1,z1,x2,y2,z2,....
      sample_00002,x1,y1,z1,x2,y2,z2,....
      ...
            参数:
        pose_samples_folder: 包含姿态样本的文件夹路径
        file_extension: 样本文件的扩展名
        file_separator: CSV 文件的分隔符
        n_landmarks: 姿态关键点的数量
        n_dimensions: 每个姿态关键点的维度
        pose_embedder: 用于生成姿态嵌入的函数或对象

        返回:
        包含所有加载的姿态样本的列表
    """
    # 获取文件夹中所有指定扩展名的文件
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

    pose_samples = []
    for file_name in file_names:
      # 使用文件名作为姿态类别名称
      class_name = file_name[:-(len(file_extension) + 1)]

      # 解析 CSV 文件
      with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
          # 确保每行的值的数量正确
          assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
          # 将姿态关键点数据转换为 NumPy 数组
          landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
          # 创建并添加 PoseSample 对象
          pose_samples.append(PoseSample(
              name=row[0],
              landmarks=landmarks,
              class_name=class_name,
              embedding=pose_embedder(landmarks),
          ))

    return pose_samples

  def find_pose_sample_outliers(self):
    """
    对每个样本进行分类，找出异常样本。

    返回:
    异常样本列表
    """
    outliers = []
    for sample in self._pose_samples:
      # 对目标姿态进行分类
      pose_landmarks = sample.landmarks.copy()
      pose_classification = self.__call__(pose_landmarks)
      # 获取得分最高的类别
      class_names = [class_name for class_name, count in pose_classification.items() if count == max(pose_classification.values())]

      # 如果最近的姿态类别与样本的类别不同，或者有多个最近的类别，则认为是异常样本
      if sample.class_name not in class_names or len(class_names) != 1:
        outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))

    return outliers

  def __call__(self, pose_landmarks):
    """
    对给定的姿态进行分类。

    分类分两个阶段进行：
    1. 首先根据最大距离选择 top-N 个样本，以去除与给定姿态几乎相同但某些关节弯曲方向不同的样本。
    2. 然后根据平均距离选择 top-N 个样本。在第一步去除异常值后，我们可以选择平均距离最近的样本。

    参数:
    pose_landmarks: 形状为 (N, 3) 的 NumPy 数组，包含 3D 姿态关键点坐标。

    返回:
    字典，包含数据库中最接近的姿态样本的计数。例如：
    {
      'pushups_down': 8,
      'pushups_up': 2,
    }
    """
    # 检查提供的姿态和目标姿态是否具有相同的形状
    assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(pose_landmarks.shape)

    # 获取给定姿态的嵌入表示
    pose_embedding = self._pose_embedder(pose_landmarks)
    # 获取水平翻转后的姿态嵌入表示
    flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

    # 根据最大距离进行过滤
    # 这有助于去除异常值 - 与给定姿态几乎相同但某个关节弯曲方向不同的姿态，
    # 这些姿态实际上代表了不同的姿态类别。
    max_dist_heap = []
    for sample_idx, sample in enumerate(self._pose_samples):
      # 计算当前样本与给定姿态（包括翻转后的姿态）的最大距离
      max_dist = min(
          np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
          np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
      )
      max_dist_heap.append([max_dist, sample_idx])
    # 根据最大距离对样本进行排序，并选择 top-N 个
    max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
    max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

    # 根据平均距离进行过滤
    # 在去除异常值后，我们可以通过平均距离找到最近的姿态。
    mean_dist_heap = []
    for _, sample_idx in max_dist_heap:
      sample = self._pose_samples[sample_idx]
      # 计算当前样本与给定姿态（包括翻转后的姿态）的平均距离
      mean_dist = min(
          np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
          np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
      )
      mean_dist_heap.append([mean_dist, sample_idx])
    # 根据平均距离对样本进行排序，并选择 top-N 个
    mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
    mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

    # 将结果收集到字典中：(class_name -> n_samples)
    class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
    result = {class_name: class_names.count(class_name) for class_name in set(class_names)}

    return result

"""# 姿态分类结果平滑"""

class EMADictSmoothing(object):
  """Smoothes pose classification."""

  def __init__(self, window_size=10, alpha=0.2):
    self._window_size = window_size
    self._alpha = alpha

    self._data_in_window = []

  def __call__(self, data):
    """Smoothes given pose classification.

    Smoothing is done by computing Exponential Moving Average for every pose
    class observed in the given time window. Missed pose classes arre replaced
    with 0.

    Args:
      data: Dictionary with pose classification. Sample:
          {
            'pushups_down': 8,
            'pushups_up': 2,
          }

    Result:
      Dictionary in the same format but with smoothed and float instead of
      integer values. Sample:
        {
          'pushups_down': 8.3,
          'pushups_up': 1.7,
        }
    """
    # Add new data to the beginning of the window for simpler code.
    self._data_in_window.insert(0, data)
    self._data_in_window = self._data_in_window[:self._window_size]

    # Get all keys.
    keys = set([key for data in self._data_in_window for key, _ in data.items()])

    # Get smoothed values.
    smoothed_data = dict()
    for key in keys:
      factor = 1.0
      top_sum = 0.0
      bottom_sum = 0.0
      for data in self._data_in_window:
        value = data[key] if key in data else 0.0

        top_sum += factor * value
        bottom_sum += factor

        # Update factor.
        factor *= (1.0 - self._alpha)

      smoothed_data[key] = top_sum / bottom_sum

    return smoothed_data

"""# 动作计数器"""

class RepetitionCounter(object):
  """计算给定目标姿态类别的重复次数。"""

  def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
    self._class_name = class_name

    # 如果姿态计数器超过给定阈值，则我们进入该姿态
    self._enter_threshold = enter_threshold
    self._exit_threshold = exit_threshold

    # 表示我们是否处于给定姿态中
    self._pose_entered = False

    # 记录我们退出姿态的次数
    self._n_repeats = 0

  @property
  def n_repeats(self):
    return self._n_repeats

  def __call__(self, pose_classification):
    """计算直到给定帧发生的重复次数。
    我们使用两个阈值。首先需要超过较高的阈值才能进入姿态，
    然后需要低于较低的阈值才能退出姿态。两个阈值之间的差异
    使其对预测抖动保持稳定（如果只有一个阈值，会导致错误计数）。
    Args:
      pose_classification: 当前帧的姿态分类字典。
        样例:
          {
            'pushups_down': 8.3,
            'pushups_up': 1.7,
          }
    Returns:
      重复次数的整数计数器。
    """
    # 获取姿态置信度
    pose_confidence = 0.0
    if self._class_name in pose_classification:
      pose_confidence = pose_classification[self._class_name]

    # 在第一帧或者如果我们不在姿态中，只需检查我们是否在这一帧进入了姿态并更新状态
    if not self._pose_entered:
      self._pose_entered = pose_confidence > self._enter_threshold
      return self._n_repeats

    # 如果我们在姿态中并且正在退出，则增加计数器并更新状态
    if pose_confidence < self._exit_threshold:
      self._n_repeats += 1
      self._pose_entered = False

    return self._n_repeats

"""# 可视化模块"""

class PoseClassificationVisualizer(object):
  """跟踪每一帧的分类并渲染它们。"""

  def __init__(self,
               class_name,
               plot_location_x=0.05,
               plot_location_y=0.05,
               plot_max_width=0.4,
               plot_max_height=0.4,
               plot_figsize=(9, 4),
               plot_x_max=None,
               plot_y_max=None,
               counter_location_x=0.85,
               counter_location_y=0.05,
               counter_font_path='fonts\Roboto-Regular.ttf',  # 修改字体路径
               counter_font_color='red',
               counter_font_size=0.15):
    self._class_name = class_name
    self._plot_location_x = plot_location_x
    self._plot_location_y = plot_location_y
    self._plot_max_width = plot_max_width
    self._plot_max_height = plot_max_height
    self._plot_figsize = plot_figsize
    self._plot_x_max = plot_x_max
    self._plot_y_max = plot_y_max
    self._counter_location_x = counter_location_x
    self._counter_location_y = counter_location_y
    self._counter_font_path = counter_font_path
    self._counter_font_color = counter_font_color
    self._counter_font_size = counter_font_size

    self._counter_font = None

    self._pose_classification_history = []
    self._pose_classification_filtered_history = []

  def __call__(self,
               frame,
               pose_classification,
               pose_classification_filtered,
               repetitions_count):
    """渲染给定帧的姿态分类和计数器。"""
    # 扩展分类历史
    self._pose_classification_history.append(pose_classification)
    self._pose_classification_filtered_history.append(pose_classification_filtered)

    # 输出带有分类图和计数器的帧
    output_img = Image.fromarray(frame)

    output_width = output_img.size[0]
    output_height = output_img.size[1]

    # 绘制图表
    img = self._plot_classification_history(output_width, output_height)

    # 使用新的重采样方法
    try:
      resampling_method = Image.Resampling.LANCZOS
    except AttributeError:
      resampling_method = Image.ANTIALIAS

    img.thumbnail((int(output_width * self._plot_max_width),
                   int(output_height * self._plot_max_height)),
                  resampling_method)

    output_img.paste(img,
                     (int(output_width * self._plot_location_x),
                      int(output_height * self._plot_location_y)))

    # 绘制计数
    output_img_draw = ImageDraw.Draw(output_img)
    if self._counter_font is None:
      font_size = int(output_height * self._counter_font_size)
      # 从本地加载字体文件
      self._counter_font = ImageFont.truetype(self._counter_font_path, size=font_size)
    output_img_draw.text((output_width * self._counter_location_x,
                          output_height * self._counter_location_y),
                         str(repetitions_count),
                         font=self._counter_font,
                         fill=self._counter_font_color)

    return output_img

  def _plot_classification_history(self, output_width, output_height):
    """绘制分类历史图表"""
    fig = plt.figure(figsize=self._plot_figsize)

    for classification_history in [self._pose_classification_history,
                                   self._pose_classification_filtered_history]:
      y = []
      for classification in classification_history:
        if classification is None:
          y.append(None)
        elif self._class_name in classification:
          y.append(classification[self._class_name])
        else:
          y.append(0)
      plt.plot(y, linewidth=7)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Frame')
    plt.ylabel('Confidence')
    plt.title('Classification history for `{}`'.format(self._class_name))
    plt.legend(loc='upper right')

    if self._plot_y_max is not None:
      plt.ylim(top=self._plot_y_max)
    if self._plot_x_max is not None:
      plt.xlim(right=self._plot_x_max)

    # 将图表转换为图像
    buf = io.BytesIO()
    dpi = min(
        output_width * self._plot_max_width / float(self._plot_figsize[0]),
        output_height * self._plot_max_height / float(self._plot_figsize[1]))
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img

"""# 提取训练集关键点坐标"""

class BootstrapHelper(object):
  """帮助引导图像并过滤姿势样本以进行分类。"""

  def __init__(self,
               images_in_folder,
               images_out_folder,
               csvs_out_folder):
    self._images_in_folder = images_in_folder
    self._images_out_folder = images_out_folder
    self._csvs_out_folder = csvs_out_folder

    # 获取姿势类别列表并打印图像统计信息。
    self._pose_class_names = sorted([n for n in os.listdir(self._images_in_folder) if not n.startswith('.')])

  def bootstrap(self, per_pose_class_limit=None):
    """引导给定文件夹中的图像。

        所需的输入图像文件夹结构（输出图像文件夹也使用相同结构）:
          pushups_up/
            image_001.jpg
            image_002.jpg
            ...
          pushups_down/
            image_001.jpg
            image_002.jpg
            ...
          ...

        生成的CSV输出文件夹:
          pushups_up.csv
          pushups_down.csv

        生成的CSV结构,包含3D姿势关键点:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
        """
    # 创建CSV输出文件夹。
    if not os.path.exists(self._csvs_out_folder):
      os.makedirs(self._csvs_out_folder)

    for pose_class_name in self._pose_class_names:
      print('Bootstrapping ', pose_class_name, file=sys.stderr)

      # 姿势类的路径。
      images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
      images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + '.csv')
      if not os.path.exists(images_out_folder):
        os.makedirs(images_out_folder)

      with open(csv_out_path, 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        # 获取图像列表。
        image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
        if per_pose_class_limit is not None:
          image_names = image_names[:per_pose_class_limit]

        # 引导每张图像。
        for image_name in tqdm.tqdm(image_names):
          # 加载图像。
          input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
          input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

          # 初始化新的姿势跟踪器并运行它。
          with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_tracker:
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

          # 保存带有姿势预测的图像（如果检测到姿势）。
          output_frame = input_frame.copy()
          if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)
          output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

          # 如果检测到姿势，保存关键点。
          if pose_landmarks is not None:
            # 获取关键点。
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array(
                [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                 for lmk in pose_landmarks.landmark],
                dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
            csv_out_writer.writerow([image_name] + pose_landmarks.flatten().astype(str).tolist())

          # 绘制XZ投影并与图像连接。
          projection_xz = self._draw_xz_projection(
              output_frame=output_frame, pose_landmarks=pose_landmarks)
          output_frame = np.concatenate((output_frame, projection_xz), axis=1)

  def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
    """绘制姿势关键点的XZ平面投影。"""
    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
    img = Image.new('RGB', (frame_width, frame_height), color='white')

    if pose_landmarks is None:
      return np.asarray(img)

    # 根据图像宽度缩放半径。
    r *= frame_width * 0.01

    draw = ImageDraw.Draw(img)
    for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
      # 翻转Z并将臀部中心移动到图像中心。
      x1, y1, z1 = pose_landmarks[idx_1] * [1, 1, -1] + [0, 0, frame_height * 0.5]
      x2, y2, z2 = pose_landmarks[idx_2] * [1, 1, -1] + [0, 0, frame_height * 0.5]

      draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
      draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
      draw.line([x1, z1, x2, z2], width=int(r), fill=color)

    return np.asarray(img)

  def align_images_and_csvs(self, print_removed_items=False):
    """确保图像文件夹和CSV具有相同的样本。

    仅保留图像文件夹和CSV中样本的交集。
    """
    for pose_class_name in self._pose_class_names:
      # 姿势类的路径。
      images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder, f'{pose_class_name}.csv')

      # 将CSV读入内存。
      rows = []
      with open(csv_out_path, 'r') as csv_out_file:
        csv_out_reader = csv.reader(csv_out_file, delimiter=',')
        for row in csv_out_reader:
          if len(row) > 0:  # 只添加非空行
            rows.append(row)

      # CSV中存在的图像名称。
      image_names_in_csv = []
      for row in rows:
        try:
          image_name = row[0]
          image_names_in_csv.append(image_name)
        except IndexError:
          print(f"Warning: Skipping invalid row in CSV: {row}")

      # 图像文件夹中存在的图像名称。
      image_names_in_folder = os.listdir(images_out_folder)

      # 删除所有不在CSV中的图像。
      for image_name in image_names_in_folder:
        if image_name not in image_names_in_csv:
          os.remove(os.path.join(images_out_folder, image_name))
          if print_removed_items:
            print(f'Removed image from folder: {image_name}')

      # 删除所有不在图像文件夹中的CSV行。
      rows_to_keep = []
      for row in rows:
        try:
          image_name = row[0]
          if image_name in image_names_in_folder:
            rows_to_keep.append(row)
          elif print_removed_items:
            print(f'Removed image from CSV: {image_name}')
        except IndexError:
          print(f"Warning: Skipping invalid row in CSV: {row}")

      # 覆写CSV。
      with open(csv_out_path, 'w', newline='') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',')
        for row in rows_to_keep:
          csv_out_writer.writerow(row)

  def analyze_outliers(self, outliers):
    """对每个样本与所有其他样本进行分类以找出异常值。

    如果样本的分类与原始类别不同 - 应该删除它或添加更多相似的样本。
    """
    for outlier in outliers:
      image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)

      print('Outlier')
      print('  sample path =    ', image_path)
      print('  sample class =   ', outlier.sample.class_name)
      print('  detected class = ', outlier.detected_class)
      print('  all classes =    ', outlier.all_classes)

      img = cv2.imread(image_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      # show_image(img, figsize=(20, 20))

  def remove_outliers(self, outliers):
    """从图像文件夹中删除异常值。"""
    for outlier in outliers:
      image_path = os.path.join(self._images_out_folder, outlier.sample.class_name, outlier.sample.name)
      os.remove(image_path)

  def print_images_in_statistics(self):
    """打印输入图像文件夹的统计信息。"""
    self._print_images_statistics(self._images_in_folder, self._pose_class_names)

  def print_images_out_statistics(self):
    """打印输出图像文件夹的统计信息。"""
    self._print_images_statistics(self._images_out_folder, self._pose_class_names)

  def _print_images_statistics(self, images_folder, pose_class_names):
    """打印每个姿势类别的图像数量。"""
    print('Number of images per pose class:')
    for pose_class_name in pose_class_names:
      n_images = len([
          n for n in os.listdir(os.path.join(images_folder, pose_class_name))
          if not n.startswith('.')])
      print('  {}: {}'.format(pose_class_name, n_images))