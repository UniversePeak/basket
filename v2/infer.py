import utils
import csv
import numpy as np
import cv2
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import tqdm
import os
def dump_for_the_app():#将姿势样本数据从多个CSV文件合并到一个CSV文件中
    pose_samples_folder = 'squat_csvs_out'# 包含姿势样本CSV文件的文件夹
    pose_samples_csv_path = 'squat_csvs_out_basic.csv'# 输出的合并CSV文件路径
    file_extension = 'csv' # 文件扩展名
    file_separator = ','# CSV文件分隔符

    # 获取文件夹中所有CSV文件的名称
    file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]
    # 打开输出CSV文件并写入数据
    with open(pose_samples_csv_path, 'w') as csv_out:
        csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
        for file_name in file_names:
            # 使用文件名作为姿势类别名称
            class_name = file_name[:-(len(file_extension) + 1)]

            # 读取每个输入CSV文件并将数据写入输出CSV文件
            with open(os.path.join(pose_samples_folder, file_name)) as csv_in:
                csv_in_reader = csv.reader(csv_in, delimiter=file_separator)
                for row in csv_in_reader:
                    row.insert(1, class_name) # 在第二列插入类别名称
                    csv_out_writer.writerow(row)

    print(f"CSV file saved as {pose_samples_csv_path}")
# 执行数据合并函数
dump_for_the_app()

# 指定视频路径和输出名称
video_path = 'test2.mp4'
class_name='down'
out_video_path = 'test2-output.mp4'

video_cap = cv2.VideoCapture(video_path)

# Get some video parameters to generate output video with classification.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 包含姿势类别CSV文件的文件夹路径
pose_samples_folder = 'squat_csvs_out'

# 初始化姿势追踪器
pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化姿势嵌入器
pose_embedder = utils.FullBodyPoseEmbedder()

# 初始化姿势分类器
pose_classifier = utils.PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# 初始化EMA平滑处理
pose_classification_filter = utils.EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# 指定动作的两个阈值,始化重复计数器
repetition_counter = utils.RepetitionCounter(
    class_name=class_name,
    enter_threshold=6,
    exit_threshold=4)

# 初始化可视化器
pose_classification_visualizer = utils.PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=video_n_frames,
    # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
    plot_y_max=10)

# 打开输出视频写入器
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

frame_idx = 0
output_frame = None
# 使用tqdm创建进度条
with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
    while True:
        # 读取下一帧
        success, input_frame = video_cap.read()
        if not success:
            break

        # 运行姿势追踪器
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        # 绘制姿势预测
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

        if pose_landmarks is not None:
            # 获取关键点坐标
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # 对当前帧的姿势进行分类
            pose_classification = pose_classifier(pose_landmarks)

            # 使用EMA平滑分类结果
            pose_classification_filtered = pose_classification_filter(pose_classification)

            # 计算重复次数
            repetitions_count = repetition_counter(pose_classification_filtered)
        else:
            # 如果没有检测到姿势,设置相关变量为None或保持当前值
            pose_classification = None

            # Still add empty classification to the filter to maintain correct
            # smoothing for future frames.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            # Don't update the counter presuming that person is 'frozen'. Just
            # take the latest repetitions count.
            repetitions_count = repetition_counter.n_repeats

        # 绘制分类图和重复计数器
        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count)

        # 将输出帧写入视频
        out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

        # Show intermediate frames of the video to track progress.
        # if frame_idx % 50 == 0:
        #     show_image(output_frame)

        # 更新帧索引和进度条
        frame_idx += 1
        pbar.update()

# 关闭输出视频
out_video.release()

# Release MediaPipe resources.
pose_tracker.close()

# Show the last frame of the video.
if output_frame is not None:
    utils.show_image(output_frame)

print(f"Output video saved as {out_video_path}")