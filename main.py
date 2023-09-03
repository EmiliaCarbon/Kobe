import os
from typing import Tuple
from moviepy.editor import VideoFileClip, AudioClip, VideoClip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from io import BytesIO
from PIL import Image
from speed import *


def sample_frames(frame_index, speed, fps, delta_t=0.5) -> Tuple[np.ndarray, int]:
    source_frame = int(fps * delta_t)
    sample_frame = max(int(fps * delta_t * abs(speed)), 1)
    sign = int(np.sign(speed))
    index_list = np.zeros((source_frame,), dtype=np.int32)
    scale = source_frame / sample_frame
    pre_idx = 0
    for i in range(sample_frame):
        idx = min(max(int(i * scale), pre_idx), source_frame - 1)
        index_list[pre_idx: idx + 1] = frame_index + i * sign
        pre_idx = idx + 1
    index_list[pre_idx:] = frame_index + (sample_frame - 1) * sign
    return index_list, index_list[-1] + sign


def sample(video: VideoClip,
           audio: AudioClip,
           speed_func: SpeedFunc) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_t = speed_func.delta_t
    audio_indices, speeds = [], []
    frame_index, clip_idx, frame_count = 0, 0, round(audio.duration * audio.fps)

    while frame_index < frame_count:
        sp = speed_func(clip_idx)
        clip_indices, frame_index = sample_frames(frame_index, sp, audio.fps, delta_t)
        audio_indices.extend(clip_indices)
        speeds.append(sp)
        clip_idx += 1

    scale = audio.fps / video.fps
    video_indices = np.zeros(shape=(int(len(audio_indices) / scale),), dtype=np.int32)
    for idx in range(len(video_indices)):
        video_indices[idx] = int(audio_indices[int(idx * scale)] / scale)

    audio_indices = np.clip(np.array(audio_indices, dtype=np.int32), 0, frame_count - 1)
    video_indices = np.clip(np.array(video_indices, dtype=np.int32), 0, round(video.duration * video.fps) - 1)
    return video_indices, audio_indices, np.array(speeds)


def draw_curves(t, delta_t, speeds, width, height):
    x_min, x_max = t - 2, t + 4
    y_min, y_max = min(speeds) - 0.2, max(speeds) + 0.2
    delta_x, delta_y = width / (x_max - x_min), height / (y_max - y_min)
    curr_speed = speeds[min(int(t / delta_t), len(speeds) - 1)]

    dpi = 60
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot()
    t_range = np.linspace(x_min, x_max, 100)
    y_range = speeds[np.clip((t_range / delta_t).astype(np.int32), 0, len(speeds) - 1)]
    curve = mlines.Line2D(t_range, y_range, linewidth=2, c='b')
    line = mlines.Line2D((t, t), (0, curr_speed), linewidth=2, c='r')
    point = mpatches.Ellipse((t, curr_speed), 10 / delta_x, 10 / delta_y, ec='r', fc='r')
    point.set_zorder(5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Speed')
    ax.add_line(curve)
    ax.add_line(line)
    ax.add_patch(point)
    with BytesIO() as buffer:
        plt.savefig(buffer, format='jpg')
        image = np.asarray(Image.open(buffer))
    plt.close()
    return image


def process(show_speed=True):
    root = r'C:\Users\86139\Desktop\kb'
    video = VideoFileClip(os.path.join(root, 'see you again.mp4'))
    audio = video.audio

    # speed_func = SinSpeed(10, 1, (1, 0.5), delta_t=0.01)
    # speed_func = SinSpeed(10, 0, (3, 1), delta_t=0.01)
    # speed_func = SquareWaveSpeed((2, 1.5, 1), (3, 1.5, 1), delta_t=0.01)
    # speed_func = SawtoothSpeed(10, (1, 2.5), delta_t=0.01)
    # speed_func = RandomSpeed((1, 3), delta_t=0.5)
    # speed_func = ExpSpeed()
    # speed_func = LogSpeed()
    # speed_func = SigmoidSpeed(offset_t=20)
    # speed_func = XSinSpeed(10, 0.05)
    # speed_func = PowerSpeed(2, 0.01)
    # speed_func = TanSpeed()
    speed_func = ArcsinhSpeed()

    video_frames, audio_frames, speeds = sample(video, audio, speed_func)

    def make_frames_audio(t):
        if isinstance(t, int):
            return audio.get_frame(audio_frames[min(t * audio.fps, len(audio_frames) - 1)] / audio.fps)
        frame_indices = np.clip((t * audio.fps).astype(np.int32), 0, len(audio_frames) - 1)
        times = audio_frames[frame_indices] / audio.fps
        return audio.get_frame(times)

    def make_frames_video(t):
        frame = video.get_frame(video_frames[min(int(t * video.fps), len(video_frames) - 1)] / video.fps).copy()
        if show_speed:
            width, height = video.size[0] // 4, video.size[1] // 2
            curve = draw_curves(t, speed_func.delta_t, speeds, height, height)
            frame[-curve.shape[0]:, :curve.shape[1]] = curve
        return frame

    new_audio = AudioClip(
        make_frames_audio,
        duration=len(audio_frames) / audio.fps,
        fps=audio.fps
    )

    new_video = VideoClip(
        make_frames_video,
        duration=len(video_frames) / video.fps
    )
    new_video = new_video.set_audio(new_audio)
    new_video.write_videofile(os.path.join(root, '反双曲正切函数.mp4'), video.fps)


if __name__ == '__main__':
    process()
