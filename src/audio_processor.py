import time
import pyaudio
import torchaudio
import wave
import numpy as np
import threading
from matplotlib import pyplot as plt
from pydub import AudioSegment
import speech_recognition as sr
import os
from demucs import pretrained
from demucs.apply import apply_model
import torch
import logging
from sympy.physics.control.control_plots import matplotlib


class AudioProcessor:
		def __init__(self, log_file='audio_processor.log'):
				self.is_recording = False
				self.input_language = 'en'

				# 配置日志记录
				logging.basicConfig(filename=log_file, level=logging.INFO,
									format='%(asctime)s - %(levelname)s - %(message)s')
				self.logger = logging.getLogger(__name__)

		def record_audio(self, filename,
						 sample_rate=44100,
						 chunk_size=1024,
						 silence_threshold=2,
						 silence_duration=4):
				self.is_recording = True
				audio_format = pyaudio.paInt16
				channels = 1
				silence_chunk_count = int(silence_duration * sample_rate / chunk_size)

				p = pyaudio.PyAudio()
				stream = p.open(format=audio_format,
								channels=channels,
								rate=sample_rate,
								input=True,
								frames_per_buffer=chunk_size)

				self.logger.info("Recording started")
				print("Recording...")
				time.sleep(3)
				frames = []
				silence_chunks = 0

				while self.is_recording:
						data = stream.read(chunk_size)
						frames.append(data)

						# 将音频数据转换为 numpy 数组
						audio_data = np.frombuffer(data, dtype=np.int16)
						# 计算音频信号的均方根（RMS）,防止音频信号为0
						epsilon = np.finfo(float).eps
						rms = np.sqrt(np.mean(np.maximum(audio_data ** 2, epsilon)))

						# print(rms, silence_threshold, silence_chunks, silence_chunk_count)

						# 检查是否达到静音阈值
						if rms < silence_threshold:
								silence_chunks += 1
						else:
								silence_chunks = 0

						# 如果静音块数超过阈值，结束录音
						if silence_chunks > silence_chunk_count:
								self.logger.info("Silence detected, stopping recording")
								print("Silence detected, stopping recording...")
								break

				stream.stop_stream()
				stream.close()
				p.terminate()

				wf = wave.open(filename, 'wb')
				wf.setnchannels(channels)
				wf.setsampwidth(p.get_sample_size(audio_format))
				wf.setframerate(sample_rate)
				wf.writeframes(b''.join(frames))
				wf.close()

				self.logger.info(f"Recording saved to {filename}")

		def stop_recording(self):
				self.is_recording = False
				self.logger.info("Recording stopped by user")
				print("Recording stopped by user.")

		def reduce_noise(self, input_file, output_file):
				"""
				need adjustment
				:param input_file:
				:param output_file:
				:return:
				"""
				try:
						# 加载音频文件
						waveform, sample_rate = torchaudio.load(input_file)
						metadata = torchaudio.info(input_file)
						# print(metadata)

						# print("Waveform shape:", waveform.shape)
						# print("Sample rate:", sample_rate)

						# 计算短时傅里叶变换 (STFT)
						stft_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=None)
						spectrogram = stft_transform(waveform)

						# print("Spectrogram shape:", spectrogram.shape)
						# 计算噪声频谱
						noise_sample = waveform[:,
									   :int(0.5 * sample_rate)]  # 从音频中选取一段只包含噪声的部分，计算其平均频谱以作为噪声谱。假设音频的前0.5秒为纯噪声。
						noise_spectrogram = stft_transform(noise_sample)
						noise_spectrum = noise_spectrogram.mean(dim=-1, keepdim=True)

						# print("Noise spectrum shape:", noise_spectrum.shape)

						# Perform Spectral Subtraction
						clean_spectrogram = spectrogram - noise_spectrum

						# Ensure the magnitude is non-negative
						clean_magnitude = torch.abs(clean_spectrogram)
						clean_phase = torch.angle(clean_spectrogram)
						clean_spectrogram = clean_magnitude * torch.exp(1j * clean_phase)
						# print("Clean spectrogram shape:", clean_spectrogram.shape)

						# 计算逆短时傅里叶变换 (ISTFT)
						istft_transform = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=512)
						clean_waveform = istft_transform(clean_spectrogram)

						# print("Clean waveform shape:", clean_waveform.shape)
						# 保存降噪后的音频
						torchaudio.save(output_file, clean_waveform, sample_rate,
										bits_per_sample=metadata.bits_per_sample, encoding=metadata.encoding)

						# metadata = torchaudio.info(output_file)
						# print(metadata)
						self.logger.info(f"Noise reduced audio saved to {output_file}")

				except Exception as e:
						self.logger.error(f"Error reducing noise: {e}")

		def change_volume(self, input_file, output_file, target_db=-20):
				try:
						# 自动识别音频文件类型并加载
						audio = AudioSegment.from_file(input_file)

						# 计算需要的增益值
						change_in_dBFS = target_db - audio.dBFS

						# 调整音量
						normalized_audio = audio.apply_gain(change_in_dBFS)

						# 保存调整后的音频
						normalized_audio.export(output_file, format=os.path.splitext(output_file)[1][1:])
						self.logger.info(f"Audio volume adjusted to {target_db} dB and saved to {output_file}")
				except Exception as e:
						self.logger.error(f"Error changing volume: {e}")

		def reduce_noise_by_ai_models(self, input_file, output_file):
				'''
				need adjustment
				:param input_file:
				:param output_file:
				:return:
				'''

				try:
						# 使用 Demucs 进行降噪
						model = pretrained.get_model('htdemucs')
						model.eval()
						model.cpu()

						wav, sr = torchaudio.load(input_file)

						metadata_input_file = torchaudio.info(input_file)

						# 如果是单声道，转换为立体声
						# 检查 wav 的形状并调整
						if wav.shape[0] == 1:
								wav = wav.repeat(2, 1)
						if len(wav.shape) == 2:
								wav = wav.unsqueeze(0)  # 增加批次维度
						with torch.no_grad():
								sources = apply_model(model, wav, device='cpu', shifts=1, split=True)
						denoised = sources[0, 0, :, :]  # Assuming the first source is the clean audio

						# 保存降噪后的音频

						# 归一化音频
						max_amplitude = torch.abs(denoised).max()
						normalized_waveform = denoised / max_amplitude

						# 将音频缩放至-20DB
						rms = torch.sqrt(torch.mean(normalized_waveform ** 2))
						scalar = 10 ** (target_db / 20) / rms
						scaled_waveform = normalized_waveform * scalar
						mono_waveform = torch.mean(scaled_waveform, dim=0)

						final_denoised = scaled_waveform
						torchaudio.save(output_file, final_denoised, sr,
										bits_per_sample=metadata_input_file.bits_per_sample,
										encoding=metadata_input_file.encoding)

						metadata_output_file = torchaudio.info(output_file)
						# print(metadata_output_file)
						self.logger.info(f"Noise reduced audio saved to {output_file}")

				except Exception as e:
						self.logger.error(f"Error reducing noise: {e}")

		def recognize_audio(self, filename):
				recognizer = sr.Recognizer()
				try:
						with sr.AudioFile(filename) as source:
								audio_data = recognizer.record(source)  # 使用 recognizer.record 方法
								try:
										import whisper
										text = recognizer.recognize_whisper(audio_data=audio_data, model='base',
																			language=self.input_language)
										self.logger.info(f"Recognized <{filename}> text: {text}")
										print(f"Recognized <{filename}> text:" + '\n' + text)
								except ImportError:
										self.logger.error("Whisper module is not installed or could not be imported.")
										print("Whisper module is not installed or could not be imported.")
								except sr.UnknownValueError:
										self.logger.error("Whisper could not understand audio")
										print("Whisper could not understand audio")
								except sr.RequestError as e:
										self.logger.error(f"Could not request results from Whisper service; {e}")
										print(f"Could not request results from Whisper service; {e}")
				except Exception as e:
						self.logger.error(f"Error recognizing audio: {e}")

		def create_test_audio(self, input_file, output_file):
				from pydub.generators import WhiteNoise
				# 生成白噪音
				noise = WhiteNoise().to_audio_segment(duration=20000)  # 20秒的白噪音
				noise = noise.apply_gain(volume_change=-40)

				# 创建一个简单的语音片段
				voice = AudioSegment.silent(duration=1000) + AudioSegment.from_file(
						input_file) + AudioSegment.silent(duration=1000)

				# 将噪音和语音混合在一起

				combined = voice.overlay(noise)

				# 保存到文件
				print(os.path.splitext(output_file))
				combined.export(output_file, format=os.path.splitext(output_file)[1][1:])

		def save_audio_plot(self, input_file, save_folder='./'):
				try:
						waveform, sample_rate = torchaudio.load(input_file)

						# Convert waveform to numpy array
						waveform = waveform.numpy()

						# Get the number of channels and frames
						# print(waveform.shape)
						num_channels, num_frames = waveform.shape
						# Calculate time axis
						time_axis = torch.arange(0, num_frames) / sample_rate

						matplotlib.use('TkAgg')  # using windows to show the plot
						# Create subplots
						figure, axes = plt.subplots(num_channels, 1, figsize=(12, 6))
						if num_channels == 1:
								axes = [axes]  # Ensure axes is a list even for single channel

						# Plot waveform for each channel
						for c in range(num_channels):
								axes[c].plot(time_axis, waveform[c], linewidth=1)
								axes[c].grid(True)
								if num_channels > 1:
										axes[c].set_ylabel(f"Channel {c + 1}")

						# Set the title and labels
						figure.suptitle(f"Waveform-{input_file}")
						plt.xlabel('Time (s)')

						save_path = save_folder + os.path.splitext(input_file)[0] + '.png'

						# Show the plot
						# plt.show()
						# Save the figure to a file
						plt.savefig(save_path)
						# Close the plot to free up resources
						plt.close(figure)
						self.logger.info(f"Audio plot <{input_file}> has been created  and saved to {save_path}")
				except Exception as e:
						self.logger.error(f"Error recognizing audio: {e}")

# 示例用法
if __name__ == "__main__":
		audio_processor = AudioProcessor()

		audio_filename = "temp/recorded_audio.wav"
		noise_reduced_filename = "temp/noise_reduced_audio.wav"  # 可以指定不同的格式
		final_audio_filename = "temp/final_audio.wav"  # 可以指定不同的格式
		target_db = -20  # 目标分贝

		# 启动录音线程
		# recording_thread = threading.Thread(target=audio_processor.record_audio, args=(audio_filename,))
		# recording_thread.start()
		#
		# # 模拟用户中途停止录音的操作
		# input("Press Enter to stop recording...\n")
		# time.sleep(5)

		# audio_processor.record_audio(audio_filename)
		# time.sleep(3)
		# audio_processor.stop_recording()

		# 等待录音线程结束
		# recording_thread.join()

		# 调整音量
		audio_processor.change_volume(audio_filename, noise_reduced_filename, target_db)
		# 降噪
		audio_processor.reduce_noise(noise_reduced_filename, final_audio_filename)

		audio_processor.save_audio_plot(input_file=audio_filename)
		audio_processor.save_audio_plot(input_file=noise_reduced_filename)
		audio_processor.save_audio_plot(input_file=final_audio_filename)

		# 识别录制的音频
		audio_processor.recognize_audio(audio_filename)
		audio_processor.recognize_audio(noise_reduced_filename)
		audio_processor.recognize_audio(final_audio_filename)
