import logging
import pymongo
import gridfs
from datetime import datetime
from bson.objectid import ObjectId
from pydub import AudioSegment
from pydub.playback import play

MONGODB_URI = 'mongodb://localhost:27017'

class AudioStorage:
		def __init__(self, mongodb_uri: str = MONGODB_URI,
					 database_name: str = 'audio_storage',
					 log_file: str = 'audio_storage.log'):
				# 连接到 MongoDB
				self.client = pymongo.MongoClient(mongodb_uri)
				self.database_name = database_name
				self.db = self.client[database_name]
				self.fs = gridfs.GridFS(self.db)

				logging.basicConfig(filename=log_file, level=logging.INFO,
									format='%(asctime)s - %(levelname)s - %(message)s')
				self.logger = logging.getLogger(__name__)

		def get_audio_properties(self, file_path: str, show_details: bool = False) -> dict:
				file_property = dict()

				try:
						audio = AudioSegment.from_file(file_path)
						# 获取音频属性
						duration_seconds = len(audio) / 1000.0  # 持续时间（秒）
						frame_rate = audio.frame_rate  # 采样率
						channels = audio.channels  # 声道数
						sample_width = audio.sample_width  # 每个样本的字节数
						frame_count = audio.frame_count()  # 帧数
						dBFS = audio.dBFS  # 相对 dBFS 的分贝

						file_property['Duration'] = f'{duration_seconds:.2f} seconds'
						file_property['Frame Rate'] = f'{frame_rate} Hz'
						file_property['Channels'] = f'{channels}'
						file_property['Sample Width'] = f'{sample_width} bytes'
						file_property['Frame Count'] = f'{frame_count}'
						file_property['dBFS'] = f'{dBFS:.2f} dB'

						if show_details:
								# 打印音频属性
								print(f"Duration: {duration_seconds:.2f} seconds")
								print(f"Frame Rate: {frame_rate} Hz")
								print(f"Channels: {channels}")
								print(f"Sample Width: {sample_width} bytes")
								print(f"Frame Count: {frame_count}")
								print(f"dBFS: {dBFS:.2f} dB")

				except Exception as e:
						self.logger.error(f"Error saving the audio: {e}")

				return file_property

		def save_audio(self, file_path):
				try:
						# 打开文件并存储到 GridFS
						with open(file_path, 'rb') as f:
								audio_id = self.fs.put(f, filename=file_path.split('/')[-1])

						file_property = self.get_audio_properties(file_path)

						transcription, file_length = 'unknown', 'unknown'
						if file_property:
								transcription = str(file_property)
								file_length = file_property.get('Duration')

						# 音频文件的元数据
						audio_metadata = {
								'file_id': str(audio_id),
								'transcription': transcription,
								'file_name': file_path.split('/')[-1],
								'file_length': file_length,  # Length in seconds
								'upload_date': datetime.now()  # 上传日期
						}

						# 插入元数据到 MongoDB
						self.db.audio_metadata.insert_one(audio_metadata)
						print(f'Save file <{file_path}> successfully into database {self.database_name}')
						return audio_metadata

				except Exception as e:
						print(e)
						self.logger.error(f"Error saving the audio: {e}")

		def get_audio(self, file_id: str, download_path: str) -> str:
				try:
						# 从 GridFS 中获取音频文件
						with open(download_path, 'wb') as f:
								audio_file = self.fs.get(ObjectId(file_id))
								f.write(audio_file.read())
						self.logger.info(
								f"Getting the audio file <{audio_file.filename}> and download to <{download_path}>")

						return download_path
				except Exception as e:
						self.logger.error(f"Error saving the audio: {e}")

		def get_metadata(self, file_id: str, show_details: bool = False):
				try:
						# 获取音频文件的元数据
						metadata = self.db.audio_metadata.find_one({'file_id': file_id})
						audio_file = self.fs.get(ObjectId(file_id))

						if metadata and show_details:
								print(f"Audio Metadata: ")
								for key, value in metadata.items():
										print(f"{key}: {value}")
						elif not metadata:
								self.logger.error(f"No audio found with file_id: {file_id}")
						# print(f"No audio found with file_id: {file_id}")
						self.logger.info(f'Getting the audio file <{file_id} -  {audio_file.filename}>')

						return metadata

				except Exception as e:
						self.logger.error(f"Error getting the audio: {e}")

		def play_audio(self, file_id):
				try:
						audio_file = self.fs.get(ObjectId(file_id))
						audio = AudioSegment.from_file(audio_file, format="wav")

						self.logger.info(f'Begin to play the audio file <{file_id} -  {audio_file.filename}>')

						print(f"Playing the audio file <{audio_file.filename}>, duration length <{audio_file.length}>")
						play(audio)

						self.logger.info(f'Stop play the audio file <{file_id} -  {audio_file.filename}>')
				except Exception as e:
						self.logger.error(f"Error playing the audio: {e}")


# 示例用法
if __name__ == "__main__":
		path = 'recorded_audio.wav'
		name = 'audio_storage'

		audio_storage = AudioStorage(database_name=name)
		test_property = audio_storage.get_audio_properties(file_path=path, show_details=True)
		# print(str(test_property))
		# audio_storage.save_audio(file_path=file_path)
		id = '6670117a9cf20e330affb1c4'
		audio_storage.play_audio(file_id=id)
