from whisper_live.client import TranscriptionClient
client = TranscriptionClient(
  "localhost",
  9090,
  # lang="en",
  lang="zh",
  translate=False,
  # model="small",                                      # also support hf_model => `Systran/faster-whisper-small`
  model="large-v3-turbo",
  # use_vad=False,
  use_vad=True,
  save_output_recording=True,                         # Only used for microphone input, False by Default
  output_recording_filename="./output_recording.wav", # Only used for microphone input
  max_clients=4,
  max_connection_time=600,
  mute_audio_playback=True,                          # Only used for file input, False by Default
)
path = "../../samples/audio/中英混杂技术名词-男.wav"
client(path)