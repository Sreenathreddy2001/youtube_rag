import os

if not os.path.exists('transcriptions.txt'):
    import whisper
    from pytube import YouTube
    import tempfile

    # Download audio from YouTube
    youtube = YouTube("https://www.youtube.com/watch?v=Xp_umBGOgKk")
    audio = youtube.streams.filter(only_audio=True).first()

    # Load the Whisper model
    whisper_model = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download the audio file to a temporary directory
        file_path = audio.download(output_path=tmpdir)

        # Transcribe the audio file using Whisper
        transcription = whisper_model.transcribe(file_path, fp16=False)["text"].strip()

        # Write the transcription to a text file
        with open("transcriptions.txt", "w") as file:
            file.write(transcription)
        with open ("transcriptions.txt") as file:
          transcription=file.read()
transcription
