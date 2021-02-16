.\external_libs\ffmpeg\bin\ffmpeg.exe -loglevel quiet -stats  -y -f rawvideo -pixel_format rgb24  -video_size 1200x1080 -i - -c:v h264 -pix_fmt yuv420p video%1.mp4 < REPLAY.raw 
