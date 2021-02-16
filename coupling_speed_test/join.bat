(for %%i in (video*.mp4) do @echo file '%%i') > mylist.txt
"..\ffmpeg-20200628-4cfcfb3-win64-static\bin\ffmpeg.exe" -f concat -safe 0 -i mylist.txt -c copy concatenated.mp4
