(for %%i in (video*.mov) do @echo file '%%i') > mylist.txt
"..\ffmpeg-20200628-4cfcfb3-win64-static\bin\ffmpeg.exe" -f concat -safe 0 -i mylist.txt -c copy concatenated.mp4
