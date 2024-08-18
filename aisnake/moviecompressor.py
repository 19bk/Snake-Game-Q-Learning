import moviepy.editor as mp
import sys

print(sys.executable)

# Load the video file
video = mp.VideoFileClip("Screen Recording 2024-07-19 at 22.35.06.mov")

# Speed up the video by 5x
sped_up_video = video.speedx(5)

# Write the result to a new file, specifying the codec
sped_up_video.write_videofile("output_video.mov", codec='libx264')

# Close the video files
video.close()
sped_up_video.close()