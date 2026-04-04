# download_youtube.py
import yt_dlp
import os

def download_youtube_video(url, output_path='youtube_videos'):
    """Download YouTube video"""
    os.makedirs(output_path, exist_ok=True)
    
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'quiet': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Fix extension
        filename = filename.replace('.webm', '.mp4').replace('.mkv', '.mp4')
        print(f"Downloaded: {filename}")
        return filename

if __name__ == "__main__":
    # Replace with your Jedi Academy YouTube URL
    url = input("Enter YouTube URL: ")
    download_youtube_video(url)