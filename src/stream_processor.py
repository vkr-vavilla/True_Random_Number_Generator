import cv2
import yt_dlp
import time
import logging
import threading
from fish_detector import FishDetector
from entropy_extractor import EntropyExtractor

class StreamProcessor(threading.Thread):
    """
    A thread that processes a single video stream to extract entropy.
    """

    def __init__(self, stream_url, data_queue, stop_event, vis_queue, thread_id):
        super().__init__()
        self.stream_url = stream_url
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.vis_queue = vis_queue
        self.thread_id = thread_id
        self.fish_detector = FishDetector()
        self.entropy_extractor = EntropyExtractor()
        self.name = f"Stream-{thread_id}"

    def _get_direct_stream_url(self):
        """Uses yt-dlp to get the direct streamable URL."""
        ydl_opts = {'format': 'best', 'quiet': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.stream_url, download=False)
                return info['url']
        except Exception as e:
            logging.error(f"[{self.name}] Error getting direct stream URL for {self.stream_url}: {e}")
            return None

    def run(self):
        """The main loop for the thread."""
        logging.info(f"[{self.name}] Starting...")
        
        while not self.stop_event.is_set():
            direct_url = self._get_direct_stream_url()
            if not direct_url:
                logging.warning(f"[{self.name}] Could not get direct URL, retrying in 10s...")
                time.sleep(10)
                continue

            cap = cv2.VideoCapture(direct_url)
            if not cap.isOpened():
                logging.error(f"[{self.name}] Failed to open stream. Retrying in 10s...")
                cap.release()
                time.sleep(10)
                continue

            logging.info(f"[{self.name}] Stream opened successfully.")

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"[{self.name}] Lost frame, attempting to reconnect...")
                    break  # Break inner loop to re-establish connection

                timestamp_ns = time.time_ns()
                frame_height, frame_width, _ = frame.shape

                # Detect and track fish
                tracked_objects, annotated_frame = self.fish_detector.detect_and_track(frame)
                
                # Put annotated frame into the visualization queue
                self.vis_queue.put((self.thread_id, annotated_frame))

                # Filter and extract entropy
                moving_objects = [obj for obj in tracked_objects if not obj['is_static']]
                if moving_objects:
                    # We don't need to sample here as randomness comes from multiple streams
                    entropy_data = self.entropy_extractor.extract_entropy(
                        moving_objects, frame_width, frame_height, timestamp_ns, return_only=True
                    )
                    if entropy_data:
                        self.data_queue.put(entropy_data)
            
            cap.release()
            if self.stop_event.is_set():
                break
        
        logging.info(f"[{self.name}] Stopping.")
