import os
import time
import logging
import random
import cv2
import numpy as np

from multi_video_capture import MultiVideoCapture
from fish_detector import FishDetector
from entropy_extractor import EntropyExtractor
from conditioning import Conditioner
from randomness_tests import RandomnessTester
from visualization import EntropyVisualizer
from key_auditor import KeyAuditor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("trng.log"), logging.StreamHandler()]
    )

def main():
    setup_logging()

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    STREAM_URLS_FILE = os.path.join(SCRIPT_DIR, "..", "data", "live_streams.txt")
    KEYS_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "generated_keys.txt")

    ENTROPY_POOL_SIZE_BITS = 2048
    KEY_SIZE_BYTES = 32
    VIS_WIDTH, VIS_HEIGHT = 512, 512

    logging.info("Initializing TRNG components...")
    video_capture = MultiVideoCapture(STREAM_URLS_FILE, max_streams=3)
    fish_detector = FishDetector(
        model_path=os.path.join(SCRIPT_DIR, "..", "yolov8n.pt"),
        allowed_classes=[],  # Allow all by default; set to [0] if class 0 = fish
        min_area=60,
        max_area_ratio=0.25,
        iou_threshold=0.5
    )
    entropy_extractor = EntropyExtractor()
    conditioner = Conditioner(key_size=KEY_SIZE_BYTES)
    randomness_tester = RandomnessTester(alpha=0.01)
    visualizer = EntropyVisualizer(width=VIS_WIDTH, height=VIS_HEIGHT)
    auditor = KeyAuditor(min_batch=20, max_batch=50)
    logging.info("Initialization complete.")

    last_key_gen_time = time.time()
    start_time = time.time()
    keys_saved_count = 0
    entropy_extractor.sources_seen = set()

    FISH_SAMPLE_BASE = 0.6
    FISH_SAMPLE_VAR = 0.15

    try:
        while True:
            timestamp_ns = time.time_ns()
            frames_by_url = video_capture.get_frames()
            if not frames_by_url:
                time.sleep(0.1)
                continue

            # Define current_stream_url for use in logging and key generation
            current_stream_url = list(frames_by_url.keys())[0]

            annotated_frames = []
            used_ids_this_iteration = []
            # Sort by URL to ensure consistent ordering
            sorted_urls = sorted(frames_by_url.keys())
            for url in sorted_urls:
                frame = frames_by_url[url]
                frame_height, frame_width = frame.shape[0], frame.shape[1]
                tracked_objects, annotated_frame = fish_detector.detect_and_track(frame)
                cv2.putText(annotated_frame, url[:80], (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                annotated_frames.append(annotated_frame)

                moving_objects = [obj for obj in tracked_objects if not obj.get("is_static", False)]

                def motion_energy(obj):
                    vx, vy = obj.get("speed_vector", (0, 0))
                    return float(np.linalg.norm([vx, vy]) + 1e-6)

                if moving_objects:
                    p = float(np.clip(np.random.normal(FISH_SAMPLE_BASE, FISH_SAMPLE_VAR), 0.15, 0.95))
                    num_to_sample = np.random.binomial(len(moving_objects), p)
                    if num_to_sample > 0:
                        weights = np.array([motion_energy(o) for o in moving_objects], dtype=np.float64)
                        weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)
                        idxs = np.random.choice(len(moving_objects), size=num_to_sample, replace=False, p=weights)
                        sampled_objects = [moving_objects[i] for i in idxs]
                        
                        for obj in sampled_objects:
                            used_ids_this_iteration.append(obj['id'])

                        entropy_extractor.extract_entropy(
                            sampled_objects, frame_width, frame_height, timestamp_ns, source_id=url
                        )
                        entropy_extractor.sources_seen.add(url)

            # Create a grid of video feeds
            num_streams = len(annotated_frames)
            if num_streams > 0:
                target_height = 240 # Smaller height for more streams
                resized_feeds = []
                for f in annotated_frames:
                    scale = target_height / f.shape[0]
                    new_width = int(f.shape[1] * scale)
                    resized_feeds.append(cv2.resize(f, (new_width, target_height)))

                cols = 2 # Let's try 2 columns
                rows = (num_streams + cols - 1) // cols
                max_w = max(f.shape[1] for f in resized_feeds) if resized_feeds else 0
                grid_h = rows * target_height
                grid_w = cols * max_w
                
                video_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

                for i, feed in enumerate(resized_feeds):
                    row, col = i // cols, i % cols
                    h, w, _ = feed.shape
                    y_offset, x_offset = row * target_height, col * max_w
                    # Center the feed in the cell
                    x_start = x_offset + (max_w - w) // 2
                    video_grid[y_offset:y_offset+h, x_start:x_start+w] = feed
                
                # Resize the final grid to fit the dashboard
                scale = VIS_HEIGHT / video_grid.shape[0]
                new_width = int(video_grid.shape[1] * scale)
                resized_grid = cv2.resize(video_grid, (new_width, VIS_HEIGHT))
            else:
                # Create a placeholder if no streams are active
                resized_grid = np.zeros((VIS_HEIGHT, VIS_WIDTH, 3), dtype=np.uint8)
                cv2.putText(resized_grid, "No active streams", (50, VIS_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            entropy_pool = entropy_extractor.get_entropy_pool()
            current_entropy_bits = len(entropy_pool) * 8
            entropy_bitmap = visualizer.generate_bitmap(entropy_pool)
            histogram_image = visualizer.generate_histogram(entropy_pool)
            entropy_vis_color = cv2.cvtColor(entropy_bitmap, cv2.COLOR_GRAY2BGR)
            histogram_vis_color = cv2.cvtColor(histogram_image, cv2.COLOR_GRAY2BGR)

            elapsed_time = time.time() - start_time
            run_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            timer_text = f"Run Time: {run_time_str}"
            cv2.putText(entropy_vis_color, timer_text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            keys_text = f"Keys Saved: {keys_saved_count}"
            cv2.putText(entropy_vis_color, keys_text, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

            src_count_text = f"Sources: {len(entropy_extractor.sources_seen)}"
            cv2.putText(entropy_vis_color, src_count_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

            used_ids_str = ", ".join(map(str, sorted(list(set(used_ids_this_iteration)))))
            if len(used_ids_str) > 45: # Truncate if too long
                used_ids_str = used_ids_str[:45] + "..."
            
            ids_text = f"Using IDs: {used_ids_str}"
            cv2.putText(entropy_vis_color, ids_text, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

            if elapsed_time > 1: # Avoid division by zero and noisy initial values
                keys_per_minute = (keys_saved_count / elapsed_time) * 60
            else:
                keys_per_minute = 0
            
            kpm_text = f"Keys/Min: {keys_per_minute:.2f}"
            cv2.putText(entropy_vis_color, kpm_text, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 255), 1)

            vis_text = f"Entropy Pool: {current_entropy_bits}/{ENTROPY_POOL_SIZE_BITS} bits"
            cv2.putText(entropy_vis_color, vis_text, (10, VIS_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.putText(histogram_vis_color, "Byte Distribution", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            dashboard = cv2.hconcat([resized_grid, entropy_vis_color, histogram_vis_color])
            cv2.imshow("Fish TRNG Dashboard", dashboard)

            ready = (current_entropy_bits >= ENTROPY_POOL_SIZE_BITS) and (len(entropy_extractor.sources_seen) >= 2)
            if ready:
                logging.info("--- Sufficient Entropy Collected ---")
                logging.info(f"Time to collect: {time.time() - last_key_gen_time:.2f} seconds")

                context_meta = (current_stream_url or "").encode()[:64]
                secure_key = conditioner.condition_data(entropy_pool, context_meta=context_meta)
                logging.info(f"Generated 256-bit Secure Key: {secure_key.hex()}")

                key_bit_sequence = "".join(format(byte, "08b") for byte in secure_key)
                min_entropy = randomness_tester.min_entropy_per_bit(key_bit_sequence)
                logging.info(f"Min-Entropy per bit: {min_entropy:.4f}")

                test_results = randomness_tester.run_all_tests(key_bit_sequence)
                logging.info("Randomness Test Results:")
                for test_name, result in test_results.items():
                    status = "PASSED" if result["passed"] else "FAILED"
                    logging.info(f"  - {test_name}: {status} (p-value: {result['p_value']:.6f})")

                critical_ok = all([
                    test_results.get("monobit_test", {}).get("passed", False),
                    test_results.get("runs_test", {}).get("passed", False),
                    test_results.get("longest_run_of_ones_test", {}).get("passed", False),
                    test_results.get("discrete_fourier_transform_test", {}).get("passed", False),
                ])

                if not critical_ok:
                    logging.warning("Key failed one or more critical tests. Not saving key.")
                else:
                    with open(KEYS_OUTPUT_FILE, "a") as f:
                        f.write(secure_key.hex() + "\n")
                    logging.info(f"Key saved to {KEYS_OUTPUT_FILE}")
                    auditor.add_key(secure_key)
                    keys_saved_count += 1

                entropy_extractor.clear_entropy_pool()
                entropy_extractor.sources_seen = set()
                last_key_gen_time = time.time()
                logging.info("-------------------------------------\n")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                logging.info("Rotating video stream workers...")
                video_capture.rotate_workers()

    finally:
        logging.info("Shutting down...")
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()