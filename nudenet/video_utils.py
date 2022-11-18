import cv2
import os
import logging
from dataclasses import dataclass, field

# logging.basicConfig(level=logging.DEBUG)

from skimage import metrics as skimage_metrics


def is_similar_frame(f1, f2, resize_to=(64, 64), thresh=0.5, return_score=False):
    thresh = float(os.getenv("FRAME_SIMILARITY_THRESH", thresh))

    if f1 is None or f2 is None:
        return False

    if isinstance(f1, str) and os.path.exists(f1):
        try:
            f1 = cv2.imread(f1)
        except Exception as ex:
            logging.exception(ex, exc_info=True)
            return False

    if isinstance(f2, str) and os.path.exists(f2):
        try:
            f2 = cv2.imread(f2)
        except Exception as ex:
            logging.exception(ex, exc_info=True)
            return False

    if resize_to:
        f1 = cv2.resize(f1, resize_to)
        f2 = cv2.resize(f2, resize_to)

    if len(f1.shape) == 3:
        f1 = f1[:, :, 0]

    if len(f2.shape) == 3:
        f2 = f2[:, :, 0]

    score = skimage_metrics.structural_similarity(f1, f2, multichannel=False)

    if return_score:
        return score

    if score >= thresh:
        return True

    return False


@dataclass
class IndexedFrame:
    index: int
    frame: None  # numpy array

    def with_frame(self, new_frame):
        return IndexedFrame(self.index, new_frame)


@dataclass
class Fifo:
    length: int
    _list: list = field(default_factory=list)

    def push(self, new_element):
        self._list.append(new_element)
        self._list = self._list[-self.length:]
        return new_element

    def list(self):
        return self._list


def get_interest_frames_from_video(
    video_path,
    frame_similarity_threshold=0.5,
    similarity_context_n_frames=3,
    skip_n_frames=0.5,
    output_frames_to_dir=None,
):
    skip_n_frames = float(os.getenv("SKIP_N_FRAMES", skip_n_frames))

    important_frames = []
    fps = 0
    video_length = 0

    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        def generate_important_frames():
            logging.info(
                f"Important frames will be processed from {video_path} of length {video_length}"
            )

            _skip_n_frames = skip_n_frames
            if _skip_n_frames < 1:
                _skip_n_frames = int(_skip_n_frames * fps)
                logging.info(f"skip_n_frames: {_skip_n_frames}")

            past_frames = Fifo(similarity_context_n_frames)
            for frame_i in range(video_length + 1):
                read_flag, current_frame = video.read()

                if not read_flag:
                    break

                if _skip_n_frames > 0:
                    if frame_i % _skip_n_frames != 0:
                        continue

                frame_i += 1

                found_similar = False
                for context_frame in reversed(past_frames.list()):
                    if is_similar_frame(
                        context_frame.frame, current_frame, thresh=frame_similarity_threshold
                    ):
                        logging.debug(f"{frame_i} is similar to {context_frame.index}")
                        found_similar = True
                        break

                if not found_similar:
                    logging.debug(f"{frame_i} is added to important frames")
                    new_frame = IndexedFrame(frame_i, current_frame)
                    past_frames.push(new_frame)
                    if output_frames_to_dir:
                        if not os.path.exists(output_frames_to_dir):
                            os.mkdir(output_frames_to_dir)

                        stripped_dir = output_frames_to_dir.rstrip("/")
                        cv2.imwrite(
                            f"{stripped_dir}/{str(frame_i).zfill(10)}.png",
                            current_frame,
                        )
                    yield new_frame

        important_frames = generate_important_frames()

    except Exception as ex:
        logging.exception(ex, exc_info=True)

    return important_frames, fps, video_length


if __name__ == "__main__":
    import sys

    imp_frames = get_interest_frames_from_video(
        sys.argv[1], output_frames_to_dir="./frames/"
    )
    print([i[0].index for i in imp_frames])
