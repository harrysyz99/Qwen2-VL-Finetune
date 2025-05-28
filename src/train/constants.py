IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

# Configuration for Multi-Head Model
TASK_TYPES = ["regression", "color_classification", "shape_classification", "general_classification"]
TASK_CONFIG = {
    "regression": {"output_dim": 1, "type": "regression"},
    "color_classification": {"num_categories": 5, "type": "classification"},  # e.g., red, green, blue, yellow, black
    "shape_classification": {"num_categories": 3, "type": "classification"},  # e.g., circle, square, triangle
    "general_classification": {"num_categories": 100, "type": "classification"} # Example for a general task
}
TASK_TYPE_TO_ID = {name: i for i, name in enumerate(TASK_TYPES)}
ID_TO_TASK_TYPE = {i: name for i, name in enumerate(TASK_TYPES)}

# Check if the number of task types matches the config length
assert len(TASK_TYPES) == len(TASK_CONFIG), "TASK_TYPES list and TASK_CONFIG dictionary must have the same number of tasks."