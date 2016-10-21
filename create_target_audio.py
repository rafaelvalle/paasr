from params import LEFT_CONTEXT, RIGHT_CONTEXT, FRAME_SIZE, KEYWORD
from params import text_grid_glob_str, TARGET_AUDIO_DIRECTORY
from build_kws_data import create_target_audio as cta

cta(text_grid_glob_str, KEYWORD, FRAME_SIZE, LEFT_CONTEXT, RIGHT_CONTEXT,
    TARGET_AUDIO_DIRECTORY)
