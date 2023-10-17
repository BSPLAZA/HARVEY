# Import necessary libraries
import tensorflow as tf
import numpy as np
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.utils import config_util

# Set up the paths
MODEL_DIR = 'model/'
PIPELINE_CONFIG_PATH = 'path/to/pipeline.config'
CHECKPOINT_PATH = 'path/to/pretrained_checkpoint'

# Load the pipeline config
configs = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG_PATH)
model_config = configs['model']
model_config.ssd.num_classes = num_classes  # Number of object classes

# Build the detection model
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(CHECKPOINT_PATH).expect_partial()

# Set up the detection function
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load and preprocess your image (while loop this for the entire set)
image_path = 'path/to/your/image.jpg'
image_np = np.array(tf.image.decode_image(tf.io.read_file(image_path)))

# Make detections
input_tensor = tf.convert_to_tensor(image_np)
detections = detect_fn(input_tensor)

# Visualize the results
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np,
    detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(int),
    detections['detection_scores'][0].numpy(),
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.30,
)

# Display or save the result
plt.imshow(image_np)
plt.show()
