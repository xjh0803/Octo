from absl import flags
from absl import app
import os
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import jax

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained/finetuned Octo checkpoint directory."
)
flags.DEFINE_string(
    "dataset_name", None, "Name of dataset loading for the model."
)
flags.DEFINE_string("image_path", None, "Path to image.")
flags.DEFINE_string("language_instruction", None, "Language instruction for task.")

def main():
    # load pre-trained/finetuned model
    from octo.model.octo_model import OctoModel
    model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    image = FLAGS.image_path
    img = np.array(Image.open(requests.get(image, stream=True).raw).resize((256, 256)))
    plt.imshow(img)

    # create obs & task dict, run inference
    # add batch + time horizon 1
    img = img[np.newaxis,np.newaxis,...]
    observation = {"image_primary": img, "timestep_pad_mask": np.array([[True]])}
    task = model.create_tasks(texts=[FLAGS.language_instruction])
    action = model.sample_actions(
        observation,
        task,
        unnormalization_statistics=model.dataset_statistics[FLAGS.dataset_name]["action"],
        rng=jax.random.PRNGKey(0)
    )
    print(action)   # [batch, action_chunk, action_dim]

if __name__ == '__main__':
    app.run(main)