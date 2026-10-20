import spaces  # must be imported before torch so ZeroGPU can patch CUDA
import os
import gradio as gr
from huggingface_hub import hf_hub_download
from blueprints.pokemon_classifier.model import PokemonClassifier

# Hugging Face model repo holding pokemon_v1.pth and class_mapping.json
HF_MODEL_REPO = os.environ.get('HF_MODEL_REPO', 'heytony/pokemon-classifier')


def model_file(filename: str) -> str:
    # Use the local copy in models/ when running locally; the Space downloads it from the Hub
    local_path = os.path.join('models', filename)
    return local_path if os.path.exists(local_path) else hf_hub_download(HF_MODEL_REPO, filename)


# Inference runs on the Space's CPU (~1s per image), so visitors don't use up ZeroGPU quota,
# which is tiny for anonymous calls coming from the GitHub Pages site
classifier = PokemonClassifier(
    device='cpu',
    model_path=model_file('pokemon_v1.pth'),
    class_mapping_path=model_file('class_mapping.json'),
)


@spaces.GPU(duration=5)
def _zerogpu_placeholder():
    # Never called; ZeroGPU Spaces must define at least one @spaces.GPU function to start
    pass


def predict(image):
    if image is None:
        raise gr.Error('Please upload an image')
    return classifier.predict(image, return_top_k=3)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label='Pokemon image'),
    outputs=gr.JSON(label='Prediction'),
    title='Pokemon Gen1 Classifier',
    description='VGG16 fine-tuned on the original 151 Pokemon. Used by https://heytonyy.github.io/ml-with-pokemon/',
    api_name='predict',
    flagging_mode='never',
)

if __name__ == '__main__':
    demo.launch()
