import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


class Args:
    model_path: str = "liuhaotian/LLaVA-Lightning-MPT-7B-preview"
    model_base: str = None
    num_gpus: int = 1
    conv_mode: str = None
    temperature: float = 0.2
    max_new_tokens: int = 512
    load_8bit: bool = False
    load_4bit: bool = False
    debug: bool = False
    

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(image_filename):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(Args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(Args.model_path, Args.model_base, model_name, Args.load_8bit, Args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if Args.conv_mode is not None and conv_mode != Args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, Args.conv_mode, Args.conv_mode))
    else:
        Args.conv_mode = conv_mode

    conv = conv_templates[Args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(image_filename)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")
        # Fixme: remove
        print(Args.load_8bit, Args.load_4bit, Args.debug)

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Fixme: remove
        print("PROMPT: ", prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        # print("FINALLY: ", conv)
        # print("CONV_MSGS: ", conv.messages)
        # print("CONV_MSGS: ", conv.messages[-1][-1])
        # print("CONV_MSGS: ", conv.messages[-1][-1][:-10]) # TODO: USE THIS FOR OUTPUT

        if Args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    # python -m llava.serve.cli \
    #     --model-path liuhaotian/LLaVA-Lightning-MPT-7B-preview \
    #     --image-file "https://llava-vl.github.io/static/images/view.jpg" \

    # prompt: Describe the image in 500 characters. Only include what you see and nothing else. Include information about the background, the time of day, relative positioning of objects, lighting, any text in the image, objects and people present, and the setting.
    image_filename = "https://llava-vl.github.io/static/images/view.jpg"
    main()
