import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import cv2
import einops
import config
import random

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

class ControlNet(AbstractFunction):
    @setup(cacheable=False, function_type="controlnet", batchable=True)
    def setup(self, prompt = None, a_prompt = "best quality, extremely detailed", n_prompt = "longbody, lowres, " \
              "bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality", \
                num_samples = 1, image_resolution = 512, ddim_steps = 20, guess_mode = False, strength = 1.0, \
                scale = 6.0, seed = -1, eta = 0.0, low_threshold = 50, high_threshold = 60, preprocessor = None):
        self.prompt = prompt
        self.a_prompt = a_prompt
        self.n_prompt = n_prompt
        self.num_samples = num_samples
        self.image_resolution = image_resolution
        self.ddim_steps = ddim_steps
        self.guess_mode = guess_mode
        self.strength = strength
        self.scale = scale
        self.seed = seed
        self.eta = eta
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.preprocessor = preprocessor

    @property
    def name(self):
        return "controlnet"
    
    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["controlframe"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None)],
            )
        ],
    )
    def forward(self, frame: pd.DataFrame) -> pd.DataFrame:
        print(self.prompt)

        def HWC3(x):
            # assert x.dtype == np.uint8
            if x.ndim == 2:
                x = x[:, :, None]
            assert x.ndim == 3
            H, W, C = x.shape
            assert C == 1 or C == 3 or C == 4
            if C == 3:
                return x
            if C == 1:
                return np.concatenate([x, x, x], axis=2)
            if C == 4:
                color = x[:, :, 0:3].astype(np.float32)
                alpha = x[:, :, 3:4].astype(np.float32) / 255.0
                y = color * alpha + 255.0 * (1.0 - alpha)
                y = y.clip(0, 255).astype(np.uint8)
                return y


        def resize_image(input_image, resolution):
            H, W, C = input_image.shape
            H = float(H)
            W = float(W)
            k = float(resolution) / min(H, W)
            H *= k
            W *= k
            H = int(np.round(H / 64.0)) * 64
            W = int(np.round(W / 64.0)) * 64
            img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
            return img

        model_name = 'control_v11p_sd15_canny'
        model = create_model(f'./models/{model_name}.yaml').cpu()
        model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
        model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
        model = model.cuda()
        ddim_sampler = DDIMSampler(model)

        def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):

            if det == 'Canny':
                if not isinstance(self.preprocessor, CannyDetector):
                    self.preprocessor = CannyDetector()

            with torch.no_grad():
                input_image = HWC3(input_image)

                if det == 'None':
                    detected_map = input_image.copy()
                else:
                    detected_map = self.preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
                    detected_map = HWC3(detected_map)

                img = resize_image(input_image, image_resolution)
                H, W, C = img.shape

                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

                control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                control = torch.stack([control for _ in range(num_samples)], dim=0)
                control = einops.rearrange(control, 'b h w c -> b c h w').clone()

                if seed == -1:
                    seed = random.randint(0, 65535)
                seed_everything(seed)

                # if config.save_memory:
                #     model.low_vram_shift(is_diffusing=False)

                cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
                shape = (4, H // 8, W // 8)

                # if config.save_memory:
                #     model.low_vram_shift(is_diffusing=True)

                model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

                samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                            shape, cond, verbose=False, eta=eta,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond)

                # if config.save_memory:
                #     model.low_vram_shift(is_diffusing=False)

                x_samples = model.decode_first_stage(samples)
                x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

                results = [x_samples[i] for i in range(num_samples)]
            return results
        
        ret = pd.DataFrame({"controlframe": process("Canny", frame["data"].to_numpy()[0], self.prompt, self.a_prompt, \
                                                        self.n_prompt, self.num_samples, self.image_resolution, 512, \
                                                        self.ddim_steps, self.guess_mode, self.strength, self.scale, \
                                                        self.seed, self.eta, self.low_threshold, self.high_threshold)})
        return ret