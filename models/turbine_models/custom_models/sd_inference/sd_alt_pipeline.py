from turbine_models.model_runner import vmfbRunner
import os
import time
import sys
from pathlib import Path
from argparse import ArgumentParser
from transformers import CLIPTokenizer

from random import randint
from turbine_models.custom_models.sd_inference import schedulers
import torch
import numpy as np
import iree.runtime as ireert
from iree.runtime import BufferUsage


class AltSdPipeline:
    def __init__(
        self,
        rt_device,
        device,
        punet_vmfb_path,
        clip_vmfb_path,
        vae_vmfb_path,
        unet_param_file,
        clip_param_file,
        vae_param_file,
        hf_model_name,
    ):
        self.rt_device = rt_device
        self.device = device
        self.punet_vmfb_path = punet_vmfb_path
        self.clip_vmfb_path = clip_vmfb_path
        self.vae_vmfb_path = vae_vmfb_path
        self.unet_param_file = unet_param_file
        self.clip_param_file = clip_param_file
        self.vae_param_file = vae_param_file
        self.hf_model_name = hf_model_name

        # TODO: Remove the hard-code
        self.model_max_length = 512

        self.tokenizers = [
            CLIPTokenizer.from_pretrained(self.hf_model_name, subfolder="tokenizer"),
            CLIPTokenizer.from_pretrained(self.hf_model_name, subfolder="tokenizer_2"),
        ]

        self.num_steps = 10

    def encode_prompts_sdxl(self, prompt, negative_prompt):
        # Tokenize prompt and negative prompt.
        text_input_ids_list = []
        uncond_input_ids_list = []

        for tokenizer in self.tokenizers:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids_list += text_inputs.input_ids.unsqueeze(0)
            uncond_input_ids_list += uncond_input.input_ids.unsqueeze(0)

        return text_input_ids_list, uncond_input_ids_list

    def load_clip(self, prompt, neg_prompt):
        prompt_embeds, add_text_embeds = self.encode_prompts_sdxl(prompt, neg_prompt)
        unrolled_inputs = [*prompt_embeds] + [*add_text_embeds]
        runner = vmfbRunner(
            "hip://0", str(self.clip_vmfb_path), str(self.clip_param_file)
        )
        inputs = [
            ireert.asdevicearray(runner.config.device, inp, dtype="int64")
            for inp in unrolled_inputs
        ]
        results = runner.ctx.modules.compiled_clip["encode_prompts"](*inputs)
        # runner.unload()
        return results

    def run_punet(
        self,
        sample,
        prompt_embeds,
        add_text_embeds,
        steps,  # TODO: Use this steps instead of hardcoding as class attr.
        guidance_scale,
    ):
        self.unet = vmfbRunner(
            "hip://" + str(self.rt_device),
            str(self.punet_vmfb_path),
            str(self.unet_param_file),
        )
        scheduler = schedulers.get_scheduler(self.hf_model_name, "EulerDiscrete")
        self.scheduler = schedulers.SharkSchedulerCPUWrapper(
            scheduler,
            1,  # TODO: Add a flag for batch size
            self.unet.config.device,
            latents_dtype=torch.int8,
        )
        self.scheduler.use_punet = True
        image = None
        strength = 0
        latents, add_time_ids, step_indexes, timesteps = self.prepare_latents(
            sample, self.num_steps, image, strength
        )
        guidance_scale = ireert.asdevicearray(
            self.unet.config.device,
            [guidance_scale],
            dtype=np.float16,
            allowed_usage=BufferUsage.DEFAULT,
        )

        for i, t in enumerate(timesteps):
            latent_model_input, t = self.scheduler.scale_model_input(
                latents,
                t,
            )
            t = t.type(torch.int8)
            unet_inputs = [
                latent_model_input,
                t,
                prompt_embeds,
                add_text_embeds,
                add_time_ids,
                guidance_scale,
            ]
            for inp_idx, inp in enumerate(unet_inputs):
                if not isinstance(inp, ireert.DeviceArray):
                    unet_inputs[inp_idx] = ireert.asdevicearray(
                        self.unet.config.device, inp, dtype=np.float16
                    )
            noise_pred = self.unet.ctx.modules.compiled_punet["main"](*unet_inputs)
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
            )
        return latents

    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 30,
        batch_count: int = 2,
        guidance_scale: float = 7.5,
        seed: float = -1,
        # scheduler_id: str = "EulerDiscrete",
    ):
        samples = self.get_rand_latents(seed, batch_count)

        # Tokenize prompt and negative prompt.
        prompt_embeds, negative_embeds = self.load_clip(prompt, negative_prompt)
        vae_runner = vmfbRunner(
            "hip://" + str(self.rt_device),
            str(self.vae_vmfb_path),
            str(self.vae_param_file),
        )
        for i in range(batch_count):
            produce_latents_input = [
                samples[i],
                prompt_embeds,
                negative_embeds,
                steps,
                guidance_scale,
            ]
            latents = self.run_punet(*produce_latents_input)
            image = vae_runner.ctx.modules.compiled_vae["decode"](latents)
            if image is not None:
                print("Generated image successfully")
        self.unet.unload()
        vae_runner.unload()

    def prepare_latents(self, sample, noise, image=None, strength=None):
        self.scheduler.do_guidance = False
        self.scheduler.repeat_sample = False
        (
            sample,
            add_time_ids,
            step_indexes,
            timesteps,
        ) = self.scheduler.initialize_sdxl(sample, noise)
        return (
            sample,
            add_time_ids,
            step_indexes,
            timesteps,
        )

    def get_rand_latents(self, seed, batch_count):
        samples = []
        uint32_info = np.iinfo(np.uint32)
        uint32_min, uint32_max = uint32_info.min, uint32_info.max
        if seed < uint32_min or seed >= uint32_max:
            seed = randint(uint32_min, uint32_max)
        for i in range(batch_count):
            generator = torch.manual_seed(seed + i)
            rand_sample = torch.randn(
                (
                    1,  # self.batch_size,
                    4,
                    1024 // 8,  # self.height // 8,
                    1024 // 8,  # self.width // 8,
                ),
                generator=generator,
                dtype=torch.float16,
            )
            samples.append(rand_sample)
        return samples


def main(args):
    rt_device = args.rt_device
    device = args.device
    punet_vmfb_path = args.punet_vmfb
    clip_vmfb_path = args.clip_vmfb
    vae_vmfb_path = args.vae_vmfb
    unet_param_file = args.unet_params
    clip_param_file = args.clip_params
    vae_param_file = args.vae_params
    hf_model = args.hf_model

    pipeline = AltSdPipeline(
        rt_device,
        device,
        punet_vmfb_path,
        clip_vmfb_path,
        vae_vmfb_path,
        unet_param_file,
        clip_param_file,
        vae_param_file,
        hf_model,
    )

    pipeline.generate_images(
        prompt="A tabby cat with magnificent whiskers", negative_prompt="Low resolution"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    default_artifacts_dir = "/data/mlperf_sdxl/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16/turbine_sdxl_quant/bs1/"
    default_weights_dir = "/data/mlperf_sdxl/models/SDXL/official_pytorch/fp16/stable_diffusion_fp16/safetensors_quant/"
    parser.add_argument(
        "--rt_device",
        type=int,
        default=0,
        help="The device ID on which the runtime will be invoked.",
    )
    parser.add_argument(
        "--device", type=str, default="gfx942", help="Target device for the runtime"
    )
    parser.add_argument(
        "--punet-vmfb",
        type=Path,
        default=(
            default_artifacts_dir
            + "checkpoint_pipe_bs1_64_1024x1024_i8_punet_gfx942.vmfb"
        ),
        help="Path to the PUNET VMFB module",
    )
    parser.add_argument(
        "--clip-vmfb",
        type=Path,
        default=(
            default_artifacts_dir
            + "checkpoint_pipe_bs1_64_fp16_prompt_encoder_rocm_gfx942.vmfb"
        ),
        help="Path to the CLIP VMFB module",
    )
    parser.add_argument(
        "--vae-vmfb",
        type=Path,
        default=(
            default_artifacts_dir
            + "checkpoint_pipe_bs1_1024x1024_fp16_vae_decomp_attn_gfx942.vmfb"
        ),
        help="Path to the VAE VMFB module",
    )
    parser.add_argument(
        "--sched-vmfb",
        type=Path,
        default=(
            default_artifacts_dir
            + "checkpoint_pipe_EulerDiscreteScheduler_bs1_1024x1024_fp16_20_gfx942.vmfb"
        ),
        help="Path to the VAE VMFB module",
    )
    parser.add_argument(
        "--unet-params",
        type=Path,
        default=(default_weights_dir + "checkpoint_pipe_punet_dataset_i8.irpa"),
        help="Path to the VAE params file",
    )
    parser.add_argument(
        "--clip-params",
        type=Path,
        default=(default_weights_dir + "checkpoint_pipe_text_encoder_fp16.irpa"),
        help="Path to the CLIP params file",
    )
    parser.add_argument(
        "--vae-params",
        type=Path,
        default=(default_weights_dir + "vae.irpa"),
        help="Path to the VAE params file",
    )

    parser.add_argument(
        "--hf-model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HF repo path for the model",
    )

    main(parser.parse_args(args=(sys.argv[1:] if sys.argv is not None else ["--help"])))
