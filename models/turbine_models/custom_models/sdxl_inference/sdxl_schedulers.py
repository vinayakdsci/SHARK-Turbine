# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# from @aviator19941's gist : https://gist.github.com/aviator19941/4e7967bd1787c83ee389a22637c6eea7

import os
import sys

from iree import runtime as ireert
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import torch
import torch._dynamo as dynamo
from diffusers import UNet2DConditionModel
from shark_turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)

import safetensors


class SDXLScheduler(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        num_inference_steps,
        scheduler,
        hf_auth_token=None,
        precision="fp32",
    ):
        super().__init__()
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        self.guidance_scale = 7.5
        if precision == "fp16":
            try:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                    variant="fp16",
                )
            except:
                self.unet = UNet2DConditionModel.from_pretrained(
                    hf_model_name,
                    subfolder="unet",
                    auth_token=hf_auth_token,
                    low_cpu_mem_usage=False,
                )
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                hf_model_name,
                subfolder="unet",
                auth_token=hf_auth_token,
                low_cpu_mem_usage=False,
            )

    def forward(self, sample, prompt_embeds, text_embeds, time_ids):
        sample = sample * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                added_cond_kwargs = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                }
                latent_model_input = torch.cat([sample] * 2)
                t = t.unsqueeze(0)
                # print('UNSQUEEZE T:', t)
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, timestep=t
                )
                noise_pred = self.unet.forward(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                sample = self.scheduler.step(noise_pred, t, sample, return_dict=False)[
                    0
                ]
        return sample


def export_scheduler(
    scheduler,
    hf_model_name,
    batch_size,
    height,
    width,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    ireec_flags=None,
):
    mapper = {}
    utils.save_external_weights(
        mapper, scheduler, external_weights, external_weight_path
    )

    decomp_list = DEFAULT_DECOMPOSITIONS

    decomp_list.extend(
        [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
        ]
    )
    # tensor shapes for tracing
    sample = (batch_size, 4, height // 8, width // 8)
    prompt_embeds = (2, 77, 2048)
    text_embeds = (2, 1280)
    time_ids = (2, 6)

    class CompiledScheduler(CompiledModule):
        if external_weights:
            params = export_parameters(
                scheduler, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(scheduler)

        def main(
            self,
            sample=AbstractTensor(*sample, dtype=torch.float32),
            prompt_embeds=AbstractTensor(*prompt_embeds, dtype=torch.float32),
            text_embeds=AbstractTensor(*text_embeds, dtype=torch.float32),
            time_ids=AbstractTensor(*time_ids, dtype=torch.float32),
        ):
            return jittable(scheduler.forward, decompose_ops=decomp_list)(
                sample, prompt_embeds, text_embeds, time_ids
            )

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledScheduler(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))

    safe_name = utils.create_safe_name(hf_model_name, "-scheduler")
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(module_str)
    print("Saved to", safe_name + ".mlir")

    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, ireec_flags, safe_name)


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    hf_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    schedulers = utils.get_schedulers(args.hf_model_name)
    scheduler = schedulers[args.scheduler_id]
    scheduler_module = SDXLScheduler(
        args.hf_model_name,
        args.num_inference_steps,
        scheduler,
        hf_auth_token=None,
        precision=args.precision,
    )

    print("export scheduler begin")
    mod_str = export_scheduler(
        scheduler_module,
        args.hf_model_name,
        args.batch_size,
        args.height,
        args.width,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.ireec_flags,
    )
    print("export scheduler complete")
    safe_name = utils.create_safe_name(args.hf_model_name, "-scheduler")
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
