from transformers import (
    CLIPFeatureExtractor,
    CLIPTokenizer,
    FlaxCLIPTextModel,
    FlaxT5EncoderModel,
    T5Tokenizer,
)
from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker
import jax
import jax.numpy as jnp

# ====================[Init all models]==================== #

# TODO: extract this variable out
out_dir = "sd1.5-t5-base"
model_dir = "/home/user/data_dump/sd1.5-t5-e0"  # insert SD1.5 flax model dir here
weight_dtype = jnp.float32

vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    model_dir,
    dtype=weight_dtype,
    subfolder="vae",
)

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    model_dir, subfolder="unet", dtype=weight_dtype, use_memory_efficient=True
)

t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")


t5_encoder, t5_encoder_params = FlaxT5EncoderModel.from_pretrained(
    "google/flan-t5-base", dtype=weight_dtype, _do_init=False
)


scheduler, _ = FlaxPNDMScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
)


# ====================[save pipeline]==================== #


pipeline = FlaxStableDiffusionPipeline(
    text_encoder=t5_encoder,
    vae=vae,
    unet=unet,
    tokenizer=t5_tokenizer,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    ),
)


pipeline.save_pretrained(
    out_dir,
    params={
        "text_encoder": (t5_encoder_params),
        "vae": (vae_params),
        "unet": (unet_params),
    },
)


print()
