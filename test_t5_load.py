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

base_model_name = "stable-diffusion-v1-5-flax-e"
model_dir = "/home/user/data_dump/sd1.5-t5-e0"
weight_dtype = jnp.bfloat16

# tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")


# text_encoder, text_encoder_params = FlaxCLIPTextModel.from_pretrained(
#     model_dir, subfolder="text_encoder", dtype=weight_dtype, _do_init=False
# )

vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    model_dir,
    dtype=weight_dtype,
    subfolder="vae",
)

unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
    model_dir, subfolder="unet", dtype=weight_dtype, use_memory_efficient=True
)

t5_tokenizer = T5Tokenizer.from_pretrained(model_dir, subfolder="tokenizer")


t5_encoder, t5_encoder_params = FlaxT5EncoderModel.from_pretrained(
    model_dir, subfolder="text_encoder", dtype=weight_dtype, _do_init=False
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
    "/home/user/data_dump/sd1.5-t5-e0-test",
    params={
        "text_encoder": (t5_encoder_params),
        "vae": (vae_params),
        "unet": (unet_params),
    },
)


print()
