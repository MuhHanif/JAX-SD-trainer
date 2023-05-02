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
model_dir = "/home/user/e6_dump/size-576-704-832-squared_no-eos-bos_shuffled_lion-optim-low-lr-e20"
weight_dtype = jnp.bfloat16

tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")


text_encoder, text_encoder_params = FlaxCLIPTextModel.from_pretrained(
    model_dir, subfolder="text_encoder", dtype=weight_dtype, _do_init=False
)

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


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


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
    "testing",
    params={
        "text_encoder": get_params_to_save(t5_encoder_params),
        "vae": get_params_to_save(vae_params),
        "unet": get_params_to_save(unet_params),
    },
)


print()
