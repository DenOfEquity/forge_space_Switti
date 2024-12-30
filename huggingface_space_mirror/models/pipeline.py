import torch
from torchvision.transforms import ToPILImage
from PIL.Image import Image as PILImage

from models.vqvae import VQVAEHF
from models.clip import FrozenCLIPEmbedder
from models.switti import SwittiHF, get_crop_condition
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng


class SwittiPipeline:
    def __init__(self, switti, vae, text_encoder, text_encoder_2):
        self.switti = switti
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

        self.switti.eval()
        self.vae.eval()

        self.last_positive = ""
        self.last_negative = ""
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.attn_bias = None

        self.device = 'cuda'

    @classmethod
    def from_pretrained(cls):
        switti = SwittiHF.from_pretrained("yresearch/Switti", device='cpu').to(torch.bfloat16)
        vae = VQVAEHF.from_pretrained("yresearch/VQVAE-Switti").to(torch.bfloat16).to('cpu')
        text_encoder = None
        text_encoder_2 = None

        return cls(switti, vae, text_encoder, text_encoder_2)

    @staticmethod
    def to_image(tensor):
        return [ToPILImage()(
            (255 * img.cpu().detach()).to(torch.uint8))
        for img in tensor]

    def _encode_prompt(self, prompt: str | list[str]):
        prompt = [prompt] if isinstance(prompt, str) else prompt
 
        encodings = []
 
        if self.text_encoder is None:
            self.text_encoder = FrozenCLIPEmbedder("openai/clip-vit-large-patch14", device='cuda').to(torch.bfloat16)
        else:
            self.text_encoder.to('cuda')
        encodings.append(self.text_encoder.encode(prompt))
        self.text_encoder.to('cpu')

        if self.text_encoder_2 is None:
            self.text_encoder_2 = FrozenCLIPEmbedder("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", device='cuda').to(torch.bfloat16)
        else:
            self.text_encoder_2.to('cuda')
        encodings.append(self.text_encoder_2.encode(prompt))
        self.text_encoder_2.to('cpu')

        prompt_embeds = torch.concat(
            [encoding.last_hidden_state for encoding in encodings], dim=-1
        )
        pooled_prompt_embeds = encodings[-1].pooler_output
        attn_bias = encodings[-1].attn_bias

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def encode_prompt(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        encode_null: bool = True,
    ):

        if prompt == self.last_positive and null_prompt == self.last_negative and self.prompt_embeds is not None:
            prompt_embeds = self.prompt_embeds
            pooled_prompt_embeds = self.pooled_prompt_embeds
            attn_bias = self.attn_bias
        else:
            prompt_embeds, pooled_prompt_embeds, attn_bias = self._encode_prompt(prompt)
            if encode_null:
                B, L, hidden_dim = prompt_embeds.shape
                pooled_dim = pooled_prompt_embeds.shape[1]

                null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(null_prompt)
                
                null_embeds = null_embeds[:, :L].expand(B, L, hidden_dim).to(prompt_embeds.device)
                null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(pooled_prompt_embeds.device)
                null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)

                prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
                attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)

            self.last_positive = prompt
            self.last_negative = null_prompt
            self.prompt_embeds = prompt_embeds
            self.pooled_prompt_embeds = pooled_prompt_embeds
            self.attn_bias = attn_bias

        return prompt_embeds.to('cuda'), pooled_prompt_embeds.to('cuda'), attn_bias.to('cuda')

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        seed: int | None = None,
        cfg: float = 4.0,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        image_size: tuple[int, int] = (512, 512),
        last_scale_temp: float = 1.,
    ) -> torch.Tensor | list[PILImage]:
        """
        only used for inference, on autoregressive mode
        :param prompt: text prompt to generate an image
        :param null_prompt: negative prompt for CFG
        :param seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: sampling using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if return_pil: list of PIL Images, else: torch.tensor (B, 3, H, W) in [0, 1]
        """

        if self.switti is None:
            self.switti = SwittiHF.from_pretrained("yresearch/Switti", device='cuda').to(torch.bfloat16)
        else:
            self.switti.to('cuda')
        if self.vae is None:
            self.vae = VQVAEHF.from_pretrained("yresearch/VQVAE-Switti").to(torch.bfloat16).to('cpu')
        
        vae_quant = self.vae.quantize
        vae_quant.to('cuda')

        if seed is None:
            rng = None
        else:
            rng = torch.Generator(self.device).manual_seed(seed)

        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)

        B = context.shape[0] // 2

        cond_vector = self.switti.text_pooler(cond_vector)

        if self.switti.use_crop_cond:
            crop_coords = get_crop_condition(2 * B * [image_size[0]],
                                             2 * B * [image_size[1]],
                                             ).to(cond_vector.device)
            crop_embed = self.switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, self.switti.D)
            crop_cond = self.switti.crop_proj(crop_embed)
        else:
            crop_cond = None

        sos = cond_BD = cond_vector

        lvl_pos = self.switti.lvl_embed(self.switti.lvl_1L)
        if not self.switti.rope:
            lvl_pos += self.switti.pos_1LC
        next_token_map = (
            sos.unsqueeze(1)
            + self.switti.pos_start.expand(2 * B, self.switti.first_l, -1)
            + lvl_pos[:, : self.switti.first_l]
        )
        cur_L = 0
        f_hat = sos.new_zeros(B, self.switti.Cvae, self.switti.patch_nums[-1], self.switti.patch_nums[-1])

        for b in self.switti.blocks:
            b.attn.kv_caching(self.switti.use_ar) # Use KV caching if switti is in the AR mode 
            b.cross_attn.kv_caching(True)

        for si, pn in enumerate(self.switti.patch_nums):  # si: i-th segment
            ratio = si / self.switti.num_stages_minus_1
            x_BLC = next_token_map

            if self.switti.rope:
                freqs_cis = self.switti.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = self.switti.freqs_cis

            if si >= turn_off_cfg_start_si:
                apply_smooth = False
                x_BLC = x_BLC[:B]
                context = context[:B]
                context_attn_bias = context_attn_bias[:B]
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                for b in self.switti.blocks:
                    if b.attn.caching and b.attn.cached_k is not None:
                        b.attn.cached_k = b.attn.cached_k[:B]
                        b.attn.cached_v = b.attn.cached_v[:B]
                    if b.cross_attn.caching  and b.cross_attn.cached_k is not None:
                        b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                        b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
            else:
                apply_smooth = more_smooth

            for block in self.switti.blocks:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=context,
                    context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
            cur_L += pn * pn

            logits_BlV = self.switti.get_logits(x_BLC, cond_BD)

            # Guidance
            if si < turn_on_cfg_start_si:
                # t = 0, i. e. no guidance
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                # default const cfg
                t = cfg
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            else:
                logits_BlV = logits_BlV / last_scale_temp

            if apply_smooth and si >= smooth_start_si:
                # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            else:
                # defaul nucleus sampling
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.switti.Cvae, pn, pn)
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                    si, len(self.switti.patch_nums), f_hat, h_BChw,
            )
            if si != self.switti.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.switti.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    self.switti.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + self.switti.patch_nums[si + 1] ** 2]
                )
                # double the batch sizes due to CFG
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in self.switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)

        self.switti.to('cpu')
        self.vae.to('cuda')

        # de-normalize, from [-1, 1] to [0, 1]
        img = self.vae.fhat_to_img(f_hat).add(1).mul(0.5)
        if return_pil:
            img = self.to_image(img)
            
        self.vae.to('cpu')

        return img
