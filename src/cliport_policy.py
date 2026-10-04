from pathlib import Path
import numpy as np
import torch
import clip
import jax
import jax.numpy as jnp

from .cliport_model import TransporterNets


class CLIPortPolicy:
    """Inference-only wrapper around the repository's pretrained CLIPort variant."""

    def __init__(self, checkpoint_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        rng = jax.random.PRNGKey(0)
        _, key = jax.random.split(rng)

        init_img = jnp.ones((1, 224, 224, 5), jnp.float32)
        init_text = jnp.ones((1, 512), jnp.float32)
        init_pix = jnp.zeros((1, 2), np.int32)

        self.params = TransporterNets().init(
            key, init_img, init_text, init_pix
        )["params"]

        checkpoint_path = str(checkpoint_path)
        restored = __import__("flax.training.checkpoints", fromlist=["restore_checkpoint"]).restore_checkpoint(
            checkpoint_path, target=None
        )

        if "target" in restored:
            self.params = restored["target"]
        elif "params" in restored:
            self.params = restored["params"]
        else:
            raise KeyError(
                f"Unsupported checkpoint structure. Keys: {list(restored.keys())}"
            )

    def _encode_text(self, text):
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens).float()
        feats /= feats.norm(dim=-1, keepdim=True)
        return np.float32(feats.cpu())

    def predict(self, observation):
        """Return pick/place pixels and world coordinates without executing."""
        text = observation["_instruction"]
        xyzmap = observation["xyzmap"]

        text_feats = self._encode_text(text)

        img = observation["image"][None, ...] / 255.0
        coords_x, coords_y = np.meshgrid(
            np.linspace(-1, 1, 224),
            np.linspace(-1, 1, 224),
            indexing="ij",
        )
        coords = np.concatenate(
            (coords_x[..., None], coords_y[..., None]), axis=2
        )
        img = np.concatenate((img, coords[None, ...]), axis=3)

        batch = {
            "img": jnp.float32(img),
            "text": jnp.float32(text_feats),
        }

        pick_map, place_map = TransporterNets().apply(
            {"params": self.params}, batch["img"], batch["text"]
        )

        pick_map = np.asarray(pick_map, dtype=np.float32)
        place_map = np.asarray(place_map, dtype=np.float32)

        pick_index = int(np.argmax(pick_map))
        place_index = int(np.argmax(place_map))

        pick_yx = (pick_index // 224, pick_index % 224)
        place_yx = (place_index // 224, place_index % 224)

        pick_yx = (
            int(np.clip(pick_yx[0], 20, 204)),
            int(np.clip(pick_yx[1], 20, 204)),
        )
        place_yx = (
            int(np.clip(place_yx[0], 20, 204)),
            int(np.clip(place_yx[1], 20, 204)),
        )

        return {
            "pick_yx": pick_yx,
            "place_yx": place_yx,
            "pick_xyz": xyzmap[pick_yx[0], pick_yx[1]].copy(),
            "place_xyz": xyzmap[place_yx[0], place_yx[1]].copy(),
            "pick_map": pick_map.reshape(224, 224),
            "place_map": place_map.reshape(224, 224),
        }
