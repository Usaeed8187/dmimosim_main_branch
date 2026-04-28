from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import ChannelMamba


@dataclass
class DMIMOChannelMambaConfig:
    prev_len: int = 16
    pred_len: int = 1
    rb_size: int = 12
    max_num_rb: int | None = None
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1234
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    td_headdim: int = 64
    ngroups: int = 1
    num_td_layers: int = 2
    num_bimamba_layers: int = 2
    num_ffn_layers: int = 1
    d_ff_ratio: int = 4
    d_tf: int = 128
    n_head: int = 4
    num_transformer_layers: int = 1
    fusion_type: str = "concat"
    dropout: float = 0.1
    ssm_backend: str = "auto"
    device: str = "cuda"
    checkpoint_path: str | None = None
    freeze_loaded_checkpoint: bool = True
    checkpoint_metadata: dict | None = None
    allow_mismatch_reset: bool = False


class DMIMOChannelMambaPredictor:
    """Offline-trained, online-inference ChannelMamba predictor for dMIMO CSI."""

    def __init__(self, cfg: DMIMOChannelMambaConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model: ChannelMamba | None = None
        self.d_in: int | None = None
        self._loaded_from_checkpoint = False

    @staticmethod
    def _to_pair_indices(node_idx: int, num_bs_ant: int, num_ue_ant: int) -> np.ndarray:
        if node_idx == 0:
            return np.arange(0, num_bs_ant)
        return np.arange(
            num_bs_ant + (node_idx - 1) * num_ue_ant,
            num_bs_ant + node_idx * num_ue_ant,
        )

    @staticmethod
    def _ensure_prev_len(features: np.ndarray, prev_len: int) -> np.ndarray:
        """Ensure time axis has at least prev_len steps by front-padding."""

        m, t_hist, d_in = features.shape
        if t_hist >= prev_len:
            return features
        pad_count = prev_len - t_hist
        pad = np.repeat(features[:, 0:1, :], pad_count, axis=1)
        return np.concatenate([pad, features], axis=1).reshape(m, prev_len, d_in)

    def _compress_history_to_features(self, curr_h: np.ndarray) -> Tuple[np.ndarray, int, int, int, int, int]:
        """
        Convert CSI history block to ChannelMamba inputs.

        curr_h shape: [T, B, 1, N_r, 1, N_t, N_sym, N_sc]
        return features shape: [N_r*N_t, T, 2*N_rb]
        """

        hist = np.asarray(curr_h)
        t_hist, _, _, n_r, _, n_t, n_sym, n_sc = hist.shape
        hist = hist[:, 0, 0, :, 0, :, :, :]  # [T, N_r, N_t, N_sym, N_sc]
        hist_sym_mean = np.mean(hist, axis=3)  # [T, N_r, N_t, N_sc]

        rb_size = int(self.cfg.rb_size)
        max_num_rb = self.cfg.max_num_rb
        n_rb_full = n_sc // rb_size
        n_rb = n_rb_full if max_num_rb is None else min(n_rb_full, int(max_num_rb))
        if n_rb <= 0:
            raise ValueError(f"Invalid RB configuration: n_sc={n_sc}, rb_size={rb_size}, max_num_rb={max_num_rb}")

        sc_used = n_rb * rb_size
        hist_trim = hist_sym_mean[..., :sc_used]
        rb_vals = hist_trim.reshape(t_hist, n_r, n_t, n_rb, rb_size).mean(axis=-1)  # [T, N_r, N_t, N_rb]
        feat = np.concatenate([np.real(rb_vals), np.imag(rb_vals)], axis=-1)  # [T, N_r, N_t, 2*N_rb]
        feat = np.transpose(feat, (1, 2, 0, 3)).reshape(n_r * n_t, t_hist, 2 * n_rb)
        return feat.astype(np.float32), n_rb, n_sc, n_sym, n_r, n_t

    def _reconstruct_block_from_features(
        self,
        pred_features: np.ndarray,
        n_rb: int,
        n_sc: int,
        n_sym: int,
        n_r: int,
        n_t: int,
    ) -> np.ndarray:
        """Reconstruct full-sc/symbol CSI block from one-step prediction features."""

        rb_size = int(self.cfg.rb_size)
        pred_features = pred_features.reshape(n_r, n_t, 2 * n_rb)
        rb_real = pred_features[..., :n_rb]
        rb_imag = pred_features[..., n_rb:]
        rb_complex = rb_real + 1j * rb_imag  # [N_r, N_t, N_rb]

        sc_vals = np.repeat(rb_complex, rb_size, axis=-1)  # [N_r, N_t, N_rb*rb_size]
        if sc_vals.shape[-1] < n_sc:
            rem = n_sc - sc_vals.shape[-1]
            tail = np.repeat(sc_vals[..., -1:], rem, axis=-1)
            sc_vals = np.concatenate([sc_vals, tail], axis=-1)
        elif sc_vals.shape[-1] > n_sc:
            sc_vals = sc_vals[..., :n_sc]

        sym_vals = np.repeat(sc_vals[:, :, np.newaxis, :], n_sym, axis=2)  # [N_r, N_t, N_sym, N_sc]

        out = np.zeros((1, 1, n_r, 1, n_t, n_sym, n_sc), dtype=np.complex64)
        out[0, 0, :, 0, :, :, :] = sym_vals.astype(np.complex64)
        return out

    def _build_model(self, d_in: int) -> ChannelMamba:
        model = ChannelMamba(
            d_in=d_in,
            prev_len=int(self.cfg.prev_len),
            pred_len=int(self.cfg.pred_len),
            d_model=int(self.cfg.d_model),
            d_state=int(self.cfg.d_state),
            d_conv=int(self.cfg.d_conv),
            expand=int(self.cfg.expand),
            td_headdim=int(self.cfg.td_headdim),
            ngroups=int(self.cfg.ngroups),
            num_td_layers=int(self.cfg.num_td_layers),
            num_bimamba_layers=int(self.cfg.num_bimamba_layers),
            num_ffn_layers=int(self.cfg.num_ffn_layers),
            d_ff_ratio=int(self.cfg.d_ff_ratio),
            d_tf=int(self.cfg.d_tf),
            n_head=int(self.cfg.n_head),
            num_transformer_layers=int(self.cfg.num_transformer_layers),
            fusion_type=str(self.cfg.fusion_type),
            dropout=float(self.cfg.dropout),
            ssm_backend=str(self.cfg.ssm_backend),
        ).to(self.device)
        return model

    def _maybe_load_checkpoint(self, d_in: int) -> None:
        ckpt = self.cfg.checkpoint_path
        if ckpt is None:
            return
        if self._loaded_from_checkpoint and bool(self.cfg.freeze_loaded_checkpoint):
            return
        if not os.path.exists(ckpt):
            print(f"[channelmamba] checkpoint not found at {ckpt}; starting from scratch")
            return
        if self.model is None:
            self.model = self._build_model(d_in)
        try:
            state = torch.load(ckpt, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load ChannelMamba checkpoint at '{ckpt}': {exc}") from exc
        saved_metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
        expected_metadata = dict(self.cfg.checkpoint_metadata or {})
        expected_metadata["d_in"] = int(d_in)
        if expected_metadata:
            mismatch = []
            for key, expected_value in expected_metadata.items():
                saved_value = saved_metadata.get(key)
                if saved_value != expected_value:
                    mismatch.append((key, saved_value, expected_value))
            if mismatch:
                mismatch_str = ", ".join([f"{k}: saved={sv}, expected={ev}" for k, sv, ev in mismatch])
                if bool(self.cfg.allow_mismatch_reset):
                    print(
                        "[channelmamba] checkpoint metadata mismatch; ignoring checkpoint and starting from scratch: "
                        f"{mismatch_str}"
                    )
                    return
                raise ValueError(
                    "ChannelMamba checkpoint metadata mismatch for "
                    f"'{ckpt}': {mismatch_str}"
                )
        state_dict = state.get("model_state_dict", state)
        try:
            self.model.load_state_dict(state_dict)
        except Exception as exc:
            raise RuntimeError(f"Failed to restore ChannelMamba model weights from '{ckpt}': {exc}") from exc
        self.model.eval()
        self._loaded_from_checkpoint = True
        if saved_metadata:
            print(f"[channelmamba] loaded checkpoint metadata: {saved_metadata}")

    def load_checkpoint_only(self, d_in: int | None = None) -> None:
        """Load checkpoint weights without running offline fitting."""
        if d_in is None:
            ckpt = self.cfg.checkpoint_path
            if ckpt is None or not os.path.exists(ckpt):
                raise ValueError(
                    "ChannelMamba eval mode requires d_in or a valid checkpoint "
                    f"at '{ckpt}'."
                )
            state = torch.load(ckpt, map_location="cpu")
            metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
            if "d_in" not in metadata:
                raise ValueError(
                    "ChannelMamba checkpoint metadata is missing 'd_in'; "
                    "cannot initialize eval-only predictor."
                )
            d_in = int(metadata["d_in"])
        if self.model is None:
            self.model = self._build_model(int(d_in))
        self.d_in = int(d_in)
        self._maybe_load_checkpoint(int(d_in))
        if not self._loaded_from_checkpoint:
            raise ValueError(
                "ChannelMamba eval mode requires a valid checkpoint, "
                f"but none was loaded from '{self.cfg.checkpoint_path}'."
            )
        self.model.eval()

    def fit_offline(
        self,
        h_freq_csi_history: np.ndarray | Sequence[np.ndarray],
        ns3cfg,
        num_bs_ant: int = 4,
        num_ue_ant: int = 2,
    ) -> None:
        """Fit model from offline slot history.

        Behavior depends on mode:
        - eval/frozen mode: load checkpoint and skip fitting.
        - train mode: if checkpoint exists, warm-start from it and continue fitting.
        """

        rng_seed = int(self.cfg.seed)
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)

        samples_x: List[np.ndarray] = []
        samples_y: List[np.ndarray] = []
        prev_len = int(self.cfg.prev_len)
        pred_len = int(self.cfg.pred_len)

        if isinstance(h_freq_csi_history, np.ndarray):
            h_freq_csi_histories = [h_freq_csi_history]
        else:
            h_freq_csi_histories = [np.asarray(h_hist) for h_hist in h_freq_csi_history]

        for drop_seq_idx, curr_drop_h_hist in enumerate(h_freq_csi_histories):
            per_drop_windows = 0
            for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
                for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
                    tx_ant_idx = self._to_pair_indices(tx_node_idx, num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)
                    rx_ant_idx = self._to_pair_indices(rx_node_idx, num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)

                    curr_h = curr_drop_h_hist[:, :, :, rx_ant_idx, :, ...]
                    curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]
                    features, n_rb, _, _, _, _ = self._compress_history_to_features(curr_h)

                    d_in = 2 * n_rb
                    if self.d_in is None:
                        self.d_in = d_in
                    elif self.d_in != d_in:
                        raise ValueError(f"Inconsistent d_in across links: existing={self.d_in}, new={d_in}")

                    for m_idx in range(features.shape[0]):
                        seq = features[m_idx]  # [T, d_in]
                        if seq.shape[0] < prev_len + pred_len:
                            continue
                        for t_idx in range(prev_len, seq.shape[0] - pred_len + 1):
                            x = seq[t_idx - prev_len:t_idx]
                            y = seq[t_idx:t_idx + pred_len]
                            samples_x.append(x.astype(np.float32))
                            samples_y.append(y.astype(np.float32))
                            per_drop_windows += 1
            # print(f"[channelmamba] drop-seq-{drop_seq_idx}: generated {per_drop_windows} offline windows")

        if self.d_in is None:
            raise ValueError("No valid training samples built for ChannelMamba.")

        if self.model is None:
            self.model = self._build_model(self.d_in)
        self._maybe_load_checkpoint(self.d_in)

        if self._loaded_from_checkpoint and bool(self.cfg.freeze_loaded_checkpoint):
            return
        if len(samples_x) == 0:
            raise ValueError("No offline windows were generated for ChannelMamba training.")

        x_np = np.asarray(np.stack(samples_x), dtype=np.float32, order="C")
        y_np = np.asarray(np.stack(samples_y), dtype=np.float32, order="C")

        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=int(self.cfg.batch_size), shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch_idx in range(int(self.cfg.epochs)):
            epoch_loss = 0.0
            num_batches = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                num_batches += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            # print(f"[channelmamba] offline epoch={epoch_idx+1}/{self.cfg.epochs}, mse={avg_loss:.6e}")
        self.model.eval()
        if self.cfg.checkpoint_path:
            metadata = dict(self.cfg.checkpoint_metadata or {})
            metadata["d_in"] = int(self.d_in) if self.d_in is not None else None
            torch.save({"model_state_dict": self.model.state_dict(), "metadata": metadata}, self.cfg.checkpoint_path)
            print(f"[channelmamba] saved checkpoint to {self.cfg.checkpoint_path}")

    def fit_offline_pair(
        self,
        h_freq_csi_history_pair: np.ndarray | Sequence[np.ndarray],
    ) -> None:
        """Fit a predictor for a single tx/rx node pair using pooled drop histories."""

        rng_seed = int(self.cfg.seed)
        torch.manual_seed(rng_seed)
        np.random.seed(rng_seed)

        samples_x: List[np.ndarray] = []
        samples_y: List[np.ndarray] = []
        prev_len = int(self.cfg.prev_len)
        pred_len = int(self.cfg.pred_len)

        if isinstance(h_freq_csi_history_pair, np.ndarray):
            pair_histories = [h_freq_csi_history_pair]
        else:
            pair_histories = [np.asarray(h_hist) for h_hist in h_freq_csi_history_pair]

        for drop_seq_idx, pair_hist in enumerate(pair_histories):
            features, n_rb, _, _, _, _ = self._compress_history_to_features(pair_hist)
            d_in = 2 * n_rb
            if self.d_in is None:
                self.d_in = d_in
            elif self.d_in != d_in:
                raise ValueError(f"Inconsistent d_in across pooled drops: existing={self.d_in}, new={d_in}")

            per_drop_windows = 0
            for m_idx in range(features.shape[0]):
                seq = features[m_idx]  # [T, d_in]
                if seq.shape[0] < prev_len + pred_len:
                    continue
                for t_idx in range(prev_len, seq.shape[0] - pred_len + 1):
                    x = seq[t_idx - prev_len:t_idx]
                    y = seq[t_idx:t_idx + pred_len]
                    samples_x.append(x.astype(np.float32))
                    samples_y.append(y.astype(np.float32))
                    per_drop_windows += 1
            # print(f"[channelmamba] pair drop-seq-{drop_seq_idx}: generated {per_drop_windows} offline windows")

        if self.d_in is None:
            raise ValueError("No valid training samples built for ChannelMamba pair training.")

        if self.model is None:
            self.model = self._build_model(self.d_in)
        self._maybe_load_checkpoint(self.d_in)

        if self._loaded_from_checkpoint and bool(self.cfg.freeze_loaded_checkpoint):
            return
        if len(samples_x) == 0:
            raise ValueError("No offline windows were generated for ChannelMamba pair training.")

        x_np = np.asarray(np.stack(samples_x), dtype=np.float32, order="C")
        y_np = np.asarray(np.stack(samples_y), dtype=np.float32, order="C")

        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=int(self.cfg.batch_size), shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        criterion = nn.MSELoss()

        self.model.train()
        for epoch_idx in range(int(self.cfg.epochs)):
            epoch_loss = 0.0
            num_batches = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                num_batches += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            # print(f"[channelmamba] pair offline epoch={epoch_idx+1}/{self.cfg.epochs}, mse={avg_loss:.6e}")
        self.model.eval()
        if self.cfg.checkpoint_path:
            metadata = dict(self.cfg.checkpoint_metadata or {})
            metadata["d_in"] = int(self.d_in) if self.d_in is not None else None
            torch.save({"model_state_dict": self.model.state_dict(), "metadata": metadata}, self.cfg.checkpoint_path)
            print(f"[channelmamba] saved pair checkpoint to {self.cfg.checkpoint_path}")

    @torch.no_grad()
    def predict_pair(self, h_freq_csi_history_pair: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("ChannelMamba pair predictor is not initialized. Run fit_offline_pair first.")

        prev_len = int(self.cfg.prev_len)
        features, n_rb, n_sc, n_sym, n_r, n_t = self._compress_history_to_features(h_freq_csi_history_pair)
        features = self._ensure_prev_len(features, prev_len=prev_len)
        model_in = torch.from_numpy(features[:, -prev_len:, :]).to(self.device)
        pred = self.model(model_in).detach().cpu().numpy()[:, 0, :]  # one-step: [M, d_in]
        return self._reconstruct_block_from_features(pred, n_rb=n_rb, n_sc=n_sc, n_sym=n_sym, n_r=n_r, n_t=n_t)

    @torch.no_grad()
    def predict_all_links(self, h_freq_csi_history: np.ndarray, ns3cfg, num_bs_ant: int = 4, num_ue_ant: int = 2):
        if self.model is None:
            raise ValueError("ChannelMamba predictor is not initialized. Run fit_offline first.")

        h_freq_csi = np.zeros(h_freq_csi_history[0, ...].shape, dtype=h_freq_csi_history.dtype)
        prev_len = int(self.cfg.prev_len)

        for tx_node_idx in range(ns3cfg.num_txue_sel + 1):
            for rx_node_idx in range(ns3cfg.num_rxue_sel + 1):
                tx_ant_idx = self._to_pair_indices(tx_node_idx, num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)
                rx_ant_idx = self._to_pair_indices(rx_node_idx, num_bs_ant=num_bs_ant, num_ue_ant=num_ue_ant)

                curr_h = h_freq_csi_history[:, :, :, rx_ant_idx, :, ...]
                curr_h = curr_h[:, :, :, :, :, tx_ant_idx, ...]

                features, n_rb, n_sc, n_sym, n_r, n_t = self._compress_history_to_features(curr_h)
                features = self._ensure_prev_len(features, prev_len=prev_len)
                model_in = torch.from_numpy(features[:, -prev_len:, :]).to(self.device)
                pred = self.model(model_in).detach().cpu().numpy()[:, 0, :]  # one-step: [M, d_in]
                tmp = self._reconstruct_block_from_features(pred, n_rb=n_rb, n_sc=n_sc, n_sym=n_sym, n_r=n_r, n_t=n_t)

                rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

        return h_freq_csi