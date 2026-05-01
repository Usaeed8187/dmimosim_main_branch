from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import os
import time
import multiprocessing
from contextlib import contextmanager
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
    grad_clip: float = 0.0
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
    enable_tf32: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int | None = 2
    torch_compile: bool = False
    torch_compile_mode: str | None = None

@contextmanager
def _timed_section(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[channelmamba][timing] {name}: {elapsed:.3f}s")

class DMIMOChannelMambaPredictor:
    """Offline-trained, online-inference ChannelMamba predictor for dMIMO CSI."""

    def __init__(self, cfg: DMIMOChannelMambaConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model: ChannelMamba | None = None
        self.d_in: int | None = None
        self._loaded_from_checkpoint = False
        self._configure_runtime_backends()

    def _configure_runtime_backends(self) -> None:
        use_cuda = self.device.type == "cuda"
        enable_tf32 = bool(getattr(self.cfg, "enable_tf32", True))
        if use_cuda and enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    def _build_dataloader(self, dataset: TensorDataset, shuffle: bool = True) -> DataLoader:
        use_cuda = self.device.type == "cuda"
        requested_workers = int(getattr(self.cfg, "dataloader_num_workers", 4))
        requested_workers = max(0, requested_workers)
        cpu_count = multiprocessing.cpu_count()
        num_workers = min(requested_workers, cpu_count)

        pin_memory = bool(getattr(self.cfg, "dataloader_pin_memory", True)) and use_cuda
        persistent_workers = bool(getattr(self.cfg, "dataloader_persistent_workers", True)) and num_workers > 0
        prefetch_factor = getattr(self.cfg, "dataloader_prefetch_factor", 2)

        dataloader_kwargs = {
            "batch_size": int(self.cfg.batch_size),
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if num_workers > 0 and prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
        return DataLoader(dataset, **dataloader_kwargs)
        
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
        return features shape: [N_sym*N_sc, T, 2*N_r*N_t] ((sym, sc) are parallel streams)
        """

        hist = np.asarray(curr_h)
        t_hist, _, _, n_r, _, n_t, n_sym, n_sc = hist.shape
        hist = hist[:, 0, 0, :, 0, :, :, :]  # [T, N_r, N_t, N_sym, N_sc]

        feat_real = np.real(hist)
        feat_imag = np.imag(hist)
        # Keep (sym, sc) as the stream axis to maximize sample multiplicity while vectorizing antennas.
        stream_feat = np.stack([feat_real, feat_imag], axis=3)  # [T, N_r, N_t, 2, N_sym, N_sc]
        feat = stream_feat.transpose(4, 5, 0, 1, 2, 3).reshape(n_sym * n_sc, t_hist, 2 * n_r * n_t)
        return feat.astype(np.float32), n_sc, n_sc, n_sym, n_r, n_t
    
    def _reconstruct_block_from_features(
        self,
        pred_features: np.ndarray,
        n_sc: int,
        n_sym: int,
        n_r: int,
        n_t: int,
    ) -> np.ndarray:
        """Reconstruct full-sc/symbol CSI block from one-step prediction features."""

        # Inverse of _compress_history_to_features() stream layout:
        # [N_sym*N_sc, T, 2*N_r*N_t] -> one-step [N_sym*N_sc, 2*N_r*N_t].
        # Stream order is (sym, sc). Restore per-stream antenna vectors and map back.
        pred_features = pred_features.reshape(n_sym, n_sc, n_r, n_t, 2)
        sym_sc_real = pred_features[..., 0].transpose(2, 3, 0, 1)  # [N_r, N_t, N_sym, N_sc]
        sym_sc_imag = pred_features[..., 1].transpose(2, 3, 0, 1)
        sym_vals = sym_sc_real + 1j * sym_sc_imag

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
        compile_enabled = bool(getattr(self.cfg, "torch_compile", False))
        if compile_enabled and hasattr(torch, "compile"):
            compile_mode = getattr(self.cfg, "torch_compile_mode", None)
            try:
                model = torch.compile(model, mode=compile_mode)
                print(f"[channelmamba] torch.compile enabled (mode={compile_mode!r})")
            except Exception as exc:
                print(f"[channelmamba] torch.compile unavailable; falling back to eager mode: {exc}")
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
                    features, _n_sc_meta, n_sc, n_sym, n_r, n_t = self._compress_history_to_features(curr_h)

                    d_in = 2 * n_r * n_t
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

        with _timed_section("fit_offline_pair.numpy_stack"):
            x_np = np.asarray(np.stack(samples_x), dtype=np.float32, order="C")
            y_np = np.asarray(np.stack(samples_y), dtype=np.float32, order="C")

        with _timed_section("fit_offline_pair.tensorize_dataset"):
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            y_tensor = torch.tensor(y_np, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, y_tensor)

        with _timed_section("fit_offline_pair.build_dataloader"):
            loader = self._build_dataloader(dataset, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.cfg.epochs)),
        )
        criterion = nn.MSELoss()
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.model.train()
        for epoch_idx in range(int(self.cfg.epochs)):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            num_batches = 0
            transfer_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            optim_time = 0.0
            for x_batch, y_batch in loader:
                t0 = time.perf_counter()
                x_batch = x_batch.to(self.device, non_blocking=use_amp)
                y_batch = y_batch.to(self.device, non_blocking=use_amp)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                transfer_time += time.perf_counter() - t0
                optimizer.zero_grad()
                t1 = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = self.model(x_batch)
                    loss = criterion(pred, y_batch)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                forward_time += time.perf_counter() - t1

                t2 = time.perf_counter()
                scaler.scale(loss).backward()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                backward_time += time.perf_counter() - t2
                clip_val = float(getattr(self.cfg, "grad_clip", 0.0) or 0.0)
                if clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
                t3 = time.perf_counter()
                scaler.step(optimizer)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                optim_time += time.perf_counter() - t3
                scaler.update()
                epoch_loss += float(loss.item())
                num_batches += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            scheduler.step()
            epoch_elapsed = time.perf_counter() - epoch_start
            print(
                f"[channelmamba] offline epoch={epoch_idx+1}/{self.cfg.epochs}, "
                f"mse={avg_loss:.6e}, epoch_time_sec={epoch_elapsed:.3f}"
            )
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

        with _timed_section("fit_offline_pair.prepare_histories"):
            if isinstance(h_freq_csi_history_pair, np.ndarray):
                pair_histories = [h_freq_csi_history_pair]
            else:
                pair_histories = [np.asarray(h_hist) for h_hist in h_freq_csi_history_pair]

        with _timed_section("fit_offline_pair.build_windows"):
            for drop_seq_idx, pair_hist in enumerate(pair_histories):
                features, _n_sc_meta, _, _, n_r, n_t = self._compress_history_to_features(pair_hist)
                d_in = 2 * n_r * n_t
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

        with _timed_section("fit_offline_pair.numpy_stack"):
            x_np = np.asarray(np.stack(samples_x), dtype=np.float32, order="C")
            y_np = np.asarray(np.stack(samples_y), dtype=np.float32, order="C")

        with _timed_section("fit_offline_pair.tensorize_dataset"):
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            y_tensor = torch.tensor(y_np, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, y_tensor)

        with _timed_section("fit_offline_pair.build_dataloader"):
            loader = self._build_dataloader(dataset, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.cfg.epochs)),
        )
        criterion = nn.MSELoss()
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.model.train()
        for epoch_idx in range(int(self.cfg.epochs)):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            num_batches = 0
            transfer_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            optim_time = 0.0
            for x_batch, y_batch in loader:
                t0 = time.perf_counter()
                x_batch = x_batch.to(self.device, non_blocking=use_amp)
                y_batch = y_batch.to(self.device, non_blocking=use_amp)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                transfer_time += time.perf_counter() - t0

                optimizer.zero_grad()
                t1 = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = self.model(x_batch)
                    loss = criterion(pred, y_batch)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                forward_time += time.perf_counter() - t1

                t2 = time.perf_counter()
                scaler.scale(loss).backward()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                backward_time += time.perf_counter() - t2
                clip_val = float(getattr(self.cfg, "grad_clip", 0.0) or 0.0)
                if clip_val > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_val)
                t3 = time.perf_counter()
                scaler.step(optimizer)
                scaler.update()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                optim_time += time.perf_counter() - t3
                epoch_loss += float(loss.item())
                num_batches += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            scheduler.step()
            epoch_elapsed = time.perf_counter() - epoch_start
            print(
                f"[channelmamba] pair offline epoch={epoch_idx+1}/{self.cfg.epochs}, "
                f"mse={avg_loss:.6e}, epoch_time_sec={epoch_elapsed:.3f}, "
                f"transfer_sec={transfer_time:.3f}, forward_sec={forward_time:.3f}, "
                f"backward_sec={backward_time:.3f}, optim_sec={optim_time:.3f}"
            )
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
        t_pre = time.perf_counter()
        features, _n_sc_meta, n_sc, n_sym, n_r, n_t = self._compress_history_to_features(h_freq_csi_history_pair)
        features = self._ensure_prev_len(features, prev_len=prev_len)
        t_h2d = time.perf_counter()
        model_in = torch.from_numpy(features[:, -prev_len:, :]).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_fwd = time.perf_counter()
        pred = self.model(model_in)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        pred = pred.detach().cpu().numpy()[:, 0, :]  # one-step: [M, d_in]
        t_post = time.perf_counter()
        print(
            f"[channelmamba][timing] predict_pair prep_sec={t_h2d - t_pre:.3f}, "
            f"h2d_sec={t_fwd - t_h2d:.3f}, forward_d2h_sec={t_post - t_fwd:.3f}"
        )
        return self._reconstruct_block_from_features(pred, n_sc=n_sc, n_sym=n_sym, n_r=n_r, n_t=n_t)

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

                features, _n_sc_meta, n_sc, n_sym, n_r, n_t = self._compress_history_to_features(curr_h)
                features = self._ensure_prev_len(features, prev_len=prev_len)
                model_in = torch.from_numpy(features[:, -prev_len:, :]).to(self.device)
                pred = self.model(model_in).detach().cpu().numpy()[:, 0, :]  # one-step: [M, d_in]
                tmp = self._reconstruct_block_from_features(pred, n_sc=n_sc, n_sym=n_sym, n_r=n_r, n_t=n_t)

                rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)

        return h_freq_csi