from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Device setup: CUDA > MPS (if supported) > CPU

HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif HAS_MPS:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(
    f"CUDA available: {torch.cuda.is_available()}, "
    f"MPS available: {HAS_MPS}, "
    f"Device: {device}"
)


def initialize_rnn_hidden_state(dim_hid, n_layers, n_params):
    '''
    dim_his : hidden state dimension (for each param)
    n_layers : number of layers of LSTM
    n_params : number of parameters in the optimizee model
    '''
    h0 = torch.zeros(n_layers, n_params, dim_hid).requires_grad_(requires_grad=False).to(device)
    c0 = torch.zeros(n_layers, n_params, dim_hid).requires_grad_(requires_grad=False).to(device)

    return h0, c0


class MetaOptimizerLSTM(nn.Module):
    # ===== UNCHANGED from the original file =====

    def __init__(self, model, num_layers, hidden_size, preproc_factor=10.0, preproc=True,
                 param_chunk_size=None, double_backward="auto"):
        super(MetaOptimizerLSTM, self).__init__()
        self.meta_model = model
        # ---- double-backward (grad-of-grad) safety ----------------------
        # cuDNN's FUSED RNN kernels have no double-backward implementation:
        #     RuntimeError: the derivative for '_cudnn_rnn_backward' is not
        #     implemented. Double backwards is not supported for CuDNN RNNs.
        # GAM and GSAM differentiate THROUGH a gradient (grad-norm penalty /
        # gradient decomposition), so the meta-loss needs grad-of-grad across
        # the unrolled LSTM and the fused path explodes. SAM / No-Reg do not.
        #   "auto" (default) : use native kernels whenever autograd is
        #                      recording, i.e. exactly while the meta-graph is
        #                      being built. Correct for EVERY regularizer.
        #   True             : same as auto (force native while grad is on).
        #   False            : always allow cuDNN (only if you are certain no
        #                      double backward happens — e.g. SAM-only runs).
        # Cheap here: this LSTM runs with seq_len=1, and the fused kernel's
        # advantage is sequence unrolling, so there is little left to lose.
        # Meta-TEST runs under no_grad and keeps cuDNN either way.
        self.double_backward = double_backward
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.preproc = preproc
        # If set (e.g. 2_000_000), the coordinate-wise LSTM is applied in
        # chunks along the parameter dimension. Mathematically identical
        # (coordinates are independent), but bounds peak memory so that
        # ResNet-18 / VGG-scale optimizees (~11-15M params) fit on a V100.
        # Not part of state_dict, so it can be set after load_state_dict.
        self.param_chunk_size = param_chunk_size

        if self.preproc:
            self.first_order = nn.LSTM(2, self.hidden_size, self.num_layers, batch_first=True, bias=False)
        else:
            self.first_order = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True, bias=False)

        self.outputer = nn.Linear(hidden_size, 1, bias=False)
        self.outputer.weight.data.mul_(0.1)
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

    def reset_lstm(self, keep_states=False, model=None):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states:
            self.h0 = self.h0.data
            self.c0 = self.c0.data
        else:
            self.h0, self.c0 = initialize_rnn_hidden_state(
                self.hidden_size, self.num_layers, self.meta_model.get_flat_params().size(0)
            )

    def _preprocess(self, x):
        if self.preproc:
            inp = torch.zeros(x.size()[0], 2).to(device)
            keep_grads = (torch.abs(x) >= self.preproc_threshold).squeeze()
            inp[:, 0][keep_grads] = (torch.log(torch.abs(x[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp[:, 1][keep_grads] = torch.sign(x[keep_grads]).squeeze()
            inp[:, 0][~keep_grads] = -1
            inp[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * x[~keep_grads]).squeeze()
            x = inp.unsqueeze(1)
            x = x.to(device)
        return x

    def snapshot_state(self):
        """Detached copies of (h0, c0) — for window-start replays (penalty
        mode re-unrolls) and for freezing state in the smoothing term."""
        self._assert_state()
        return self.h0.detach().clone(), self.c0.detach().clone()

    def restore_state(self, state):
        h0, c0 = state
        self.h0 = h0.detach().clone()
        self.c0 = c0.detach().clone()

    def _assert_state(self):
        if not hasattr(self, 'h0'):
            raise RuntimeError(
                "MetaOptimizerLSTM has no hidden state: call "
                "reset_lstm(keep_states=False, model=...) before the first "
                "forward/meta_update (this was the silent-crash mode of the "
                "old eval_opt_func — Change.MD item 21).")

    # cuDNN LSTM kernels fail (CUDNN_STATUS_INTERNAL_ERROR) or thrash for
    # absurd batch sizes; 11.17M coordinates in one call — or even the old
    # 2,000,000 suggestion — is absurd. Above this size we auto-chunk even
    # when the caller asked for none. Chunking is EXACT (coordinates are
    # independent; matching hidden-state slices), so this only bounds peak
    # transient memory / kernel batch, never changes results.
    SAFE_MAX_CHUNK = 262144

    def state_memory_bytes(self):
        """Persistent per-coordinate LSTM state (h0 + c0). This is the
        algorithm's state, NOT waste: memory = 2 x layers x n_params x
        hidden x 4B (e.g. ResNet-18: 3.33 GB fp32) and cannot be chunked
        away — but it fits comfortably on any HPC GPU."""
        return 2 * self.h0.numel() * self.h0.element_size() if hasattr(self, 'h0') else 0

    def set_double_backward(self, mode):
        """'auto' | True | False — see __init__. Safe to call any time."""
        self.double_backward = mode

    def _use_native_lstm(self, x):
        """True -> run the LSTM on native kernels (cuDNN locally disabled)."""
        if self.double_backward is False:
            return False
        if not x.is_cuda:
            return False                  # CPU/MPS already use native kernels
        if not torch.backends.cudnn.enabled:
            return False                  # cuDNN already off; nothing to do
        # "auto" and True: only while autograd is recording — that is the
        # only situation in which a (double) backward can be taken.
        return torch.is_grad_enabled()

    def _run_lstm(self, x, h, c):
        """THE single place the LSTM is invoked. cuDNN is disabled here and
        ONLY here (never globally) when grad-of-grad may be required; the
        autograd graph records the native ops, so the later backward — and
        the backward-of-backward — resolve correctly."""
        if self._use_native_lstm(x):
            if not getattr(self, "_native_note_printed", False):
                print("[meta_optimizer] LSTM forward on NATIVE kernels "
                      "(cuDNN locally disabled): cuDNN RNNs have no "
                      "double-backward, which GAM/GSAM meta-training needs. "
                      "GPU execution is unaffected.")
                self._native_note_printed = True
            with torch.backends.cudnn.flags(enabled=False):
                return self.first_order(x, (h, c))
        return self.first_order(x, (h, c))

    def forward(self, x):
        self._assert_state()
        x = self._preprocess(x)
        n = x.size(0)
        chunk = self.param_chunk_size
        if (chunk is None or chunk <= 0 or chunk > self.SAFE_MAX_CHUNK) \
                and n > self.SAFE_MAX_CHUNK:
            if not getattr(self, '_chunk_note_printed', False):
                print(f"[meta_optimizer] auto-chunking LSTM batch: n_params="
                      f"{n:,} > {self.SAFE_MAX_CHUNK:,} (requested chunk="
                      f"{chunk}) — using {self.SAFE_MAX_CHUNK:,} to stay "
                      f"inside cuDNN kernel limits (exact computation).")
                self._chunk_note_printed = True
            chunk = self.SAFE_MAX_CHUNK

        if chunk is None or chunk <= 0 or n <= chunk:
            output1, (hn1, cn1) = self._run_lstm(x, self.h0, self.c0)
            self.h0 = hn1
            self.c0 = cn1
            o1 = self.outputer(output1)
            return o1.squeeze()

        # Chunked path: coordinates are independent, so processing slices
        # of the parameter dimension with the matching hidden-state slices
        # gives exactly the same result with bounded peak memory.
        #
        # MEMORY (the exp5/exp4b OOM, root-caused): the old code kept THREE
        # copies of the full LSTM state alive at once — the old h0/c0
        # (VGG-16: 4.39 GB), the new_h/new_c chunk lists (another 4.39 GB),
        # and the torch.cat result (2.20 GB per tensor) — ~11 GB peak, and
        # the failing "Tried to allocate 2.20 GiB ... 10.01 GiB reserved"
        # in logs/eval_exp5_*.err was exactly that cat. Meta-TEST runs the
        # whole unroll under torch.no_grad() (optimizee_train.py wraps
        # meta_update), so there the state can be updated IN PLACE: zero
        # extra full-size allocations. When autograd IS recording (only if
        # someone meta-trains a chunk-sized net — not in the 5-experiment
        # plan), we keep the original cat path, because BPTT must flow
        # through the hidden-state chain and in-place writes would sever it.
        if not torch.is_grad_enabled():
            if self.h0.grad_fn is not None:      # stale history from a
                self.h0 = self.h0.detach()       # grad-enabled call; not
                self.c0 = self.c0.detach()       # needed under no_grad
            pieces = []
            for i in range(0, n, chunk):
                j = min(i + chunk, n)
                h_i = self.h0[:, i:j, :].contiguous()
                c_i = self.c0[:, i:j, :].contiguous()
                o, (hn, cn) = self._run_lstm(x[i:j], h_i, c_i)
                pieces.append(self.outputer(o))
                # h_i/c_i are copies, so writing back is safe and free of
                # read-after-write hazards.
                self.h0[:, i:j, :].copy_(hn)
                self.c0[:, i:j, :].copy_(cn)
                del o, hn, cn, h_i, c_i
            return torch.cat(pieces, dim=0).squeeze()

        outs, new_h, new_c = [], [], []
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            h_i = self.h0[:, i:j, :].contiguous()
            c_i = self.c0[:, i:j, :].contiguous()
            o, (hn, cn) = self._run_lstm(x[i:j], h_i, c_i)
            outs.append(self.outputer(o))
            new_h.append(hn)
            new_c.append(cn)
        self.h0 = torch.cat(new_h, dim=1)
        self.c0 = torch.cat(new_c, dim=1)
        return torch.cat(outs, dim=0).squeeze()

    def meta_update(self, model_with_grads):
        try:                       # measured meta-training cost counter
            import flops_utils as _fu
            _fu.bump_meta_update(bptt=torch.is_grad_enabled())
        except Exception:
            pass
        # collect gradients from optimizee (data only, no graph)
        grads = []
        for param in model_with_grads.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1).unsqueeze(-1))

        self.meta_model.copy_params_from(model_with_grads)
        flat_params = self.meta_model.get_flat_params()
        flat_grads = torch.cat(grads)
        if flat_grads.numel() != flat_params.numel():
            raise RuntimeError(
                f"meta_update: gradient count ({flat_grads.numel()}) != "
                f"parameter count ({flat_params.numel()}). An optimizee "
                f"parameter has grad=None or a non weight/bias parameter "
                f"exists — alignment would silently corrupt updates.")

        inputs = flat_grads.view(-1, 1).unsqueeze(1)

        update = self(inputs)

        flat_params = flat_params + update

        self.meta_model.set_flat_params(flat_params)

        # copy data back so the caller's model sees the update
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model


# =============================================================================
# MetaModel — PATCHED to preserve the autograd graph through optimizee params.
#
# THE BUG:
#   The previous implementation used `param.data = X`, which STRIPS the autograd
#   graph_fn from X. That detaches the optimizee parameters from the LSTM, so
#   when the meta-loss is backpropagated nothing reaches the LSTM weights and
#   `LOs_optimizer.step()` becomes a no-op. Result: the LSTM never trains, and
#   every regularizer config (No Reg / SAM / L2) saves byte-identical checkpoints
#   (the seed-1 random init).
#
# THE FIX:
#   Replace `module._parameters[name]` directly with the new tensor instead of
#   assigning to `.data`. The new tensor keeps its `grad_fn` and the BPTT chain
#   stays intact from optimizee output -> LSTM weights.
#
#   This matches the working reference implementation in
#   extras/dmo/gam_reference/GAM_trial_with_comments.ipynb.
# =============================================================================
class MetaModel:

    def __init__(self, model):
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def reset(self):
        """
        Detach optimizee parameters from any old computational graph by
        replacing them with their `.data` (a plain Tensor, no grad_fn).
        This wipes the BPTT chain cleanly between unroll groups.
        """
        for module in self.model.modules():
            if len(module._parameters) == 0:
                continue
            if module._parameters.get('weight') is not None:
                module._parameters['weight'] = module._parameters['weight'].data
            if module._parameters.get('bias') is not None:
                module._parameters['bias'] = module._parameters['bias'].data

    def get_flat_params(self):
        params = []
        for module in self.model.modules():
            if len(module._parameters) == 0:
                continue
            if module._parameters.get('weight') is not None:
                params.append(module._parameters['weight'].view(-1))
            if module._parameters.get('bias') is not None:
                params.append(module._parameters['bias'].view(-1))
        return torch.cat(params)

    def set_flat_params(self, flat_params):
        """
        Install the (autograd-connected) updated parameters into the model.

        CRITICAL: assign DIRECTLY to `module._parameters[name]`, NOT to
        `param.data`. The new value keeps its `grad_fn`, so backprop through
        a forward pass of this model reaches the LSTM weights.
        """
        offset = 0
        for module in self.model.modules():
            if len(module._parameters) == 0:
                continue
            if module._parameters.get('weight') is not None:
                weight_shape = module._parameters['weight'].size()
                weight_flat_size = reduce(mul, weight_shape, 1)
                module._parameters['weight'] = flat_params[
                    offset:offset + weight_flat_size
                ].view(*weight_shape)
                offset += weight_flat_size
            if module._parameters.get('bias') is not None:
                bias_shape = module._parameters['bias'].size()
                bias_flat_size = reduce(mul, bias_shape, 1)
                module._parameters['bias'] = flat_params[
                    offset:offset + bias_flat_size
                ].view(*bias_shape)
                offset += bias_flat_size

    def copy_params_from(self, model):
        """Copy raw data: source `model` -> meta-wrapped model. No graph."""
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        """Copy raw data: meta-wrapped model -> caller's `model`. No graph."""
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)