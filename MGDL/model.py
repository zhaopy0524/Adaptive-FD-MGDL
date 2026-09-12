"""MGDL neural-network architecture, training, and prediction."""

import copy
import time
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# -------------------- Custom Sin module (wraps torch.sin) --------------------
class Sin(nn.Module):
    """Sine activation module."""
    def forward(self, x):
        return torch.sin(x)


# -------------------- MGDL multi-grade neural network --------------------
class MGDL(nn.Module):
    """
    Multi-Grade Deep Learning (MGDL) framework.

    Args:
        input_dim (int): Input feature dimension (typically 2 for (x1, x2)).
        output_dim (int): Output dimension (typically 1).
        grade_hidden_dims (List[List[int]]): Hidden layer widths per grade.
            Example [[16], [32, 16], [64]] means:
                Grade 1: 1 hidden layer with 16 neurons;
                Grade 2: 2 hidden layers with 32 and 16 neurons;
                Grade 3: 1 hidden layer with 64 neurons.
        grade_activations (List[List[str]]): Activation names per hidden layer per grade.
            Example [['sin'], ['sin', 'relu'], ['relu']] matches grade_hidden_dims.
            Supported: 'sin', 'relu', 'tanh'.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 grade_hidden_dims: List[List[int]],
                 grade_activations: List[List[str]]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grade_hidden_dims = grade_hidden_dims
        self.grade_activations = grade_activations
        self.num_grades = len(grade_hidden_dims)
        assert len(grade_activations) == self.num_grades, \
            "grade_activations must have the same length as grade_hidden_dims"

        # Activation function map (classes; instantiated later)
        self.activation_map = {
            'sin': Sin,
            'relu': nn.ReLU,
            'tanh': nn.Tanh
        }

        # Networks for each grade (ModuleList ensures parameters are registered)
        self.grades = nn.ModuleList()
        # Features/outputs cached during training (internal use only)
        self._features = []
        self._cumulative_output = None
        self.loss_history = []
        self.rse_history = []
        self.grade_train_hparams = [None for _ in range(self.num_grades)]
        self.total_train_time_sec: Optional[float] = None
        # Operator-loss matrix A and RHS vector b (set externally)
        self.A: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None

    def _build_grade_net(self, grade_idx: int, in_dim: int, out_dim: int) -> nn.Sequential:
        """Build the network for the given grade and initialize parameters."""
        hidden_dims = self.grade_hidden_dims[grade_idx]
        activations = self.grade_activations[grade_idx]
        assert len(hidden_dims) == len(activations), \
            f"grade {grade_idx}: number of hidden layers and activations must match"

        layers = []
        prev_dim = in_dim
        for i, (h_dim, act_name) in enumerate(zip(hidden_dims, activations)):
            layers.append(nn.Linear(prev_dim, h_dim))
            act_class = self.activation_map.get(act_name)
            if act_class is None:
                raise ValueError(f"Unsupported activation: {act_name}")
            layers.append(act_class())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        net = nn.Sequential(*layers)

        linear_layers = [m for m in net if isinstance(m, nn.Linear)]
        if len(linear_layers) == 0:
            return net

        if grade_idx == 0:
            omega0 = 30.0
            for i, lin in enumerate(linear_layers):
                fan_in = lin.weight.shape[1]
                if i == 0:
                    bound = 1.0 / fan_in
                else:
                    bound = (np.sqrt(6.0 / fan_in) / omega0)
                with torch.no_grad():
                    lin.weight.uniform_(-bound, bound)
                    if lin.bias is not None:
                        lin.bias.uniform_(-bound, bound)
            return net
        
        #---------------------sin-2--------------------------
        if grade_idx == 1:
            omega0 = 30.0
            for i, lin in enumerate(linear_layers):
                fan_in = lin.weight.shape[1]
                if i == 0:
                    bound = 1.0 / fan_in
                else:
                    bound = (np.sqrt(6.0 / fan_in) / omega0)
                with torch.no_grad():
                    lin.weight.uniform_(-bound, bound)
                    if lin.bias is not None:
                        lin.bias.uniform_(-bound, bound)
                        
            out_lin = linear_layers[-1]
            with torch.no_grad():
                nn.init.zeros_(out_lin.weight)
                if out_lin.bias is not None:
                    nn.init.zeros_(out_lin.bias)
            return net

        for lin in linear_layers[:-1]:
            with torch.no_grad():
                nn.init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(lin.weight)
                    bound = np.sqrt(6.0 / (fan_in + fan_out))
                    lin.bias.uniform_(-bound, bound)

        out_lin = linear_layers[-1]
        with torch.no_grad():
            nn.init.zeros_(out_lin.weight)
            if out_lin.bias is not None:
                nn.init.zeros_(out_lin.bias)

        return net

    def _forward_features(self, grade_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Extract features of the given grade (forward up to, but excluding, the output layer)."""
        net = self.grades[grade_idx]
        # Iterate over all layers except the last one
        for layer in net[:-1]:
            x = layer(x)
        return x

    def _forward_output(self, grade_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass of the given grade (outputs the prediction)."""
        return self.grades[grade_idx](x)

    def _reinit_grade(self, grade_idx: int) -> None:
        """Re-initialize the parameters of the specified grade (used for retraining this grade only)."""
        if grade_idx == 0:
            in_dim = self.input_dim
        else:
            in_dim = self.grade_hidden_dims[grade_idx - 1][-1]
        out_dim = self.output_dim
        self.grades[grade_idx] = self._build_grade_net(grade_idx, in_dim, out_dim)

    def _train_one_grade(self,
                         grade_idx: int,
                         x: torch.Tensor,
                         target: torch.Tensor,
                         num_epochs: int,
                         lr: float,
                         device: torch.device,
                         lr_gamma: float = 1.0,
                         verbose: bool = True,
                         cumulative_prev: Optional[torch.Tensor] = None,
                         u_true: Optional[torch.Tensor] = None) -> tuple[List[float], List[float]]:
        """Train a single grade network and record the loss history including the initial evaluation."""
        net = self.grades[grade_idx].to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        loss_hist: List[float] = []
        rse_hist: List[float] = []

        if cumulative_prev is None:
            cumulative_prev = torch.zeros(x.shape[0], device=device)
        if u_true is not None:
            cumulative_prev = cumulative_prev.view(-1)
            u_true = u_true.view(-1)

        if self.A is None:
            raise ValueError("Operator matrix A is not set on the model (model.A is None).")
        if self.b is None:
            raise ValueError("RHS vector b is not set on the model (model.b is None).")
        A = self.A.to(device)
        b = self.b.to(device).view(-1)

        def loss_fn(output: torch.Tensor, _target_vec: torch.Tensor) -> torch.Tensor:
            out_flat = output.view(-1)
            pred_flat = cumulative_prev.view(-1) + out_flat
            res = A @ pred_flat - b
            return torch.mean(res ** 2)

        with torch.no_grad():
            output_init = net(x).squeeze()
            loss_init = loss_fn(output_init, target).item()
            loss_hist.append(loss_init)
            if u_true is not None:
                pred_init = cumulative_prev + output_init.view(-1)
                diff_true = u_true - pred_init
                rse_init = torch.sum(diff_true ** 2) / (torch.sum(u_true ** 2) + 1e-15)
                rse_hist.append(float(rse_init.item()))

        if verbose:
            print(f"  grade {grade_idx+1}, Initial Loss = {loss_init:.3e}")

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = net(x).squeeze()
            loss = loss_fn(output, target)
            loss_hist.append(loss.item())
            if u_true is not None:
                pred = cumulative_prev + output.view(-1)
                diff_true = u_true - pred
                rse = torch.sum(diff_true ** 2) / (torch.sum(u_true ** 2) + 1e-15)
                rse_hist.append(float(rse.item()))
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            # def closure():
            #     optimizer.zero_grad()
            #     output = net(x).squeeze()
            #     loss = loss_fn(output, target)
            #     loss.backward()
            #     return loss

            # loss = optimizer.step(closure)

            # with torch.no_grad():
            #     output = net(x).squeeze()
            #     loss_hist.append(loss_fn(output, target).item())
            #     if u_true is not None:
            #         pred = cumulative_prev + output.view(-1)
            #         diff_true = u_true - pred
            #         rse = torch.sum(diff_true ** 2) / (torch.sum(u_true ** 2) + 1e-15)
            #         rse_hist.append(float(rse.item()))

            if verbose and (epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs - 1):
                print(f"  grade {grade_idx+1}, Epoch {epoch:4d}, Loss = {loss.item():.3e}")

        return loss_hist, rse_hist

    def fit(self,
            X: Union[np.ndarray, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            grade_epochs: List[int],
            grade_lrs: List[float],
            grade_lr_gammas: Union[List[float], None] = None,
            device: Union[str, torch.device] = 'cpu',
            verbose: bool = True,
            u_true: Union[np.ndarray, torch.Tensor, None] = None,
            interactive: bool = False,
            continue_prompt: str = "Choose next step: Enter/next=accept current attempt, r=retrain current grade, p=pick final attempt from history, q=quit: ",
            input_fn: Callable[[str], str] = input) -> torch.Tensor:
        """Train all grade networks.

        Args:
            X: Input coordinates, shape (n_samples, input_dim).
            y: Target values, shape (n_samples,).
            grade_epochs: Number of epochs per grade.
            grade_lrs: Initial learning rate per grade.
            grade_lr_gammas: ExponentialLR gamma per grade (same length as num_grades).
                None means all 1.0 (no decay).
            device: Compute device.
            verbose: Whether to print training logs.
            u_true: Ground-truth solution vector used to record per-epoch RSE, shape (n_samples,).

        Returns:
            The final prediction on training points (cumulative sum), shape (n_samples,).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        X, y = X.to(device), y.to(device)
        n_samples = X.shape[0]

        # Rebuild grade networks on every fit() call so repeated fitting with
        # the same MGDL instance cannot reuse or append stale grade modules.
        self.grades = nn.ModuleList()
        self._features = []
        self._cumulative_output = None
        self.loss_history = [[] for _ in range(self.num_grades)]
        self.rse_history = [[] for _ in range(self.num_grades)]
        self.grade_train_hparams = [None for _ in range(self.num_grades)]
        self.total_train_time_sec = None

        u_true_t: Optional[torch.Tensor]
        if u_true is None:
            u_true_t = None
        else:
            if not isinstance(u_true, torch.Tensor):
                u_true_t = torch.tensor(u_true, dtype=torch.float32, device=device)
            else:
                u_true_t = u_true.to(device)
            u_true_t = u_true_t.view(-1)

        if len(grade_epochs) != self.num_grades:
            raise ValueError(
                f"grade_epochs length ({len(grade_epochs)}) must equal num_grades ({self.num_grades})"
            )
        if len(grade_lrs) != self.num_grades:
            raise ValueError(
                f"grade_lrs length ({len(grade_lrs)}) must equal num_grades ({self.num_grades})"
            )

        # Build networks grade by grade
        for l in range(self.num_grades):
            if l == 0:
                in_dim = self.input_dim
            else:
                # The last hidden width of the previous grade is used as the feature dimension
                in_dim = self.grade_hidden_dims[l-1][-1]
            out_dim = self.output_dim
            net = self._build_grade_net(l, in_dim, out_dim)
            self.grades.append(net)

        # Initialize cumulative output
        cumulative_output = torch.zeros(n_samples, device=device)
        # Current-grade input: grade 0 uses the raw coordinates
        current_input = X

        if grade_lr_gammas is None:
            grade_lr_gammas = [1.0] * self.num_grades
        if len(grade_lr_gammas) != self.num_grades:
            raise ValueError(
                f"grade_lr_gammas length ({len(grade_lr_gammas)}) must equal num_grades ({self.num_grades})"
            )

        # Local mutable copies (interactive retraining may change lr/lrT for the current grade)
        grade_lrs_cur = list(grade_lrs)
        grade_lr_gammas_cur = list(grade_lr_gammas)

        for l in range(self.num_grades):
            print(f"\n========== Training grade {l+1}/{self.num_grades} ==========")

            # At the start of this grade, cumulative output contains only grades < l
            cumulative_before = cumulative_output.detach().clone()

            def _record_grade_hparams(*, lr0: float, gamma: float, epochs: int, attempt_id: int, time_sec_selected: float) -> None:
                lr0_f = float(lr0)
                gamma_f = float(gamma)
                epochs_i = int(epochs)
                self.grade_train_hparams[l] = {
                    "epochs": epochs_i,
                    "lr0": lr0_f,
                    "gamma": gamma_f,
                    "lrT": lr0_f * (gamma_f ** epochs_i),
                    "attempt": int(attempt_id),
                    "num_attempts": int(len(attempts)),
                    "time_sec": float(time_sec_selected),
                }

            attempts = []
            attempt = 0
            while True:
                attempt += 1

                # Target residual = y - cumulative output from previous grades
                target_residual = y - cumulative_before

                attempt_t0 = time.perf_counter()
                loss_hist, rse_hist = self._train_one_grade(
                    grade_idx=l,
                    x=current_input,
                    target=target_residual,
                    num_epochs=grade_epochs[l],
                    lr=grade_lrs_cur[l],
                    lr_gamma=grade_lr_gammas_cur[l],
                    device=device,
                    verbose=verbose,
                    cumulative_prev=cumulative_before,
                    u_true=u_true_t
                )
                attempt_time_sec = time.perf_counter() - attempt_t0

                # Compute this grade's output (used to update the cumulative output)
                with torch.no_grad():
                    output_l = self._forward_output(l, current_input).squeeze()
                cumulative_output = cumulative_before + output_l

                # Save a snapshot of this attempt (you can pick any attempt as the final result later)
                attempts.append({
                    "attempt": attempt,
                    "loss_hist": loss_hist,
                    "rse_hist": rse_hist,
                    "state_dict": copy.deepcopy(self.grades[l].state_dict()),
                    "cumulative_output": cumulative_output.detach().clone().cpu(),
                    "lr0": grade_lrs_cur[l],
                    "gamma": grade_lr_gammas_cur[l],
                    "time_sec": float(attempt_time_sec),
                })

                if interactive:
                    loss_init = loss_hist[0] if len(loss_hist) > 0 else float("nan")
                    loss_last = loss_hist[-1] if len(loss_hist) > 0 else float("nan")
                    if len(rse_hist) > 0:
                        rse_init = rse_hist[0]
                        rse_last = rse_hist[-1]
                    else:
                        rse_init = float("nan")
                        rse_last = float("nan")
                    print(
                        f"[Grade {l+1} Summary | attempt {attempt}] operator-loss init={loss_init:.3e}, last={loss_last:.3e}; "
                        f"RSE init={rse_init:.3e}, last={rse_last:.3e}"
                    )

                    # Plot the grade RSE curve to help decide whether to retrain or move on
                    if len(rse_hist) > 0:
                        plt.figure()
                        plt.semilogy(range(1, len(rse_hist) + 1), rse_hist)
                        plt.xlabel("Epoch Index (includes initial)")
                        plt.ylabel("RSE")
                        plt.title(f"Grade {l+1} RSE Curve (attempt {attempt})")
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()

                # In non-interactive mode, accept this attempt and proceed
                if not interactive:
                    self.loss_history[l] = loss_hist
                    self.rse_history[l] = rse_hist
                    _record_grade_hparams(
                        lr0=float(grade_lrs_cur[l]),
                        gamma=float(grade_lr_gammas_cur[l]),
                        epochs=int(grade_epochs[l]),
                        attempt_id=int(attempt),
                        time_sec_selected=float(attempt_time_sec),
                    )
                    break

                ans = input_fn(continue_prompt).strip().lower()
                # Enter/next: accept current attempt; r: retrain current grade; p: pick final from history; q: quit
                if ans in ("", "next", "y", "yes", "1", "true", "t", "c", "continue"):
                    # Accept the current attempt as the final result for this grade
                    self.loss_history[l] = loss_hist
                    self.rse_history[l] = rse_hist
                    _record_grade_hparams(
                        lr0=float(grade_lrs_cur[l]),
                        gamma=float(grade_lr_gammas_cur[l]),
                        epochs=int(grade_epochs[l]),
                        attempt_id=int(attempt),
                        time_sec_selected=float(attempt_time_sec),
                    )
                    break
                if ans in ("r", "reinit", "retrain", "reset"):
                    cur_lr0 = grade_lrs_cur[l]
                    cur_lrT = cur_lr0 * (grade_lr_gammas_cur[l] ** grade_epochs[l])

                    lr0_in = input_fn(
                        f"Enter new learning rate lr0 (current {cur_lr0:g}, press Enter to keep): "
                    ).strip()
                    lrT_in = input_fn(
                        f"Enter new minimum learning rate lrT (current {cur_lrT:g}, press Enter to keep): "
                    ).strip()

                    new_lr0 = cur_lr0
                    new_lrT = cur_lrT
                    if lr0_in != "":
                        try:
                            new_lr0 = float(lr0_in)
                        except ValueError:
                            print("Cannot parse lr0; keeping it unchanged.")
                    if lrT_in != "":
                        try:
                            new_lrT = float(lrT_in)
                        except ValueError:
                            print("Cannot parse lrT; keeping it unchanged.")

                    if new_lr0 <= 0 or new_lrT <= 0:
                        print("lr0/lrT must be positive; retraining with previous hyperparameters.")
                    else:
                        grade_lrs_cur[l] = new_lr0
                        grade_lr_gammas_cur[l] = (new_lrT / new_lr0) ** (1.0 / grade_epochs[l])

                    print(
                        f"Re-train grade {l+1} (attempt reset) with lr0={grade_lrs_cur[l]:g}, "
                        f"lrT={grade_lrs_cur[l] * (grade_lr_gammas_cur[l] ** grade_epochs[l]):g}."
                    )
                    self._reinit_grade(l)
                    continue
                if ans in ("p", "pick", "choose", "select", "history"):
                    # Pick a previous attempt as the final result
                    pick_in = input_fn(
                        f"Enter attempt id to use as final (1..{len(attempts)}): "
                    ).strip()
                    if pick_in.isdigit():
                        pick_attempt = int(pick_in)
                        chosen = None
                        for a in attempts:
                            if a["attempt"] == pick_attempt:
                                chosen = a
                                break
                        if chosen is None:
                            print("Attempt id not found; waiting for next action.")
                            continue
                        # Roll back to the chosen parameters and cumulative output
                        grade_state = chosen["state_dict"]
                        self.grades[l].load_state_dict(grade_state)
                        grade_lrs_cur[l] = chosen["lr0"]
                        grade_lr_gammas_cur[l] = chosen["gamma"]
                        cumulative_output = chosen["cumulative_output"].to(device)
                        self.loss_history[l] = chosen["loss_hist"]
                        self.rse_history[l] = chosen["rse_hist"]
                        _record_grade_hparams(
                            lr0=float(chosen["lr0"]),
                            gamma=float(chosen["gamma"]),
                            epochs=int(grade_epochs[l]),
                            attempt_id=int(pick_attempt),
                            time_sec_selected=float(chosen.get("time_sec", float("nan"))),
                        )
                        print(f"Selected attempt {pick_attempt} as the final result for grade {l+1}.")

                        go_next = input_fn(
                            "Proceed to the next grade? Enter/next=proceed; q/end=stop: "
                        ).strip().lower()
                        if go_next in ("q", "quit", "exit", "stop", "e", "end"):
                            print("Exit training early.")
                            self._cumulative_output = cumulative_output
                            self.total_train_time_sec = float(
                                sum(
                                    float(cfg.get("time_sec", 0.0))
                                    for cfg in self.grade_train_hparams
                                    if cfg is not None and cfg.get("time_sec") is not None
                                )
                            )
                            return cumulative_output

                        break
                    print("Invalid input; waiting for next selection.")
                if ans in ("q", "quit", "exit", "stop", "e", "end"):
                    print("Exit training early.")
                    self._cumulative_output = cumulative_output
                    # Use the current attempt as the final result on early exit
                    self.loss_history[l] = loss_hist
                    self.rse_history[l] = rse_hist
                    _record_grade_hparams(
                        lr0=float(grade_lrs_cur[l]),
                        gamma=float(grade_lr_gammas_cur[l]),
                        epochs=int(grade_epochs[l]),
                        attempt_id=int(attempt),
                        time_sec_selected=float(attempt_time_sec),
                    )
                    self.total_train_time_sec = float(
                        sum(
                            float(cfg.get("time_sec", 0.0))
                            for cfg in self.grade_train_hparams
                            if cfg is not None and cfg.get("time_sec") is not None
                        )
                    )
                    return cumulative_output

                # Fallback: unrecognized input defaults to proceeding to the next grade
                print("Unrecognized input, defaulting to next grade.")
                self.loss_history[l] = loss_hist
                self.rse_history[l] = rse_hist
                _record_grade_hparams(
                    lr0=float(grade_lrs_cur[l]),
                    gamma=float(grade_lr_gammas_cur[l]),
                    epochs=int(grade_epochs[l]),
                    attempt_id=int(attempt),
                    time_sec_selected=float(attempt_time_sec),
                )
                break

            # If this is not the last grade, compute features for the next grade's input
            if l < self.num_grades - 1:
                with torch.no_grad():
                    current_input = self._forward_features(l, current_input)

        self._cumulative_output = cumulative_output
        self.total_train_time_sec = float(
            sum(
                float(cfg.get("time_sec", 0.0))
                for cfg in self.grade_train_hparams
                if cfg is not None and cfg.get("time_sec") is not None
            )
        )
        return cumulative_output

    def predict(self, X: Union[np.ndarray, torch.Tensor], device='cpu') -> np.ndarray:
        """Predict on new coordinates (sum of all grades).

        Args:
            X: Shape (n_samples, input_dim).

        Returns:
            NumPy array of shape (n_samples,).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=device)
        else:
            X = X.to(device)
        n_samples = X.shape[0]
        cumulative_output = torch.zeros(n_samples, device=device)
        current_input = X
        with torch.no_grad():
            for l in range(len(self.grades)):
                output_l = self._forward_output(l, current_input).squeeze()
                cumulative_output += output_l
                if l < len(self.grades) - 1:
                    current_input = self._forward_features(l, current_input)
        return cumulative_output.cpu().numpy()

    def print_network_structure(self) -> None:
        trained_grades = [
            l for l in range(self.num_grades)
            if l < len(self.loss_history) and len(self.loss_history[l]) > 0
        ]

        trained_total_params = sum(
            p.numel() for l in trained_grades for p in self.grades[l].parameters()
        )
        trained_trainable_params = sum(
            p.numel() for l in trained_grades for p in self.grades[l].parameters() if p.requires_grad
        )

        print("\n========== Network Structure ==========")
        print(
            f"MGDL(num_grades={self.num_grades}, trained_grades={len(trained_grades)}, input_dim={self.input_dim}, output_dim={self.output_dim})"
        )
        if self.total_train_time_sec is not None:
            print(f"Total selected training time (sum of chosen attempts): {self.total_train_time_sec:.3f} s")
        print(f"Trained params: {trained_total_params} (trainable {trained_trainable_params})")

        if len(trained_grades) == 0:
            print("No trained grades yet.")
            return

        for l in trained_grades:
            net = self.grades[l]
            grade_params = sum(p.numel() for p in net.parameters())
            grade_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
            in_dim = self.input_dim if l == 0 else self.grade_hidden_dims[l - 1][-1]
            hidden_dims = self.grade_hidden_dims[l]
            activations = self.grade_activations[l]
            print(
                f"\n[Grade {l+1}] in_dim={in_dim}, hidden_dims={hidden_dims}, activations={activations}, out_dim={self.output_dim}"
            )
            print(f"params={grade_params} (trainable {grade_trainable})")

            cfg = self.grade_train_hparams[l] if l < len(self.grade_train_hparams) else None
            if cfg is not None:
                epochs = int(cfg.get("epochs", 0))
                lr0 = float(cfg.get("lr0", float("nan")))
                lrT = float(cfg.get("lrT", float("nan")))
                gamma = float(cfg.get("gamma", float("nan")))
                attempt = cfg.get("attempt", None)
                num_attempts = cfg.get("num_attempts", None)
                t_sel = cfg.get("time_sec", None)
                extra = ""
                if num_attempts is not None:
                    extra += f", num_attempts={int(num_attempts)}"
                if t_sel is not None:
                    extra += f", time_selected={float(t_sel):.3f}s"
                print(
                    f"train: scheduler=ExponentialLR, epochs={epochs}, iters={epochs}, lr0={lr0:g}, lrT={lrT:g}, gamma={gamma:g}, attempt={attempt}{extra}"
                )
            else:
                print("train: (no hyperparams recorded)")

            print(net)

__all__ = ["MGDL", "Sin"]
