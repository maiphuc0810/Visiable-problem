import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List
import warnings
import time

warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ===================================================================
# 1. HÀM PHỤ TRỢ - GIỮ NGUYÊN CỦA BẠN
# ===================================================================

def phi(x, y):
    """φ(x) = 1 - (x-2)² - (y-2)² - Hàm mục tiêu của bài toán"""
    return 1 - (x - 2)**2 - (y - 2)**2

def gradient_phi(x, y):
    """∇φ(x) = [-2(x-2), -2(y-2)] - Gradient của hàm mục tiêu"""
    return np.array([-2.0 * (x - 2.0), -2.0 * (y - 2.0)])

def compute_exact_solution(grid_resolution=80):
    """Tính nghiệm giải tích ψ(x) = max_{t∈[0,1]} φ(tx, ty) - GIỮ NGUYÊN"""
    x = np.linspace(0, 4, grid_resolution)
    y = np.linspace(0, 4, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    Psi = np.zeros_like(X)
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            x_val, y_val = X[i, j], Y[i, j]
            
            if abs(x_val) < 1e-10 and abs(y_val) < 1e-10:
                Psi[i, j] = phi(0, 0)
                continue
            
            t = np.linspace(0, 1, 100)
            x_t = t * x_val
            y_t = t * y_val
            phi_vals = phi(x_t, y_t)
            Psi[i, j] = np.max(phi_vals)
    
    return X, Y, Psi

# ===================================================================
# 2. PINN NETWORK - CHỈ THAY ĐỔI ACTIVATION
# ===================================================================

class PINNNetwork_Activation(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=[64, 64, 64], output_dim=1, 
                 activation='tanh'):
        super(PINNNetwork_Activation, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Tạo các lớp ẩn - GIỮ NGUYÊN KIẾN TRÚC
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # CHỈ THAY ĐỔI Ở ĐÂY: cho phép chọn activation
            if activation == 'tanh':
                layers.append(nn.Tanh())  # Tanh activation
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())  # Sigmoid activation
            else:
                raise ValueError(f"Activation '{activation}' not supported")
            
            prev_dim = hidden_dim
        
        # Lớp đầu ra - GIỮ NGUYÊN
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Khởi tạo weights - GIỮ NGUYÊN
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

# ===================================================================
# 3. VISIBILITY PINN - THUẬT TOÁN GIỮ NGUYÊN 100%
# ===================================================================

class VisibilityPINN_Activation:
    def __init__(self, M=100, activation='tanh'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.M = M  # Số điểm mẫu - GIỮ NGUYÊN
        self.activation = activation
        
        # Tạo model với activation chỉ định
        self.model = PINNNetwork_Activation(
            input_dim=2, 
            hidden_layers=[64, 64, 64], 
            output_dim=1,
            activation=activation
        ).to(self.device)
        
        # Các tham số cố định - GIỮ NGUYÊN
        self.x_star_np = np.array([0.0, 0.0], dtype=np.float32)  # Điểm quan sát
        self.x_star = torch.tensor(self.x_star_np, dtype=torch.float32, requires_grad=False).to(self.device)
        self.domain_bounds = (np.array([0.0, 0.0], dtype=np.float32), 
                             np.array([4.0, 4.0], dtype=np.float32))  # Miền [0,4]x[0,4]
        self.phi_x_star = float(phi(self.x_star_np[0], self.x_star_np[1]))  # φ(x*)
        self.phi_x_star_tensor = torch.tensor([self.phi_x_star], dtype=torch.float32).to(self.device)
    
    def trial_solution(self, x: torch.Tensor) -> torch.Tensor:
        """ξ_N(x; θ) = φ(x*) + |x-x*|² * NN(x; θ) - GIỮ NGUYÊN"""
        if x.dtype != torch.float32:
            x = x.float()
        
        diff = x - self.x_star
        norm_squared = torch.sum(diff**2, dim=1, keepdim=True)  # |x-x*|²
        nn_output = self.model(x)  # NN(x; θ)
        
        return self.phi_x_star_tensor + norm_squared * nn_output
    
    def compute_gradients(self, x: torch.Tensor, create_graph: bool = True):
        """Tính gradient ∇ξ_N(x; θ) - GIỮ NGUYÊN"""
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        
        u = self.trial_solution(x)
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False
        )[0]
        
        return u, grad_u
    
    def _compute_phi_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Tính φ(x) cho batch điểm - GIỮ NGUYÊN LOGIC"""
        phi_values = []
        x_np = x.detach().cpu().numpy()
        for point in x_np:
            phi_values.append(phi(point[0], point[1]))
        return torch.tensor(phi_values, device=self.device, dtype=torch.float32).unsqueeze(1)
    
    def _compute_grad_phi_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Tính ∇φ(x) cho batch điểm - GIỮ NGUYÊN LOGIC"""
        grad_phi_values = []
        x_np = x.detach().cpu().numpy()
        for point in x_np:
            grad_phi_values.append(gradient_phi(point[0], point[1]))
        return torch.tensor(grad_phi_values, device=self.device, dtype=torch.float32)
    
    def uniform_sampling(self, n_points: int) -> torch.Tensor:
        """Lấy mẫu đều từ miền Ω - GIỮ NGUYÊN"""
        x_min, x_max = self.domain_bounds
        x_min_tensor = torch.tensor(x_min, device=self.device, dtype=torch.float32)
        x_max_tensor = torch.tensor(x_max, device=self.device, dtype=torch.float32)
        
        samples = torch.rand((n_points, 2), device=self.device, dtype=torch.float32)
        samples = samples * (x_max_tensor - x_min_tensor) + x_min_tensor
        
        return samples
    
    def compute_density_nu1(self, x: torch.Tensor) -> torch.Tensor:
        """ν1(θ) - density cho G1: {u > φ} ∪ {(x-x*)·∇φ ≤ 0} - GIỮ NGUYÊN"""
        with torch.no_grad():
            u = self.trial_solution(x)
            phi_val = self._compute_phi_batch(x)
            grad_phi = self._compute_grad_phi_batch(x)
            
            diff = x - self.x_star
            dot_grad_phi = torch.sum(diff * grad_phi, dim=1, keepdim=True)
            
            # Ưu tiên điểm u > φ hoặc (x-x*)·∇φ ≤ 0
            weight_u_gt_phi = torch.sigmoid((u - phi_val) * 10)  # u > φ
            weight_dot_le_zero = torch.sigmoid(-dot_grad_phi * 10)  # (x-x*)·∇φ ≤ 0
            density = weight_u_gt_phi + weight_dot_le_zero + 0.1
            
        return density
    
    def compute_density_nu2(self, x: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
        """ν2(θ) - density cho G2: {|u-φ| ≤ ε} ∩ {(x-x*)·∇φ > 0} - GIỮ NGUYÊN"""
        with torch.no_grad():
            u = self.trial_solution(x)
            phi_val = self._compute_phi_batch(x)
            grad_phi = self._compute_grad_phi_batch(x)
            
            diff = x - self.x_star
            dot_grad_phi = torch.sum(diff * grad_phi, dim=1, keepdim=True)
            
            # Ưu tiên điểm gần boundary (|u-φ| ≈ 0) VÀ (x-x*)·∇φ > 0
            weight_boundary = torch.exp(-torch.abs(u - phi_val) / (epsilon + 1e-8))
            weight_positive_dot = torch.sigmoid(dot_grad_phi * 10)
            density = weight_boundary * weight_positive_dot + 0.05
            
        return density
    
    def compute_density_nu3(self, x: torch.Tensor) -> torch.Tensor:
        """ν3(θ) - density cho G3: {φ > u} - GIỮ NGUYÊN"""
        with torch.no_grad():
            u = self.trial_solution(x)
            phi_val = self._compute_phi_batch(x)
            violation = torch.clamp(phi_val - u, min=0.0)
            density = torch.sigmoid(violation * 10) + 0.1
        return density
    
    def importance_sampling(self, density_func, n_points: int, epsilon: float = 0.05) -> torch.Tensor:
        """Importance sampling theo density - GIỮ NGUYÊN"""
        try:
            proposal_samples = self.uniform_sampling(n_points * 5)
            
            with torch.no_grad():
                if density_func.__name__ == 'compute_density_nu2':
                    target_density = density_func(proposal_samples, epsilon)
                else:
                    target_density = density_func(proposal_samples)
                
                target_density = torch.clamp(target_density, min=1e-8)
                probabilities = target_density.squeeze() / torch.sum(target_density)
            
            indices = torch.multinomial(probabilities, n_points, replacement=True)
            samples = proposal_samples[indices]
            
            return samples
            
        except Exception as e:
            return self.uniform_sampling(n_points)
    
    def compute_loss(self, epsilon: float = 1e-3):
        """Tính loss function J(θ) = J1 + J2 + J3 - GIỮ NGUYÊN THUẬT TOÁN"""
        try:
            # BƯỚC 1: Lấy mẫu các điểm - theo bài báo
            x_samples = self.importance_sampling(self.compute_density_nu1, self.M, epsilon)
            y_samples = self.importance_sampling(self.compute_density_nu2, self.M, epsilon)
            z_samples = self.importance_sampling(self.compute_density_nu3, self.M, epsilon)
            
            # BƯỚC 2: Xác định các tập G1, G2, G3
            
            # --- Tập G1 = {u > φ} ∪ {(x-x*)·∇φ(x) ≤ 0} \ {x*} ---
            x_samples = x_samples.requires_grad_(True)
            u_x, grad_u_x = self.compute_gradients(x_samples)
            phi_x = self._compute_phi_batch(x_samples)
            grad_phi_x = self._compute_grad_phi_batch(x_samples)
            
            diff_x = x_samples - self.x_star
            dot_grad_phi_x = torch.sum(diff_x * grad_phi_x, dim=1, keepdim=True)
            
            condition1 = (u_x > phi_x)                     # u > φ
            condition2 = (dot_grad_phi_x <= 0)             # (x-x*)·∇φ ≤ 0
            not_x_star = torch.norm(diff_x, dim=1, keepdim=True) > 1e-4  # Loại x*
            
            in_G1 = ((condition1 | condition2) & not_x_star).squeeze()
            
            if torch.any(in_G1):
                x_G1 = x_samples[in_G1]
                u_G1, grad_u_G1 = self.compute_gradients(x_G1)
                diff_G1 = x_G1 - self.x_star
                residuals_G1 = torch.sum(diff_G1 * grad_u_G1, dim=1, keepdim=True)
                J1 = torch.mean(residuals_G1 ** 2)  # E[(x-x*)·∇ξ_N]² trên G1
            else:
                J1 = torch.tensor(0.0, device=self.device)
            
            # --- Tập G2 = {|u-φ| ≤ ε} ∩ {(x-x*)·∇φ(x) > 0} \ {x*} ---
            y_samples = y_samples.requires_grad_(True)
            u_y, grad_u_y = self.compute_gradients(y_samples)
            phi_y = self._compute_phi_batch(y_samples)
            grad_phi_y = self._compute_grad_phi_batch(y_samples)
            
            diff_y = y_samples - self.x_star
            dot_grad_phi_y = torch.sum(diff_y * grad_phi_y, dim=1, keepdim=True)
            
            u_eq_phi = torch.abs(u_y - phi_y).squeeze() <= epsilon  # |u-φ| ≤ ε
            dot_gt_zero = (dot_grad_phi_y > 0).squeeze()           # (x-x*)·∇φ > 0
            not_x_star_y = torch.norm(diff_y, dim=1) > 1e-4        # Loại x*
            
            in_G2 = (u_eq_phi & dot_gt_zero & not_x_star_y)
            
            if torch.any(in_G2):
                y_G2 = y_samples[in_G2]
                u_G2, grad_u_G2 = self.compute_gradients(y_G2)
                grad_phi_G2 = self._compute_grad_phi_batch(y_G2)
                diff_G2 = y_G2 - self.x_star
                residuals_G2 = torch.sum(diff_G2 * (grad_u_G2 - grad_phi_G2), dim=1, keepdim=True)
                J2 = torch.mean(residuals_G2 ** 2)  # E[(x-x*)·(∇ξ_N - ∇φ)]² trên G2
            else:
                J2 = torch.tensor(0.0, device=self.device)
            
            # --- Tập G3 = {z : φ(z) > ξ_N(z;θ)} ---
            u_z = self.trial_solution(z_samples)
            phi_z = self._compute_phi_batch(z_samples)
            in_G3 = (phi_z > u_z).squeeze()
            
            if torch.any(in_G3):
                z_G3 = z_samples[in_G3]
                u_G3 = self.trial_solution(z_G3)
                phi_G3 = self._compute_phi_batch(z_G3)
                J3 = torch.mean((phi_G3 - u_G3) ** 2)  # E[(φ - ξ_N)²] trên G3
            else:
                J3 = torch.tensor(0.0, device=self.device)
            
            # Tổng loss: J(θ) = J1 + J2 + J3 - GIỮ NGUYÊN
            total_loss = J1 + J2 + J3
            
            loss_details = {
                'total': total_loss.item(),
                'J1': J1.item(),
                'J2': J2.item(),
                'J3': J3.item(),
                '|G1|': torch.sum(in_G1).item(),  # Số điểm trong G1
                '|G2|': torch.sum(in_G2).item(),  # Số điểm trong G2
                '|G3|': torch.sum(in_G3).item(),  # Số điểm trong G3
            }
            
            return total_loss, loss_details
            
        except Exception as e:
            # Xử lý lỗi - GIỮ NGUYÊN
            dummy_loss = torch.tensor(0.01, device=self.device, requires_grad=True)
            return dummy_loss, {
                'total': 0.01, 'J1': 0.0, 'J2': 0.0, 'J3': 0.0,
                '|G1|': 0, '|G2|': 0, '|G3|': 0
            }
    
    def train(self, n_iterations: int = 5000, learning_rate: float = 5e-4, 
              epsilon_start: float = 0.1, print_every: int = 1000):
        """Huấn luyện model - GIỮ NGUYÊN THUẬT TOÁN OPTIMIZATION"""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=200
        )
        
        epsilon = epsilon_start  # ε_n trong bài báo
        losses_history = []
        best_loss = float('inf')
        patience_counter = 0
        patience = 500  # Early stopping patience
        min_delta = 1e-6
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Tính loss với epsilon hiện tại
            loss, loss_details = self.compute_loss(epsilon=epsilon)
            
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            current_loss = loss_details['total']
            losses_history.append(loss_details)
            
            # Giảm epsilon theo thời gian: ε_n ↘ 0
            epsilon = max(epsilon * 0.995, 1e-5)
            
            # Early stopping
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
            
            # Điều chỉnh learning rate
            scheduler.step(current_loss)
            
            # In thông tin mỗi print_every iteration
            if iteration % print_every == 0:
                print(f"{self.activation.upper()}, {iteration}, {loss_details['total']:.6f}, {loss_details['J1']:.6f}, {loss_details['J2']:.6f}, {loss_details['J3']:.6f}, {loss_details['|G1|']}, {loss_details['|G2|']}, {loss_details['|G3|']}, {epsilon:.4f}")
        
        # In kết quả cuối cùng
        if losses_history:
            final = losses_history[-1]
            print(f"{self.activation.upper()}, FINAL, {final['total']:.6f}, {final['J1']:.6f}, {final['J2']:.6f}, {final['J3']:.6f}, {final['|G1|']}, {final['|G2|']}, {final['|G3|']}, {epsilon:.4f}")
        
        return losses_history
    
    def evaluate(self, grid_resolution=80):
        """Đánh giá nghiệm trên lưới - GIỮ NGUYÊN"""
        x = np.linspace(0, 4, grid_resolution, dtype=np.float32)
        y = np.linspace(0, 4, grid_resolution, dtype=np.float32)
        X, Y = np.meshgrid(x, y)
        
        points = np.column_stack([X.ravel(), Y.ravel()])
        U = np.zeros(len(points))
        
        batch_size = 1000
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            batch_tensor = torch.tensor(batch, device=self.device, dtype=torch.float32)
            
            with torch.no_grad():
                u_batch = self.trial_solution(batch_tensor).cpu().numpy()
            
            U[i:i+batch_size] = u_batch.squeeze()
        
        U = U.reshape(grid_resolution, grid_resolution)
        
        return X, Y, U

# ===================================================================
# 4. CHẠY SO SÁNH - CHỈ IN DATA
# ===================================================================

if __name__ == "__main__":
    print("="*60)
    print("SO SÁNH TANH vs SIGMOID - CHỈ DATA")
    print("="*60)
    print("Cấu hình: M=50, Iterations=5000, LR=5e-4")
    print("="*60)
    print("Dòng 1: Activation, Iter, Total, J1, J2, J3, |G1|, |G2|, |G3|, Epsilon")
    print("="*60)
    
    # Tính nghiệm giải tích (chỉ 1 lần)
    print("\n[1/3] Tính nghiệm giải tích...")
    X_exact, Y_exact, U_exact = compute_exact_solution(80)
    
    # ---------------------------------------------------------------
    # TRAINING TANH
    # ---------------------------------------------------------------
    print("\n[2/3] Training với TANH activation...")
    pinn_tanh = VisibilityPINN_Activation(M=50, activation='tanh')
    losses_tanh = pinn_tanh.train(n_iterations=5000, print_every=1000)
    
    # ---------------------------------------------------------------
    # TRAINING SIGMOID
    # ---------------------------------------------------------------
    print("\n[3/3] Training với SIGMOID activation...")
    pinn_sigmoid = VisibilityPINN_Activation(M=50, activation='sigmoid')
    losses_sigmoid = pinn_sigmoid.train(n_iterations=5000, print_every=1000)
    
    # ---------------------------------------------------------------
    # ĐÁNH GIÁ CUỐI CÙNG
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
    print("="*60)
    
    # Tính nghiệm cho Tanh
    X_tanh, Y_tanh, U_tanh = pinn_tanh.evaluate()
    error_tanh = np.abs(U_exact - U_tanh)
    visible_exact = U_exact >= 0
    visible_tanh = U_tanh >= 0
    classification_tanh = np.mean(visible_exact == visible_tanh) * 100
    
    # Tính nghiệm cho Sigmoid
    X_sigmoid, Y_sigmoid, U_sigmoid = pinn_sigmoid.evaluate()
    error_sigmoid = np.abs(U_exact - U_sigmoid)
    visible_sigmoid = U_sigmoid >= 0
    classification_sigmoid = np.mean(visible_exact == visible_sigmoid) * 100
    
    # ---------------------------------------------------------------
    # BẢNG SO SÁNH
    # ---------------------------------------------------------------
    print("\n[BẢNG 1] METRICS SO SÁNH")
    print("-" * 65)
    print(f"{'METRIC':<25} | {'TANH':>12} | {'SIGMOID':>12} | {'CHÊNH LỆCH':>12}")
    print("-" * 65)
    print(f"{'Final Loss':<25} | {losses_tanh[-1]['total']:12.6f} | {losses_sigmoid[-1]['total']:12.6f} | {abs(losses_tanh[-1]['total'] - losses_sigmoid[-1]['total']):12.6f}")
    print(f"{'Mean Error':<25} | {error_tanh.mean():12.6f} | {error_sigmoid.mean():12.6f} | {abs(error_tanh.mean() - error_sigmoid.mean()):12.6f}")
    print(f"{'Max Error':<25} | {error_tanh.max():12.6f} | {error_sigmoid.max():12.6f} | {abs(error_tanh.max() - error_sigmoid.max()):12.6f}")
    print(f"{'RMS Error':<25} | {np.sqrt(np.mean(error_tanh**2)):12.6f} | {np.sqrt(np.mean(error_sigmoid**2)):12.6f} | {abs(np.sqrt(np.mean(error_tanh**2)) - np.sqrt(np.mean(error_sigmoid**2))):12.6f}")
    print(f"{'Classification Acc (%)':<25} | {classification_tanh:12.2f} | {classification_sigmoid:12.2f} | {abs(classification_tanh - classification_sigmoid):12.2f}")
    print(f"{'Visible Area Exact (%)':<25} | {np.mean(visible_exact)*100:12.2f} | {np.mean(visible_exact)*100:12.2f} | {'N/A':>12}")
    print(f"{'Visible Area PINN (%)':<25} | {np.mean(visible_tanh)*100:12.2f} | {np.mean(visible_sigmoid)*100:12.2f} | {abs(np.mean(visible_tanh)*100 - np.mean(visible_sigmoid)*100):12.2f}")
    
    # ---------------------------------------------------------------
    # DATA RAW CHO PHÂN TÍCH
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("[BẢNG 2] LOSS HISTORY - TANH (mỗi 1000 iteration)")
    print("-" * 60)
    print("Iter, Total, J1, J2, J3")
    for i in range(0, len(losses_tanh), 1000):
        if i < len(losses_tanh):
            l = losses_tanh[i]
            print(f"{i}, {l['total']:.6f}, {l['J1']:.6f}, {l['J2']:.6f}, {l['J3']:.6f}")
    
    print("\n" + "="*60)
    print("[BẢNG 3] LOSS HISTORY - SIGMOID (mỗi 1000 iteration)")
    print("-" * 60)
    print("Iter, Total, J1, J2, J3")
    for i in range(0, len(losses_sigmoid), 1000):
        if i < len(losses_sigmoid):
            l = losses_sigmoid[i]
            print(f"{i}, {l['total']:.6f}, {l['J1']:.6f}, {l['J2']:.6f}, {l['J3']:.6f}")
    
    # ---------------------------------------------------------------
    # DATA DẠNG CSV ĐỂ COPY VÀO EXCEL
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("[BẢNG 4] FINAL COMPARISON CSV (copy to Excel)")
    print("="*60)
    print("Metric,Tanh,Sigmoid,Difference")
    print(f"Final Loss,{losses_tanh[-1]['total']:.6f},{losses_sigmoid[-1]['total']:.6f},{abs(losses_tanh[-1]['total'] - losses_sigmoid[-1]['total']):.6f}")
    print(f"Mean Error,{error_tanh.mean():.6f},{error_sigmoid.mean():.6f},{abs(error_tanh.mean() - error_sigmoid.mean()):.6f}")
    print(f"Max Error,{error_tanh.max():.6f},{error_sigmoid.max():.6f},{abs(error_tanh.max() - error_sigmoid.max()):.6f}")
    print(f"RMS Error,{np.sqrt(np.mean(error_tanh**2)):.6f},{np.sqrt(np.mean(error_sigmoid**2)):.6f},{abs(np.sqrt(np.mean(error_tanh**2)) - np.sqrt(np.mean(error_sigmoid**2))):.6f}")
    print(f"Classification Accuracy (%),{classification_tanh:.2f},{classification_sigmoid:.2f},{abs(classification_tanh - classification_sigmoid):.2f}")
    print(f"Visible Area PINN (%),{np.mean(visible_tanh)*100:.2f},{np.mean(visible_sigmoid)*100:.2f},{abs(np.mean(visible_tanh)*100 - np.mean(visible_sigmoid)*100):.2f}")