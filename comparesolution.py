import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# 1. ĐỊNH NGHĨA BÀI TOÁN
# ===================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

X_STAR = torch.tensor([0.0, 0.0], device=DEVICE, requires_grad=False)
CENTER = torch.tensor([2.0, 2.0], device=DEVICE)
R_OBSTACLE = 1.0

def phi(x: torch.Tensor) -> torch.Tensor:
    """φ(x) = 1 - ||x - (2,2)||"""
    dist = torch.norm(x - CENTER, dim=1, keepdim=True)
    return R_OBSTACLE - dist

def grad_phi(x: torch.Tensor) -> torch.Tensor:
    """∇φ(x) = -(x - c) / ||x - c||"""
    diff = x - CENTER
    dist = torch.norm(diff, dim=1, keepdim=True) + 1e-12
    return -diff / dist

# ===================================================================
# 2. MẠNG THẦN KINH VÀ TRIAL SOLUTION
# ===================================================================
class PINN(nn.Module):
    def __init__(self, hidden_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim, 1)
        )
        self.phi_x_star_val = 1.0 - np.sqrt(8.0)
        self.phi_x_star = torch.tensor([self.phi_x_star_val], device=DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ξ_N(x;θ) = φ(x*) + |x-x*|² * NN(x;θ) (Equation 3.18)"""
        r2 = torch.sum((x - X_STAR)**2, dim=1, keepdim=True)
        return self.phi_x_star + r2 * self.net(x)

# ===================================================================
# 3. THUẬT TOÁN 1 - CHÍNH XÁC THEO PAPER
# ===================================================================
class Algorithm1Solver:
    def __init__(self, M: int = 2000):
        self.M = M  # Sample size M
        self.model = PINN().to(DEVICE)
        self.loss_history = []
        # ε cho density function ν2 (cố định hoặc thay đổi chậm)
        self.epsilon_density = 0.15
        
    def compute_density_nu1(self, x: torch.Tensor) -> torch.Tensor:
        """Tính mật độ ν1(θ) cho G1: tập trung vào {u > φ} ∪ {(x-x*)·∇φ ≤ 0}"""
        with torch.no_grad():
            u = self.model(x)
            phi_val = phi(x)
            grad_phi_val = grad_phi(x)
            
            diff = x - X_STAR
            dot_product = torch.sum(diff * grad_phi_val, dim=1, keepdim=True)
            
            # Điểm thuộc G1
            in_G1_cond1 = (u > phi_val).squeeze()
            in_G1_cond2 = (dot_product <= 0).squeeze()
            not_x_star = torch.norm(diff, dim=1) > 1e-4
            
            in_G1 = (in_G1_cond1 | in_G1_cond2) & not_x_star
            
            # Mật độ: baseline 0.1, tăng lên 1.0 cho điểm trong G1
            density = torch.ones_like(u) * 0.1
            density[in_G1] = 1.0
            
        return density
    
    def compute_density_nu2(self, x: torch.Tensor, epsilon: float = None) -> torch.Tensor:
        """Tính mật độ ν2(θ) cho G2: tập trung vào biên {|u-φ| ≤ ε} ∩ {(x-x*)·∇φ > 0}"""
        if epsilon is None:
            epsilon = self.epsilon_density
        
        with torch.no_grad():
            u = self.model(x)
            phi_val = phi(x)
            grad_phi_val = grad_phi(x)
            
            diff = x - X_STAR
            dot_product = torch.sum(diff * grad_phi_val, dim=1, keepdim=True)
            
            # Điểm gần biên {|u-φ| ≤ ε} và (x-x*)·∇φ > 0
            u_near_phi = torch.abs(u - phi_val).squeeze() <= epsilon
            dot_gt_zero = (dot_product > 0).squeeze()
            not_x_star = torch.norm(diff, dim=1) > 1e-4
            
            in_G2_candidate = u_near_phi & dot_gt_zero & not_x_star
            
            # Mật độ: baseline 0.1, tăng lên 1.0 cho điểm candidate
            density = torch.ones_like(u) * 0.1
            density[in_G2_candidate] = 1.0
            
        return density
    
    def compute_density_nu3(self, x: torch.Tensor) -> torch.Tensor:
        """Tính mật độ ν3(θ) cho Ω: tập trung vào vi phạm constraint {φ > u}"""
        with torch.no_grad():
            u = self.model(x)
            phi_val = phi(x)
            
            # Điểm vi phạm constraint u < φ
            violates_constraint = (phi_val > u).squeeze()
            
            # Mật độ: baseline 0.1, tăng lên 1.0 cho điểm vi phạm
            density = torch.ones_like(u) * 0.1
            density[violates_constraint] = 1.0
            
        return density
    
    def importance_sampling(self, density_func, n_points: int, **kwargs) -> torch.Tensor:
        """Importance sampling theo mật độ đã cho"""
        # Tạo nhiều proposal samples
        n_proposal = n_points * 10
        proposal = torch.rand(n_proposal, 2, device=DEVICE) * 4.0  # [0,4] × [0,4]
        
        # Tính mật độ đích
        if 'epsilon' in kwargs:
            target_density = density_func(proposal, kwargs['epsilon'])
        else:
            target_density = density_func(proposal)
        
        # Đảm bảo mật độ dương
        target_density = torch.clamp(target_density, min=1e-12)
        
        # Chuẩn hóa thành phân bố xác suất
        probabilities = target_density.squeeze() / (torch.sum(target_density) + 1e-12)
        
        # Kiểm tra probabilities hợp lệ
        if torch.any(torch.isnan(probabilities)) or torch.sum(probabilities) == 0:
            probabilities = torch.ones_like(probabilities) / len(probabilities)
        
        # Lấy mẫu importance
        indices = torch.multinomial(probabilities, n_points, replacement=True)
        return proposal[indices].requires_grad_(True)
    
    def compute_loss(self, epsilon_n: float):
        """Tính loss J_n theo Algorithm 1"""
        try:
            # ========== DÒNG 5-6: SAMPLE VÀ XÁC ĐỊNH G1 ==========
            x_samples = self.importance_sampling(self.compute_density_nu1, self.M)
            
            u_x = self.model(x_samples)
            phi_x = phi(x_samples)
            grad_phi_x = grad_phi(x_samples)
            diff_x = x_samples - X_STAR
            dot_x = torch.sum(diff_x * grad_phi_x, dim=1, keepdim=True)
            
            in_G1_cond1 = (u_x > phi_x)
            in_G1_cond2 = (dot_x <= 0)
            not_x_star_x = torch.norm(diff_x, dim=1, keepdim=True) > 1e-4
            in_G1_mask = ((in_G1_cond1 | in_G1_cond2) & not_x_star_x).squeeze()
            
            # ========== DÒNG 7-8: SAMPLE VÀ XÁC ĐỊNH G2 ==========
            # QUAN TRỌNG: ν2 dùng epsilon_density, G2 dùng epsilon_n
            y_samples = self.importance_sampling(
                self.compute_density_nu2, self.M, epsilon=self.epsilon_density
            )
            
            u_y = self.model(y_samples)
            phi_y = phi(y_samples)
            grad_phi_y = grad_phi(y_samples)
            diff_y = y_samples - X_STAR
            dot_y = torch.sum(diff_y * grad_phi_y, dim=1, keepdim=True)
            
            # Điều kiện G2: |ξ_N - φ| ≤ ε_n và (x-x*)·∇φ > 0
            u_eq_phi = torch.abs(u_y - phi_y).squeeze() <= epsilon_n  # Dùng epsilon_n!
            dot_gt_zero = (dot_y > 0).squeeze()
            not_x_star_y = torch.norm(diff_y, dim=1) > 1e-4
            in_G2_mask = (u_eq_phi & dot_gt_zero & not_x_star_y)
            
            # ========== DÒNG 9-10: SAMPLE VÀ XÁC ĐỊNH G3 ==========
            z_samples = self.importance_sampling(self.compute_density_nu3, self.M)
            
            u_z = self.model(z_samples)
            phi_z = phi(z_samples)
            in_G3_mask = (phi_z > u_z).squeeze()
            
            # ========== DÒNG 11: TÍNH CÁC THÀNH PHẦN LOSS ==========
            
            # J_n^(1) = 1/|G1| Σ_{xm∈G1} [(xm - x*)·∇ξ_N(xm;θ)]²
            if in_G1_mask.any():
                x_G1 = x_samples[in_G1_mask]
                u_G1 = self.model(x_G1)
                grad_u_G1 = torch.autograd.grad(
                    u_G1.sum(), x_G1, create_graph=True, retain_graph=True
                )[0]
                diff_G1 = x_G1 - X_STAR
                residuals_G1 = torch.sum(diff_G1 * grad_u_G1, dim=1)
                J1 = torch.mean(residuals_G1**2)
            else:
                J1 = torch.tensor(0.0, device=DEVICE)
            
            # J_n^(2) = 1/|G2| Σ_{ym∈G2} [(ym - x*)·(∇ξ_N(ym;θ) - ∇φ(ym))]²
            if in_G2_mask.any():
                y_G2 = y_samples[in_G2_mask]
                u_G2 = self.model(y_G2)
                grad_u_G2 = torch.autograd.grad(
                    u_G2.sum(), y_G2, create_graph=True, retain_graph=True
                )[0]
                grad_phi_G2 = grad_phi(y_G2)
                diff_G2 = y_G2 - X_STAR
                residuals_G2 = torch.sum(diff_G2 * (grad_u_G2 - grad_phi_G2), dim=1)
                J2 = torch.mean(residuals_G2**2)
            else:
                J2 = torch.tensor(0.0, device=DEVICE)
            
            # J_n^(3) = 1/|G3| Σ_{zm∈G3} [φ(zm) - ξ_N(zm;θ)]²
            if in_G3_mask.any():
                z_G3 = z_samples[in_G3_mask]
                u_G3 = self.model(z_G3)
                phi_G3 = phi(z_G3)
                J3 = torch.mean((phi_G3 - u_G3)**2)
            else:
                J3 = torch.tensor(0.0, device=DEVICE)
            
            # Tổng loss
            total_loss = J1 + J2 + J3
            
            loss_details = {
                'total': total_loss.item(),
                'J1': J1.item() if isinstance(J1, torch.Tensor) else J1,
                'J2': J2.item() if isinstance(J2, torch.Tensor) else J2,
                'J3': J3.item() if isinstance(J3, torch.Tensor) else J3,
                '|G1|': in_G1_mask.sum().item(),
                '|G2|': in_G2_mask.sum().item(),
                '|G3|': in_G3_mask.sum().item(),
                'epsilon_n': epsilon_n,
                'epsilon_density': self.epsilon_density
            }
            
            return total_loss, loss_details
            
        except Exception as e:
            print(f"Error in compute_loss: {e}")
            import traceback
            traceback.print_exc()
            
            # Trả về loss nhỏ để tiếp tục training
            dummy_loss = torch.tensor(0.01, device=DEVICE, requires_grad=True)
            return dummy_loss, {
                'total': 0.01, 'J1': 0.0, 'J2': 0.0, 'J3': 0.0,
                '|G1|': 0, '|G2|': 0, '|G3|': 0,
                'epsilon_n': epsilon_n, 'epsilon_density': self.epsilon_density
            }
    
    def train(self, n_max: int = 5000, tol: float = 1e-5, print_every: int = 500):
        """Thực hiện Algorithm 1 chính xác"""
        # Dòng 1: θ ← θ_0
        n = 0
        J_n = float('inf')
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        # Dòng 4: tolerance sequence {ε_n} ↘ 0
        epsilon_n = 0.2  # ε_0
        
        print("=" * 80)
        print("ALGORITHM 1: PINN FOR VISIBILITY PROBLEM")
        print("=" * 80)
        print(f"M = {self.M} (sample size)")
        print(f"n_max = {n_max} (max iterations)")
        print(f"tol = {tol} (convergence threshold)")
        print(f"ε_0 = {epsilon_n} (initial tolerance)")
        print(f"ε_density = {self.epsilon_density} (for ν2 density)")
        print("=" * 80)
        
        start_time = time.time()
        
        while n < n_max and J_n > tol:
            optimizer.zero_grad()
            
            # Dòng 5-11: Tính loss
            loss, loss_details = self.compute_loss(epsilon_n)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at iteration {n}")
                n += 1
                continue
            
            # Dòng 12: θ ← θ - λ_n ∇_θ J(ξ_N(x;θ))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Cập nhật
            J_n = loss_details['total']
            self.loss_history.append(loss_details)
            
            # Giảm ε_n theo sequence: ε_{n+1} = 0.995 * ε_n
            epsilon_n = max(epsilon_n * 0.995, 1e-6)
            
            # Có thể giảm epsilon_density từ từ (tùy chọn)
            # self.epsilon_density = max(self.epsilon_density * 0.997, 0.05)
            
            # In thông tin
            if n % print_every == 0 or n == n_max - 1 or J_n <= tol:
                print(f"Iter {n:5d}: J_n = {J_n:8.6f}, "
                      f"J1={loss_details['J1']:7.5f}, J2={loss_details['J2']:7.5f}, J3={loss_details['J3']:7.5f}, "
                      f"ε_n={epsilon_n:.4f}")
                print(f"       |G1|={loss_details['|G1|']:4d}, |G2|={loss_details['|G2|']:4d}, "
                      f"|G3|={loss_details['|G3|']:4d}")
            
            # Dòng 13: n ← n + 1
            n += 1
        
        training_time = time.time() - start_time
        
        print("=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Total iterations: {n}")
        print(f"Final J_n: {J_n:.6f}")
        print(f"Final ε_n: {epsilon_n:.6f}")
        print(f"Training time: {training_time:.2f} seconds")
        print("=" * 80)
        
        return self.loss_history

# ===================================================================
# 4. VẼ ĐỒ THỊ VÀ KIỂM TRA
# ===================================================================
def plot_solution_comparison(solver: Algorithm1Solver):
    """Vẽ so sánh nghiệm PINN và nghiệm giải tích"""
    print("\n" + "="*80)
    print("VISUALIZING RESULTS")
    print("="*80)
    
    # Tạo lưới mịn hơn
    x = np.linspace(0, 4, 300)
    y = np.linspace(0, 4, 300)
    X, Y = np.meshgrid(x, y)
    
    # Tính nghiệm PINN
    solver.model.eval()
    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), 
                     dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        Z_pinn = solver.model(xy).cpu().numpy().reshape(X.shape)
    
    # Tính nghiệm giải tích chính xác hơn
    print("Computing exact solution...")
    Z_exact = np.zeros_like(X)
    points = np.linspace(0, 1, 200)
    
    for i in range(len(x)):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(x)} rows")
        for j in range(len(y)):
            x_val, y_val = X[i, j], Y[i, j]
            
            # Tại x* = (0,0)
            if abs(x_val) < 1e-10 and abs(y_val) < 1e-10:
                Z_exact[i, j] = 1.0 - np.sqrt(8.0)
                continue
            
            # Tính max_{t∈[0,1]} φ(tx)
            x_vals = x_val * points
            y_vals = y_val * points
            phi_vals = 1.0 - np.sqrt((x_vals - 2)**2 + (y_vals - 2)**2)
            Z_exact[i, j] = np.max(phi_vals)
    
    # Xác định miền vật cản
    theta = np.linspace(0, 2*np.pi, 300)
    obstacle_x = 2 + R_OBSTACLE * np.cos(theta)
    obstacle_y = 2 + R_OBSTACLE * np.sin(theta)
    
    # Visible sets
    visible_pinn = (Z_pinn > 0)
    visible_exact = (Z_exact > 0)
    error = np.abs(Z_pinn - Z_exact)
    
    # Cài đặt style
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 100
    
    # ================= HÌNH 1: SO SÁNH NGHIỆM =================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1.1 Nghiệm PINN
    im1 = axes1[0, 0].contourf(X, Y, Z_pinn, levels=100, cmap='RdYlBu_r', alpha=0.9)
    axes1[0, 0].contour(X, Y, Z_pinn, levels=[0], colors='black', linewidths=3)
    axes1[0, 0].plot(obstacle_x, obstacle_y, 'k-', linewidth=3, label='Vật cản')
    axes1[0, 0].fill_between(obstacle_x, obstacle_y, color='gray', alpha=0.6)
    axes1[0, 0].scatter(0, 0, s=200, c='red', marker='*', edgecolors='black', 
                       zorder=10, label='x* = (0,0)')
    axes1[0, 0].set_title('Nghiệm PINN: ξ_N(x;θ)', fontsize=14, fontweight='bold')
    axes1[0, 0].set_xlabel('x₁', fontsize=12)
    axes1[0, 0].set_ylabel('x₂', fontsize=12)
    axes1[0, 0].set_aspect('equal')
    axes1[0, 0].grid(True, alpha=0.3)
    axes1[0, 0].legend(loc='upper right', framealpha=0.9)
    cbar1 = plt.colorbar(im1, ax=axes1[0, 0])
    cbar1.set_label('Giá trị', fontsize=11)
    
    # 1.2 Nghiệm giải tích
    im2 = axes1[0, 1].contourf(X, Y, Z_exact, levels=100, cmap='RdYlBu_r', alpha=0.9)
    axes1[0, 1].contour(X, Y, Z_exact, levels=[0], colors='black', linewidths=3)
    axes1[0, 1].plot(obstacle_x, obstacle_y, 'k-', linewidth=3, label='Vật cản')
    axes1[0, 1].fill_between(obstacle_x, obstacle_y, color='gray', alpha=0.6)
    axes1[0, 1].scatter(0, 0, s=200, c='red', marker='*', edgecolors='black', 
                       zorder=10, label='x* = (0,0)')
    axes1[0, 1].set_title('Nghiệm giải tích: ψ(x)', fontsize=14, fontweight='bold')
    axes1[0, 1].set_xlabel('x₁', fontsize=12)
    axes1[0, 1].set_ylabel('x₂', fontsize=12)
    axes1[0, 1].set_aspect('equal')
    axes1[0, 1].grid(True, alpha=0.3)
    axes1[0, 1].legend(loc='upper right', framealpha=0.9)
    cbar2 = plt.colorbar(im2, ax=axes1[0, 1])
    cbar2.set_label('Giá trị', fontsize=11)
    
    # 1.3 So sánh chồng lên nhau
    im3 = axes1[1, 0].contourf(X, Y, Z_pinn, levels=50, cmap='coolwarm', alpha=0.7)
    axes1[1, 0].contour(X, Y, Z_exact, levels=20, colors='black', linewidths=1, alpha=0.7)
    axes1[1, 0].plot(obstacle_x, obstacle_y, 'k-', linewidth=3)
    axes1[1, 0].fill_between(obstacle_x, obstacle_y, color='gray', alpha=0.6)
    
    # Đường biên visible set
    axes1[1, 0].contour(X, Y, Z_pinn, levels=[0], colors='red', linewidths=3, 
                       linestyles='-', label='PINN (ξ_N=0)')
    axes1[1, 0].contour(X, Y, Z_exact, levels=[0], colors='blue', linewidths=3, 
                       linestyles='--', label='Giải tích (ψ=0)')
    
    axes1[1, 0].scatter(0, 0, s=200, c='red', marker='*', edgecolors='black', zorder=10)
    axes1[1, 0].set_title('So sánh PINN và Giải tích', fontsize=14, fontweight='bold')
    axes1[1, 0].set_xlabel('x₁', fontsize=12)
    axes1[1, 0].set_ylabel('x₂', fontsize=12)
    axes1[1, 0].set_aspect('equal')
    axes1[1, 0].grid(True, alpha=0.3)
    axes1[1, 0].legend(loc='upper right', framealpha=0.9)
    cbar3 = plt.colorbar(im3, ax=axes1[1, 0])
    cbar3.set_label('Giá trị PINN', fontsize=11)
    
    # 1.4 Sai số
    im4 = axes1[1, 1].contourf(X, Y, error, levels=100, cmap='hot_r', alpha=0.9)
    axes1[1, 1].plot(obstacle_x, obstacle_y, 'w-', linewidth=2)
    axes1[1, 1].fill_between(obstacle_x, obstacle_y, color='gray', alpha=0.6)
    axes1[1, 1].scatter(0, 0, s=150, c='red', marker='*', edgecolors='white', zorder=10)
    axes1[1, 1].set_title('Sai số tuyệt đối: |ξ_N - ψ|', fontsize=14, fontweight='bold')
    axes1[1, 1].set_xlabel('x₁', fontsize=12)
    axes1[1, 1].set_ylabel('x₂', fontsize=12)
    axes1[1, 1].set_aspect('equal')
    cbar4 = plt.colorbar(im4, ax=axes1[1, 1])
    cbar4.set_label('Sai số', fontsize=11)
    
    plt.tight_layout()
    plt.suptitle('KẾT QUẢ ALGORITHM 1: BÀI TOÁN VISIBILITY', fontsize=16, fontweight='bold', y=1.02)
    plt.show()
    
    # ================= HÌNH 2: VISIBLE SETS =================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # 2.1 Visible set PINN
    axes2[0].contourf(X, Y, visible_pinn, cmap='RdYlGn', alpha=0.8, 
                     levels=[-0.5, 0.5, 1.5])
    axes2[0].plot(obstacle_x, obstacle_y, 'k-', linewidth=2)
    axes2[0].fill_between(obstacle_x, obstacle_y, color='gray', alpha=0.5)
    axes2[0].scatter(0, 0, s=100, c='red', marker='*', edgecolors='black', zorder=10)
    axes2[0].set_title(f'Visible Set PINN\n({np.mean(visible_pinn)*100:.1f}% diện tích)', 
                      fontsize=13, fontweight='bold')
    axes2[0].set_xlabel('x₁')
    axes2[0].set_ylabel('x₂')
    axes2[0].set_aspect('equal')
    
    # 2.2 Visible set giải tích
    axes2[1].contourf(X, Y, visible_exact, cmap='RdYlGn', alpha=0.8, 
                     levels=[-0.5, 0.5, 1.5])
    axes2[1].plot(obstacle_x, obstacle_y, 'k-', linewidth=2)
    axes2[1].fill_between(obstacle_x, obstacle_y, color='gray', alpha=0.5)
    axes2[1].scatter(0, 0, s=100, c='red', marker='*', edgecolors='black', zorder=10)
    axes2[1].set_title(f'Visible Set Giải tích\n({np.mean(visible_exact)*100:.1f}% diện tích)', 
                      fontsize=13, fontweight='bold')
    axes2[1].set_xlabel('x₁')
    axes2[1].set_ylabel('x₂')
    axes2[1].set_aspect('equal')
    
    # 2.3 So sánh visible sets
    comparison = np.zeros_like(visible_pinn, dtype=int)
    comparison[(visible_pinn == 1) & (visible_exact == 1)] = 1  # Cả hai thấy được
    comparison[(visible_pinn == 1) & (visible_exact == 0)] = 2  # PINN thấy, giải tích không
    comparison[(visible_pinn == 0) & (visible_exact == 1)] = 3  # Giải tích thấy, PINN không
    
    cmap_custom = plt.cm.colors.ListedColormap(['white', 'green', 'orange', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap_custom.N)
    
    im = axes2[2].imshow(comparison, extent=[0, 4, 0, 4], origin='lower', 
                        cmap=cmap_custom, norm=norm, alpha=0.8)
    axes2[2].plot(obstacle_x, obstacle_y, 'k-', linewidth=2)
    axes2[2].fill_between(obstacle_x, obstacle_y, color='gray', alpha=0.5)
    axes2[2].scatter(0, 0, s=100, c='red', marker='*', edgecolors='black', zorder=10)
    axes2[2].set_title('So sánh Visible Sets', fontsize=13, fontweight='bold')
    axes2[2].set_xlabel('x₁')
    axes2[2].set_ylabel('x₂')
    axes2[2].set_aspect('equal')
    
    # Tạo custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Cả hai thấy được'),
        Patch(facecolor='orange', alpha=0.8, label='Chỉ PINN thấy'),
        Patch(facecolor='red', alpha=0.8, label='Chỉ Giải tích thấy'),
        Patch(facecolor='white', alpha=0.8, label='Không thấy'),
        Patch(facecolor='gray', alpha=0.5, label='Vật cản')
    ]
    axes2[2].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # ================= HÌNH 3: CONVERGENCE HISTORY =================
    if solver.loss_history:
        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = range(len(solver.loss_history))
        total_loss = [h['total'] for h in solver.loss_history]
        J1_values = [h['J1'] for h in solver.loss_history]
        J2_values = [h['J2'] for h in solver.loss_history]
        J3_values = [h['J3'] for h in solver.loss_history]
        epsilons = [h['epsilon_n'] for h in solver.loss_history]
        
        # 3.1 Total loss (log scale)
        axes3[0, 0].semilogy(iterations, total_loss, 'b-', linewidth=2, label='Total Loss')
        axes3[0, 0].set_xlabel('Iteration', fontsize=12)
        axes3[0, 0].set_ylabel('Loss (log scale)', fontsize=12)
        axes3[0, 0].set_title('Quá trình hội tụ tổng loss', fontsize=13, fontweight='bold')
        axes3[0, 0].grid(True, alpha=0.3)
        axes3[0, 0].legend(fontsize=11)
        
        # 3.2 Các thành phần loss
        axes3[0, 1].semilogy(iterations, J1_values, 'r-', linewidth=1.5, alpha=0.8, label='J1')
        axes3[0, 1].semilogy(iterations, J2_values, 'g-', linewidth=1.5, alpha=0.8, label='J2')
        axes3[0, 1].semilogy(iterations, J3_values, 'b-', linewidth=1.5, alpha=0.8, label='J3')
        axes3[0, 1].set_xlabel('Iteration', fontsize=12)
        axes3[0, 1].set_ylabel('Loss components (log scale)', fontsize=12)
        axes3[0, 1].set_title('Các thành phần loss', fontsize=13, fontweight='bold')
        axes3[0, 1].grid(True, alpha=0.3)
        axes3[0, 1].legend(fontsize=11)

        plt.tight_layout()
        plt.show()
    
def main():
    """Chạy thuật toán chính"""
    print("="*80)
    print("VISIBILITY PROBLEM USING PINN - ALGORITHM 1")
    print("="*80)
    
    start_time = time.time()
    
    # Khởi tạo solver với M=2000 điểm mẫu
    print("Initializing Algorithm 1 solver...")
    solver = Algorithm1Solver(M=2000)
    
    # Huấn luyện mô hình
    print("\nStarting training...")
    losses = solver.train(n_max=100000, tol=1e-5, print_every=1000)
    
    # Vẽ kết quả
    print("\nGenerating visualizations...")
    plot_solution_comparison(solver)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*80)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()