import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# =====================
# Config & Hyperparameters
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_LOW, DOMAIN_HIGH = 0.0, 4.0  # Miền [0,4] x [0,4]
X_STAR = torch.tensor([2, 2], dtype=torch.float32, device=DEVICE)  # Test với x* trong vật cản

HIDDEN_SIZE = 128
NUM_LAYERS = 4
ACT_FUNCTION = nn.Tanh()

LR_INIT = 1e-3
EPOCHS = 5000
BATCH_SIZE = 2048
EPS_INIT, EPS_FINAL = 0.1, 1e-4

# -----------------------------------------------------------------------------
# Models & Physics (SDF)
# -----------------------------------------------------------------------------
def sdf_circles(xy, centers, radii):
    if centers.numel() == 0:
        return torch.full((xy.shape[0],), -1.0, device=xy.device)
    diff = xy.unsqueeze(1) - centers.unsqueeze(0)
    dist = diff.norm(dim=2)
    return (radii.unsqueeze(0) - dist).max(dim=1).values

class VisibilityNet2D(nn.Module):
    def __init__(self, circ_centers, circ_radii):
        super().__init__()
        self.register_buffer("circ_centers", circ_centers.to(DEVICE))
        self.register_buffer("circ_radii", circ_radii.to(DEVICE))
        self.register_buffer("x_star", X_STAR)
        self.center = circ_centers[0]
        self.radius = circ_radii[0]
        
        with torch.no_grad():
            self.varphi_xs = sdf_circles(X_STAR.unsqueeze(0), circ_centers, circ_radii)[0]
            self.observer_in_obstacle = (self.varphi_xs > 0)
            
            print(f"\n{'='*50}")
            print(f"ĐIỂM QUAN SÁT: {X_STAR.cpu().numpy()}")
            print(f"φ(x*) = {self.varphi_xs:.4f}")
            print(f"Trạng thái: {'TRONG vật cản' if self.observer_in_obstacle else 'NGOÀI vật cản'}")
            print(f"{'='*50}\n")

        layers = [nn.Linear(2, HIDDEN_SIZE), ACT_FUNCTION]
        for _ in range(NUM_LAYERS - 2):
            layers.extend([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), ACT_FUNCTION])
        layers.append(nn.Linear(HIDDEN_SIZE, 1))
        self.net = nn.Sequential(*layers)

    def _varphi(self, xy): 
        return sdf_circles(xy, self.circ_centers, self.circ_radii)
    
    def _grad_varphi(self, xy):
        xy_ = xy.clone().detach().requires_grad_(True)
        phi = self._varphi(xy_)
        return torch.autograd.grad(phi.sum(), xy_, create_graph=True)[0]
    
    def _ray_intersects_circle(self, x):
        """Kiểm tra tia từ x* đến x có cắt hình tròn không"""
        x_star_batch = self.x_star.unsqueeze(0).expand_as(x)
        
        # Vector từ x* đến x
        direction = x - x_star_batch
        direction_norm = torch.norm(direction, dim=1, keepdim=True)
        direction_unit = direction / (direction_norm + 1e-8)
        
        # Vector từ x* đến tâm
        oc = self.center - x_star_batch
        
        # Giải phương trình |x* + t*direction - center|^2 = radius^2
        a = torch.ones(len(x), device=DEVICE)
        b = 2 * torch.sum(oc * direction_unit, dim=1)
        c = torch.sum(oc * oc, dim=1) - self.radius**2
        
        delta = b**2 - 4*a*c
        t1 = (-b - torch.sqrt(delta + 1e-8)) / (2*a + 1e-8)
        t_x = direction_norm.squeeze(-1)
        
        has_intersection = (delta >= 0) & (t1 > 1e-6) & (t_x > t1)
        return has_intersection
    
    def _is_visible(self, x):
        """x có nhìn thấy từ x* không?"""
        if not self.observer_in_obstacle:
            # TH1: x* NGOÀI vật cản
            # Nhìn thấy nếu tia không cắt vật cản
            return ~self._ray_intersects_circle(x)
        else:
            # TH2: x* TRONG vật cản
            # Nhìn thấy nếu x trong vật cản VÀ tia không cắt vật cản
            x_in_obstacle = (self._varphi(x) > 0)
            return x_in_obstacle & ~self._ray_intersects_circle(x)

    def forward(self, x):
        r2 = (x - self.x_star).pow(2).sum(dim=1, keepdim=True)
        
        # PDE solution
        u_pde = r2.squeeze(-1) * self.net(x).squeeze(-1)
        
        # Xác định visible points
        visible = self._is_visible(x)
        
        # Áp dụng định nghĩa mới:
        # u > 0 cho visible points
        # u ≤ 0 cho invisible points
        u = torch.where(
            visible,
            torch.sigmoid(u_pde) + 0.1,  # Visible: u > 0
            torch.tanh(u_pde) - 0.1       # Invisible: u ≤ 0
        )
        
        return u

# -----------------------------------------------------------------------------
# Algorithm 1 Components (đã sửa lỗi)
# -----------------------------------------------------------------------------
def get_importance_samples(model, n_points, mode="G1"):
    x_raw = torch.rand(n_points * 4, 2, device=DEVICE) * 4.0
    with torch.no_grad():
        u = model(x_raw)
        phi = model._varphi(x_raw)
        visible = model._is_visible(x_raw)
        
        if mode == "G1": 
            # G1: Lấy mẫu gần visible/invisible boundary
            weights = torch.sigmoid((u - 0) * 10) + 0.1
            
        elif mode == "G2": 
            # G2: Lấy mẫu gần obstacle boundary
            weights = torch.exp(-torch.abs(u - phi) * 20) + 0.05
            
        else:  # mode == "G3"
            # G3: Lấy mẫu nơi vi phạm định nghĩa
            # Khởi tạo weights với 0
            weights = torch.zeros_like(u)
            
            # 1. Vi phạm obstacle constraint: u < φ
            weights += torch.relu(phi - u) * 50
            
            # 2. Visible points mà u ≤ 0
            visible_mask = visible
            if visible_mask.any():
                weights[visible_mask] += torch.relu(-u[visible_mask]) * 50
            
            # 3. Invisible points mà u > 0
            invisible_mask = ~visible
            if invisible_mask.any():
                weights[invisible_mask] += torch.relu(u[invisible_mask]) * 50
            
            # Thêm một lượng nhỏ để tránh zero weights
            weights += 0.1
    
    idx = torch.multinomial(weights.flatten() + 1e-7, n_points, replacement=True)
    return x_raw[idx]

def compute_losses(model, X1, X2, X3, eps):
    # PDE loss
    X1.requires_grad_(True)
    u1 = model(X1)
    grad_u1 = torch.autograd.grad(u1.sum(), X1, create_graph=True)[0]
    res1 = torch.sum((X1 - model.x_star) * grad_u1, dim=1)
    j1 = torch.mean(res1**2)

    # G2 loss (original)
    X2.requires_grad_(True)
    u2, phi2 = model(X2), model._varphi(X2)
    grad_u2 = torch.autograd.grad(u2.sum(), X2, create_graph=True)[0]
    grad_phi2 = model._grad_varphi(X2)
    dot_phi2 = torch.sum((X2 - model.x_star) * grad_phi2, dim=1)
    mask2 = (torch.abs(u2 - phi2) <= eps) & (dot_phi2 > 1e-5)
    
    if mask2.any():
        res2 = torch.sum((X2 - model.x_star) * (grad_u2 - grad_phi2), dim=1)
        j2 = torch.mean(res2[mask2]**2)
    else:
        j2 = torch.tensor(0.0, device=DEVICE)

    # NEW: Visibility constraint loss
    visible = model._is_visible(X3)
    u3 = model(X3)
    
    # Khởi tạo losses
    loss_visible = torch.tensor(0.0, device=DEVICE)
    loss_invisible = torch.tensor(0.0, device=DEVICE)
    
    # Visible points phải có u > 0
    if visible.any():
        loss_visible = torch.mean(torch.relu(0.01 - u3[visible])**2)
    
    # Invisible points phải có u ≤ 0
    invisible = ~visible
    if invisible.any():
        loss_invisible = torch.mean(torch.relu(u3[invisible] - 0.01)**2)
    
    # Obstacle constraint: u ≥ φ
    phi3 = model._varphi(X3)
    loss_obstacle = torch.mean(torch.relu(phi3 - u3)**2)
    
    j3 = loss_obstacle + 10*loss_visible + 10*loss_invisible
    
    return j1, j2, j3

# -----------------------------------------------------------------------------
# Visualization - Improved version
# -----------------------------------------------------------------------------
def plot_results(model, history):
    grid = 300
    x = torch.linspace(0, 4, grid, device=DEVICE)
    y = torch.linspace(0, 4, grid, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X.ravel(), Y.ravel()], dim=1)

    with torch.no_grad():
        u = model(XY).reshape(grid, grid).cpu().numpy()
        phi = model._varphi(XY).reshape(grid, grid).cpu().numpy()
        visible = model._is_visible(XY).reshape(grid, grid).cpu().numpy()

    # Tạo figure với 3 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
   
    # ===== HÌNH 1: OBSTACLE FUNCTION φ =====
    ax1 = axes[0]
    im = ax1.contourf(X.cpu(), Y.cpu(), phi, levels=50, cmap='viridis')
    ax1.contour(X.cpu(), Y.cpu(), phi, levels=[0], colors='black', linewidths=2)

# THÊM DÒNG NÀY: Vẽ đường u = 0 màu đỏ
    ax1.contour(X.cpu(), Y.cpu(), u.reshape(grid, grid), levels=[0], colors='red', linewidths=2, linestyles='--')

    ax1.scatter(model.x_star[0].cpu(), model.x_star[1].cpu(), 
          color='red' if model.observer_in_obstacle else 'gold', 
          marker='*', s=400, edgecolor='white', linewidth=2)

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('φ(x) - Obstacle function', fontsize=10)

    ax1.set_xlim(0, 4); ax1.set_ylim(0, 4)
    ax1.set_xlabel('$x_1$', fontsize=14); ax1.set_ylabel('$x_2$', fontsize=14)
    ax1.set_title('Obstacle Function φ(x) with visibility boundary u=0 (red)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')

    
    # Vẽ thêm history
    ax2 = axes[1]
    epochs = np.arange(len(history['j1']))
    
    ax2.semilogy(epochs, history['j1'], 'b-', alpha=0.5, label='J1 (PDE)')
    ax2.semilogy(epochs, history['j2'], 'g-', alpha=0.5, label='J2 (Boundary)')
    ax2.semilogy(epochs, history['j3'], 'r-', alpha=0.5, label='J3 (Constraints)')
    ax2.semilogy(epochs, history['total'], 'k-', linewidth=2, label='Total Loss')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training History')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    # Vật cản hình tròn tâm (2,2) bán kính 1
    centers =torch.tensor([
        [2.5, 2.5] ,  # Hình tròn 1: tâm (1.5, 1.5)
        [2.5, 1.5],  # Hình tròn 2: tâm (2.5, 2.5)    
    ], device=DEVICE)
    radii = torch.tensor([
        1,  # Bán kính hình 1
        0.5,  # Bán kính hình 2
    ], device=DEVICE)
    
    model = VisibilityNet2D(centers, radii).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    
    history = {'j1': [], 'j2': [], 'j3': [], 'total': []}

    print(f"Training on {DEVICE}...")
    print(f"Obstacle: Circle at (2, 2) with radius 1")
    print(f"Observer at {X_STAR.cpu().numpy()}")
    print(f"Domain: [0,4] × [0,4]")
    print("-" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        eps = max(EPS_INIT * (0.9995 ** epoch), EPS_FINAL)
        X1, X2, X3 = [get_importance_samples(model, BATCH_SIZE, m) for m in ["G1", "G2", "G3"]]
        
        optimizer.zero_grad()
        j1, j2, j3 = compute_losses(model, X1, X2, X3, eps)
        loss = j1 + j2 + j3  # Điều chỉnh weights
        loss.backward()
        optimizer.step()

        j1_val = j1.item()
        j2_val = j2.item()
        j3_val = j3.item()
        loss_val = loss.item()
        
        history['j1'].append(j1_val)
        history['j2'].append(j2_val)
        history['j3'].append(j3_val)
        history['total'].append(loss_val)
        
        if loss_val < best_loss:
            best_loss = loss_val

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss_val:.2e} | J1: {j1_val:.2e} | J2: {j2_val:.2e} | J3: {j3_val:.2e}")

    print("-" * 50)
    print("Training completed!")
    print(f"Best loss: {best_loss:.2e}")
    print("-" * 50)
    print("Plotting results...")
    plot_results(model, history)

if __name__ == "__main__":
    main()