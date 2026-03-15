import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# =====================
# CẤU HÌNH & SIÊU THAM SỐ
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN_LOW, DOMAIN_HIGH = 0.0, 4.0
X_STAR = torch.tensor([2.2, 2.2], dtype=torch.float32, device=DEVICE)

HIDDEN_SIZE = 128
NUM_LAYERS = 4
ACT_FUNCTION = nn.Tanh()

LR_INIT = 1e-3
EPOCHS = 10000
BATCH_SIZE = 2048
EPS_INIT, EPS_FINAL = 0.1, 1e-4

# =====================
# ĐỊNH NGHĨA HÀM VẬT CẢN - 2 HÌNH TRÒN
# ===================== 
def sdf_circles(xy, centers, radii):
    """φ(x) = max(r_i - ||x - c_i||) - dương TRONG vật cản"""
    if centers.numel() == 0:
        return torch.full((xy.shape[0],), -1.0, device=xy.device)
    diff = xy.unsqueeze(1) - centers.unsqueeze(0)
    dist = diff.norm(dim=2)
    return (radii.unsqueeze(0) - dist).max(dim=1).values

# =====================
# MẠNG NƠ-RON
# =====================
class VisibilityNet(nn.Module):
    def __init__(self, circ_centers, circ_radii):
        super().__init__()
        self.register_buffer("circ_centers", circ_centers)
        self.register_buffer("circ_radii", circ_radii)
        self.register_buffer("x_star", X_STAR)
        
        with torch.no_grad():
            phi_xs = sdf_circles(X_STAR.unsqueeze(0), circ_centers, circ_radii)[0]
            self.register_buffer("phi_xs", phi_xs)

        # Mạng neural
        layers = []
        in_dim = 2
        for _ in range(NUM_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_SIZE))
            layers.append(ACT_FUNCTION)
            in_dim = HIDDEN_SIZE
        layers.append(nn.Linear(HIDDEN_SIZE, 1))
        self.net = nn.Sequential(*layers)

    def _varphi(self, xy):
        return sdf_circles(xy, self.circ_centers, self.circ_radii)
    
    def forward(self, x):
        """u(x) được tính trực tiếp từ mạng neural"""
        return self.net(x).squeeze(-1)

# =====================
# LẤY MẪU TẬP TRUNG
# =====================
def get_stratified_samples(model, n_points):
    """Lấy mẫu phân tầng - tập trung vào vùng quan trọng"""
    n_inside = n_points // 2
    n_boundary = n_points // 4
    n_outside = n_points - n_inside - n_boundary
    
    all_samples = []
    all_weights = []
    
    # 1. Mẫu trong vật cản
    if n_inside > 0:
        inside_samples = []
        while len(inside_samples) < n_inside:
            cand = torch.rand(n_inside * 2, 2, device=DEVICE) * 4.0
            phi_cand = model._varphi(cand)
            inside_cand = cand[phi_cand > 0.01]
            if len(inside_cand) > 0:
                inside_samples.append(inside_cand)
        
        if inside_samples:
            inside_samples = torch.cat(inside_samples)[:n_inside]
            all_samples.append(inside_samples)
            all_weights.append(torch.ones(len(inside_samples)) * 2.0)
    
    # 2. Mẫu gần biên
    if n_boundary > 0:
        boundary_samples = []
        for _ in range(2):  # Lấy quanh cả 2 hình tròn
            theta = torch.rand(n_boundary, device=DEVICE) * 2 * np.pi
            # Hình tròn 1
            x1 = 2.5 + 1.0 * torch.cos(theta)
            y1 = 2.5 + 1.0 * torch.sin(theta)
            b1 = torch.stack([x1, y1], dim=1) + torch.randn(n_boundary, 2, device=DEVICE) * 0.05
            
            # Hình tròn 2
            x2 = 2.5 + 0.5 * torch.cos(theta)
            y2 = 1.5 + 0.5 * torch.sin(theta)
            b2 = torch.stack([x2, y2], dim=1) + torch.randn(n_boundary, 2, device=DEVICE) * 0.05
            
            boundary_samples.extend([b1, b2])
        
        if boundary_samples:
            boundary_samples = torch.cat(boundary_samples)
            # Lọc trong domain
            mask = (boundary_samples[:, 0] >= 0) & (boundary_samples[:, 0] <= 4) & \
                   (boundary_samples[:, 1] >= 0) & (boundary_samples[:, 1] <= 4)
            boundary_samples = boundary_samples[mask][:n_boundary]
            
            if len(boundary_samples) > 0:
                all_samples.append(boundary_samples)
                all_weights.append(torch.ones(len(boundary_samples)) * 5.0)
    
    # 3. Mẫu ngoài vật cản
    if n_outside > 0:
        outside_samples = []
        while len(outside_samples) < n_outside:
            cand = torch.rand(n_outside * 2, 2, device=DEVICE) * 4.0
            phi_cand = model._varphi(cand)
            outside_cand = cand[phi_cand < -0.05]
            if len(outside_cand) > 0:
                outside_samples.append(outside_cand)
        
        if outside_samples:
            outside_samples = torch.cat(outside_samples)[:n_outside]
            all_samples.append(outside_samples)
            all_weights.append(torch.ones(len(outside_samples)) * 1.0)
    
    # Kết hợp tất cả
    if not all_samples:
        return torch.rand(n_points, 2, device=DEVICE), torch.ones(n_points, device=DEVICE)
    
    X = torch.cat(all_samples)
    weights = torch.cat(all_weights)
    
    # Nếu thiếu, lấy thêm mẫu ngẫu nhiên
    if len(X) < n_points:
        extra = torch.rand(n_points - len(X), 2, device=DEVICE) * 4.0
        X = torch.cat([X, extra])
        weights = torch.cat([weights, torch.ones(len(extra), device=DEVICE)])
    
    return X[:n_points], weights[:n_points]

# =====================
# TÍNH LOSS - CÁCH 1: DÙNG create_graph=False
# =====================
def compute_loss_v1(model, X, weights, eps):
    """Tính loss với gradient nhưng không tạo graph cho đạo hàm bậc 2"""
    u = model(X)
    phi = model._varphi(X)
    
    # Phân loại điểm
    inside_mask = phi > eps
    boundary_mask = torch.abs(phi) <= eps
    outside_mask = phi < -eps
    
    losses = {}
    total_loss = 0.0
    
    # J₁: Trong vật cản - tính gần đúng (x-x*)·∇u = 0 bằng sai phân hữu hạn
    if inside_mask.any():
        X_in = X[inside_mask]
        u_in = u[inside_mask]
        w_in = weights[inside_mask]
        
        # Tính gradient bằng sai phân hữu hạn (không cần autograd)
        h = 1e-4
        grad_u_x = torch.zeros_like(u_in)
        grad_u_y = torch.zeros_like(u_in)
        
        for i in range(len(X_in)):
            x = X_in[i:i+1].clone().detach().requires_grad_(False)
            
            # Sai phân hữu hạn theo x
            x_plus_h = x + torch.tensor([[h, 0.0]], device=DEVICE)
            x_minus_h = x - torch.tensor([[h, 0.0]], device=DEVICE)
            
            with torch.no_grad():
                u_plus = model(x_plus_h)
                u_minus = model(x_minus_h)
            
            grad_u_x[i] = (u_plus - u_minus) / (2 * h)
            
            # Sai phân hữu hạn theo y
            y_plus_h = x + torch.tensor([[0.0, h]], device=DEVICE)
            y_minus_h = x - torch.tensor([[0.0, h]], device=DEVICE)
            
            with torch.no_grad():
                u_plus = model(y_plus_h)
                u_minus = model(y_minus_h)
            
            grad_u_y[i] = (u_plus - u_minus) / (2 * h)
        
        # (x-x*)·∇u
        dot_u = (X_in[:, 0] - model.x_star[0]) * grad_u_x + \
                (X_in[:, 1] - model.x_star[1]) * grad_u_y
        
        j1 = torch.sum(w_in * dot_u**2) / (w_in.sum() + 1e-8)
        losses['j1'] = j1
        total_loss = total_loss + j1
    else:
        losses['j1'] = torch.tensor(0.0, device=DEVICE)
    
    # J₂: Trên biên - u = 0
    if boundary_mask.any():
        u_b = u[boundary_mask]
        w_b = weights[boundary_mask]
        j2 = torch.sum(w_b * u_b**2) / (w_b.sum() + 1e-8)
        losses['j2'] = j2
        total_loss = total_loss + 10.0 * j2
    else:
        losses['j2'] = torch.tensor(0.0, device=DEVICE)
    
    # J₃: Ngoài vật cản - u = 0
    if outside_mask.any():
        u_out = u[outside_mask]
        w_out = weights[outside_mask]
        j3 = torch.sum(w_out * u_out**2) / (w_out.sum() + 1e-8)
        losses['j3'] = j3
        total_loss = total_loss + 5.0 * j3
    else:
        losses['j3'] = torch.tensor(0.0, device=DEVICE)
    
    # Boundary condition tại x*
    with torch.no_grad():
        u_xs = model(model.x_star.unsqueeze(0))
    j_xs = (u_xs - model.phi_xs)**2
    losses['j_xs'] = j_xs
    total_loss = total_loss + 100.0 * j_xs
    
    losses['total'] = total_loss
    
    return losses

# =====================
# TÍNH LOSS - CÁCH 2: DÙNG create_graph=True nhưng xử lý đúng
# =====================
def compute_loss_v2(model, X, weights, eps):
    """Tính loss với autograd - yêu cầu X.requires_grad=True"""
    X.requires_grad_(True)
    
    u = model(X)
    phi = model._varphi(X)
    
    # Phân loại điểm
    inside_mask = phi > eps
    boundary_mask = torch.abs(phi) <= eps
    outside_mask = phi < -eps
    
    losses = {}
    total_loss = 0.0
    
    # J₁: Trong vật cản - (x-x*)·∇u = 0
    if inside_mask.any():
        # Tạo mask và lấy indices
        inside_indices = torch.where(inside_mask)[0]
        
        # Tính gradient cho từng điểm một (để tránh lỗi)
        j1_sum = 0.0
        w_sum = 0.0
        
        for idx in inside_indices:
            x_i = X[idx:idx+1]
            u_i = u[idx:idx+1]
            w_i = weights[idx:idx+1]
            
            # Tính gradient
            grad_u = torch.autograd.grad(u_i.sum(), x_i, create_graph=True, allow_unused=True)[0]
            
            if grad_u is not None:
                dot_u = ((x_i[0, 0] - model.x_star[0]) * grad_u[0, 0] + 
                        (x_i[0, 1] - model.x_star[1]) * grad_u[0, 1])
                j1_sum += w_i * dot_u**2
                w_sum += w_i
        
        if w_sum > 0:
            j1 = j1_sum / w_sum
        else:
            j1 = torch.tensor(0.0, device=DEVICE)
            
        losses['j1'] = j1
        total_loss = total_loss + j1
    else:
        losses['j1'] = torch.tensor(0.0, device=DEVICE)
    
    # J₂: Trên biên - u = 0
    if boundary_mask.any():
        u_b = u[boundary_mask]
        w_b = weights[boundary_mask]
        j2 = torch.sum(w_b * u_b**2) / (w_b.sum() + 1e-8)
        losses['j2'] = j2
        total_loss = total_loss + 10.0 * j2
    else:
        losses['j2'] = torch.tensor(0.0, device=DEVICE)
    
    # J₃: Ngoài vật cản - u = 0
    if outside_mask.any():
        u_out = u[outside_mask]
        w_out = weights[outside_mask]
        j3 = torch.sum(w_out * u_out**2) / (w_out.sum() + 1e-8)
        losses['j3'] = j3
        total_loss = total_loss + 5.0 * j3
    else:
        losses['j3'] = torch.tensor(0.0, device=DEVICE)
    
    # Boundary condition tại x*
    u_xs = model(model.x_star.unsqueeze(0))
    j_xs = (u_xs - model.phi_xs)**2
    losses['j_xs'] = j_xs
    total_loss = total_loss + 100.0 * j_xs
    
    losses['total'] = total_loss
    
    return losses

# =====================
# VẼ KẾT QUẢ
# =====================
def plot_results(model, history):
    grid = 300
    x = torch.linspace(0, 4, grid, device=DEVICE)
    y = torch.linspace(0, 4, grid, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    XY = torch.stack([X.ravel(), Y.ravel()], dim=1)

    with torch.no_grad():
        u = model(XY).reshape(grid, grid).cpu().numpy()
        phi = model._varphi(XY).reshape(grid, grid).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Hình 1: Nghiệm u
    ax = axes[0]
    im = ax.contourf(X.cpu(), Y.cpu(), u, levels=50, cmap='viridis')
    ax.contour(X.cpu(), Y.cpu(), phi, levels=[0], colors='red', linewidths=2)
    ax.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
              color='yellow', marker='*', s=200, edgecolor='black')
    ax.set_title(f'Solution u(x) - φ(x*)={model.phi_xs:.3f}')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # Hình 2: So sánh với điều kiện u=0 ngoài vật cản
    ax = axes[1]
    # Tạo mask: 1 nếu |u|>0.1 và outside, 0 otherwise
    outside = phi < -0.01
    violation = np.zeros_like(u)
    if outside.any():
        violation[outside.cpu().numpy()] = np.abs(u[outside.cpu().numpy()])
    
    im = ax.contourf(X.cpu(), Y.cpu(), violation, levels=50, cmap='hot')
    ax.contour(X.cpu(), Y.cpu(), phi, levels=[0], colors='blue', linewidths=2)
    ax.set_title('|u| outside obstacle (should be 0)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # Hình 3: Loss history
    ax = axes[2]
    epochs = np.arange(len(history['total']))
    
    ax.semilogy(epochs, history.get('j1', [1e-10]*len(epochs)), label='J₁ (inside)')
    ax.semilogy(epochs, history.get('j2', [1e-10]*len(epochs)), label='J₂ (boundary)')
    ax.semilogy(epochs, history.get('j3', [1e-10]*len(epochs)), label='J₃ (outside)')
    ax.semilogy(epochs, history.get('j_xs', [1e-10]*len(epochs)), label='J_xs (BC)')
    ax.semilogy(epochs, history['total'], 'k--', label='Total', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# =====================
# HÀM CHÍNH
# =====================
def main():
    # Định nghĩa 2 hình tròn
    centers = torch.tensor([
        [2.5, 2.5],
        [2.5, 1.5],
    ], device=DEVICE)
    radii = torch.tensor([
        1.0,
        0.5,
    ], device=DEVICE)
    
    print(f"Observer tại {X_STAR.cpu().numpy()}")
    
    # Kiểm tra x* có trong vật cản không
    with torch.no_grad():
        phi_xs = sdf_circles(X_STAR.unsqueeze(0), centers, radii)[0].item()
        print(f"φ(x*) = {phi_xs:.4f} {'✅' if phi_xs > 0 else '❌'}")
    
    model = VisibilityNet(centers, radii).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
    
    # Chọn cách tính loss
    use_v2 = False  # Đổi thành True nếu muốn dùng autograd
    
    if use_v2:
        print("Sử dụng compute_loss_v2 (autograd từng điểm)")
        compute_loss = compute_loss_v2
    else:
        print("Sử dụng compute_loss_v1 (sai phân hữu hạn)")
        compute_loss = compute_loss_v1
    
    history = {'j1': [], 'j2': [], 'j3': [], 'j_xs': [], 'total': []}
    best_loss = float('inf')
    
    print("\nBắt đầu training...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        eps = max(EPS_INIT * (0.999 ** epoch), EPS_FINAL)
        
        # Lấy mẫu
        X, weights = get_stratified_samples(model, BATCH_SIZE)
        
        # Tính loss
        optimizer.zero_grad()
        losses = compute_loss(model, X, weights, eps)
        
        # Backprop
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(losses['total'])
        
        # Lưu history
        for key in losses:
            if key in history:
                history[key].append(losses[key].item())
        
        if losses['total'].item() < best_loss:
            best_loss = losses['total'].item()
        
        if epoch % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:5d} | Loss: {losses['total'].item():.2e} | "
                  f"J1: {losses.get('j1', torch.tensor(0.0)).item():.2e} | "
                  f"J2: {losses.get('j2', torch.tensor(0.0)).item():.2e} | "
                  f"J3: {losses.get('j3', torch.tensor(0.0)).item():.2e} | "
                  f"Time: {elapsed:.1f}s")
    
    print(f"\nBest loss: {best_loss:.2e}")
    plot_results(model, history)

if __name__ == "__main__":
    main()