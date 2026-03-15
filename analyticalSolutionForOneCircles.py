import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import time

# =====================
# CẤU HÌNH - GIỐNG CODE PINN
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN = [0, 4]  # Miền [0,4] × [0,4]
X_STAR = torch.tensor([0.0, 0.0], device=DEVICE)  # Điểm quan sát (0,0)

# Định nghĩa vật cản - GIỐNG HỆT CODE PINN CỦA BẠN
circle_centers = torch.tensor([
    [2.0, 2.0],    # Hình tròn 1: tâm (2.0, 2.0)
    [2.6, 1.2],    # Hình tròn 2: tâm (2.6, 1.2)
], device=DEVICE)
circle_radii = torch.tensor([
    1.0,   # Bán kính hình tròn 1
    0.5,   # Bán kính hình tròn 2
], device=DEVICE)

# Hình vuông
square_center = (0.8, 3.0)  # Tâm (0.8, 3.0)
square_half_side = 0.3       # Nửa cạnh = 0.3 (cạnh = 0.6)

# =====================
# HÀM VẬT CẢN φ(x) - COPY TỪ CODE PINN
# =====================
def sdf_square(xy, center, half_side):
    """
    Hàm khoảng cách có dấu (Signed Distance Function - SDF) cho hình vuông
    """
    cx, cy = center
    x, y = xy[:, 0], xy[:, 1]
    
    dx = torch.abs(x - cx) - half_side
    dy = torch.abs(y - cy) - half_side
    
    inside = (dx <= 0) & (dy <= 0)
    outside = ~inside
    
    phi_outside = -torch.sqrt(dx.clamp(min=0)**2 + dy.clamp(min=0)**2)
    phi_inside = -torch.max(dx, dy)
    
    phi = torch.where(inside, phi_inside, phi_outside)
    return phi

def sdf_circle(xy, center, radius):
    """
    Hàm khoảng cách có dấu cho hình tròn
    """
    diff = xy - center
    dist = diff.norm(dim=1)
    return radius - dist  # φ(x) = r - |x - c|

def combined_obstacle(xy):
    """
    Hàm vật cản tổng hợp - GIỐNG HỆT CODE PINN
    φ(x) = max(φ_circle1, φ_circle2, φ_square)
    """
    # Tính φ cho từng vật cản
    phi_circle1 = sdf_circle(xy, circle_centers[0], circle_radii[0])
    phi_circle2 = sdf_circle(xy, circle_centers[1], circle_radii[1])
    phi_square = sdf_square(xy, square_center, square_half_side)
    
    # Lấy giá trị lớn nhất (vì vật cản là hợp của các miền)
    phi = torch.maximum(torch.maximum(phi_circle1, phi_circle2), phi_square)
    
    return phi

# =====================
# NGHIỆM GIẢI TÍCH - CÔNG THỨC (2.2) TRONG PAPER
# ψ(x) = max_{t∈[0,1]} φ(tx) với x* = (0,0)
# =====================
def analytical_solution(x, y, num_samples=1000):
    """
    Tính nghiệm giải tích ψ(x) = max_{t∈[0,1]} φ(tx, ty)
    
    Với x* = (0,0), công thức đơn giản thành max trên tia từ gốc
    """
    # Tạo các giá trị t từ 0 đến 1
    t = torch.linspace(0, 1, num_samples, device=x.device)
    
    # Tính tọa độ các điểm dọc theo tia: (t*x, t*y)
    x_t = t.unsqueeze(1) * x.unsqueeze(0)  # Shape: (num_samples, batch_size)
    y_t = t.unsqueeze(1) * y.unsqueeze(0)
    
    # Tạo tensor xy cho tất cả các điểm dọc tia
    xy_t = torch.stack([x_t, y_t], dim=2).reshape(-1, 2)  # (num_samples * batch_size, 2)
    
    # Tính φ tại các điểm dọc tia
    phi_values = combined_obstacle(xy_t)
    
    # Reshape lại để lấy max theo t
    phi_values = phi_values.reshape(num_samples, -1)  # (num_samples, batch_size)
    
    # Lấy giá trị lớn nhất theo t
    psi = torch.max(phi_values, dim=0)[0]  # (batch_size)
    
    return psi

# =====================
# TÍNH TOÁN TRÊN LƯỚI
# =====================
def compute_on_grid(resolution=500):
    """Tính nghiệm giải tích trên lưới 2D"""
    print("=" * 60)
    print("TÍNH NGHIỆM GIẢI TÍCH")
    print("=" * 60)
    print(f"Điểm quan sát x* = {X_STAR.cpu().numpy()}")
    print(f"Miền Ω = [{DOMAIN[0]}, {DOMAIN[1]}] × [{DOMAIN[0]}, {DOMAIN[1]}]")
    print("\nVật cản:")
    print(f"  - Hình tròn 1: tâm ({circle_centers[0,0].item():.1f}, {circle_centers[0,1].item():.1f}), bán kính {circle_radii[0].item():.1f}")
    print(f"  - Hình tròn 2: tâm ({circle_centers[1,0].item():.1f}, {circle_centers[1,1].item():.1f}), bán kính {circle_radii[1].item():.1f}")
    print(f"  - Hình vuông: tâm ({square_center[0]:.1f}, {square_center[1]:.1f}), nửa cạnh {square_half_side:.1f} (cạnh {2*square_half_side:.1f})")
    print("-" * 60)
    
    start_time = time.time()
    
    # Tạo lưới điểm
    x = torch.linspace(DOMAIN[0], DOMAIN[1], resolution, device=DEVICE)
    y = torch.linspace(DOMAIN[0], DOMAIN[1], resolution, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Chuyển thành vector để tính
    x_flat = X.reshape(-1)
    y_flat = Y.reshape(-1)
    
    print(f"Đang tính trên lưới {resolution}x{resolution} = {len(x_flat)} điểm...")
    
    # Tính nghiệm giải tích
    psi_flat = analytical_solution(x_flat, y_flat, num_samples=1000)
    
    # Tính φ trên lưới
    xy_flat = torch.stack([x_flat, y_flat], dim=1)
    phi_flat = combined_obstacle(xy_flat)
    
    # Reshape lại
    PSI = psi_flat.reshape(resolution, resolution).cpu().numpy()
    PHI = phi_flat.reshape(resolution, resolution).cpu().numpy()
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    elapsed = time.time() - start_time
    print(f"Hoàn thành trong {elapsed:.2f} giây")
    print("-" * 60)
    
    return X_np, Y_np, PSI, PHI

# =====================
# VẼ NGHIỆM GIẢI TÍCH
# =====================
def plot_analytical_solution(X, Y, PSI, PHI):
    """Vẽ nghiệm giải tích trên 1 hình duy nhất"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 1. Contour fill của nghiệm ψ(x) - màu viridis
    levels = np.linspace(PSI.min(), PSI.max(), 50)
    contour_fill = ax.contourf(X, Y, PSI, levels=levels, cmap='viridis', alpha=0.9)
    
    # 2. Đường u = 0 (ranh giới visible/invisible) - màu đỏ đậm
    if PSI.min() < 0 < PSI.max():
        u0_contour = ax.contour(X, Y, PSI, levels=[0], colors='red', linewidths=3)
        ax.clabel(u0_contour, inline=True, fontsize=12, fmt='u = 0', colors='red')
    
    # 3. Biên vật cản φ = 0 - màu vàng đứt nét
    phi_contour = ax.contour(X, Y, PHI, levels=[0], colors='yellow', linewidths=2.5, 
                             linestyles='--')
    ax.clabel(phi_contour, inline=True, fontsize=10, fmt='φ = 0', colors='yellow')
    
    # 4. Vẽ các hình tròn và hình vuông (tô màu nhạt)
    # Hình tròn 1
    circle1 = Circle((circle_centers[0,0].item(), circle_centers[0,1].item()), 
                     circle_radii[0].item(), 
                     facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=2)
    ax.add_patch(circle1)
    
    # Hình tròn 2
    circle2 = Circle((circle_centers[1,0].item(), circle_centers[1,1].item()), 
                     circle_radii[1].item(), 
                     facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=2)
    ax.add_patch(circle2)
    
    # Hình vuông
    square = Rectangle(
        (square_center[0] - square_half_side, square_center[1] - square_half_side),
        2*square_half_side, 2*square_half_side,
        facecolor='orange', alpha=0.3, edgecolor='orange', linewidth=2
    )
    ax.add_patch(square)
    
    # Đánh dấu tâm các vật cản
    # Tâm hình tròn 1
    ax.scatter(circle_centers[0,0].item(), circle_centers[0,1].item(), 
              color='orange', marker='o', s=80, edgecolor='black', linewidth=1.5, zorder=5)
    ax.annotate('C1', (circle_centers[0,0].item()+0.15, circle_centers[0,1].item()+0.15), 
                fontsize=10, fontweight='bold', color='orange')
    
    # Tâm hình tròn 2
    ax.scatter(circle_centers[1,0].item(), circle_centers[1,1].item(), 
              color='orange', marker='o', s=80, edgecolor='black', linewidth=1.5, zorder=5)
    ax.annotate('C2', (circle_centers[1,0].item()+0.15, circle_centers[1,1].item()+0.15), 
                fontsize=10, fontweight='bold', color='orange')
    
    # Tâm hình vuông
    ax.scatter(square_center[0], square_center[1], 
              color='orange', marker='s', s=80, edgecolor='black', linewidth=1.5, zorder=5)
    ax.annotate('S', (square_center[0]+0.15, square_center[1]+0.15), 
                fontsize=10, fontweight='bold', color='orange')
    
    # 5. Điểm quan sát x* - màu vàng gold
    ax.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
               color='gold', marker='*', s=600, edgecolor='black', linewidth=2,
               label=f'Observer $x^*$ = (0,0)', zorder=10)
    
    # 6. Thêm một vài contour lines có nhãn
    contour_levels = np.linspace(-2, 2, 9)
    contour_lines = ax.contour(X, Y, PSI, levels=contour_levels, 
                                colors='white', linewidths=1, alpha=0.7, linestyles='-')
    ax.clabel(contour_lines, inline=True, fontsize=9, fmt='%.1f', colors='white')
    
    # 7. Format
    ax.set_xlim(DOMAIN[0], DOMAIN[1])
    ax.set_ylim(DOMAIN[0], DOMAIN[1])
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('Nghiệm giải tích u(x,y) - 2 hình tròn + 1 hình vuông', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal')
    
    # 8. Colorbar
    cbar = plt.colorbar(contour_fill, ax=ax, shrink=0.8)
    cbar.set_label('u(x,y) - Nghiệm giải tích', fontsize=12, fontweight='bold')
    
    # 9. Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, label='u = 0 (ranh giới visible/invisible)'),
        Line2D([0], [0], color='yellow', linewidth=2.5, linestyle='--', label='φ = 0 (biên vật cản)'),
        Patch(facecolor='orange', alpha=0.3, edgecolor='orange', label='Vật cản (bên trong)'),
        Line2D([0], [0], marker='*', color='gold', markersize=15, 
               markerfacecolor='gold', markeredgecolor='black', label='Observer x* = (0,0)'),
        Line2D([0], [0], color='white', linewidth=1, label='Contour lines (giá trị u)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('analytical_solution_combined.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

# =====================
# SO SÁNH VỚI PINN (NẾU CÓ)
# =====================
def compare_with_pinn(X, Y, PSI):
    """So sánh nghiệm giải tích với kết quả từ PINN (nếu có file)"""
    
    # Thử load kết quả PINN từ file
    pinn_file = 'combined_obstacles_result.png'
    
    if Path(pinn_file).exists():
        print("\nĐã tìm thấy kết quả PINN, đang so sánh...")
        
        # Tính các thông số thống kê
        print("\n" + "=" * 60)
        print("SO SÁNH VỚI PINN")
        print("=" * 60)
        print(f"  Nghiệm giải tích:")
        print(f"    min(u) = {PSI.min():.4f}")
        print(f"    max(u) = {PSI.max():.4f}")
        print(f"    mean(u) = {PSI.mean():.4f}")
        print(f"    % visible (u ≤ 0): {(PSI <= 0).mean()*100:.2f}%")

# =====================
# MAIN
# =====================
def main():
    # Tính nghiệm giải tích
    X, Y, PSI, PHI = compute_on_grid(resolution=500)
    
    # In thông tin thống kê
    print("\nTHỐNG KÊ NGHIỆM GIẢI TÍCH:")
    print(f"  min(u) = {PSI.min():.4f}")
    print(f"  max(u) = {PSI.max():.4f}")
    print(f"  mean(u) = {PSI.mean():.4f}")
    print(f"  std(u) = {PSI.std():.4f}")
    
    visible_pct = (PSI <= 0).mean() * 100
    print(f"  Vùng visible (u ≤ 0): {visible_pct:.2f}% diện tích")
    print(f"  Vùng invisible (u > 0): {100-visible_pct:.2f}% diện tích")
    
    # Kiểm tra tại điểm quan sát
    u_at_xstar = analytical_solution(
        torch.tensor([0.0], device=DEVICE), 
        torch.tensor([0.0], device=DEVICE)
    )[0].item()
    phi_at_xstar = combined_obstacle(torch.tensor([[0.0, 0.0]], device=DEVICE))[0].item()
    print(f"\nTại điểm quan sát (0,0):")
    print(f"  u(0,0) = {u_at_xstar:.6f}")
    print(f"  φ(0,0) = {phi_at_xstar:.6f}")
    print(f"  u(0,0) = φ(0,0)? {abs(u_at_xstar - phi_at_xstar) < 1e-6}")
    
    # Vẽ nghiệm giải tích
    print("\nĐang vẽ nghiệm giải tích...")
    plot_analytical_solution(X, Y, PSI, PHI)
    
    # So sánh với PINN
    compare_with_pinn(X, Y, PSI)
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH!")
    print("=" * 60)
    print("\nKết quả đã được lưu: analytical_solution_combined.png")

if __name__ == "__main__":
    main()