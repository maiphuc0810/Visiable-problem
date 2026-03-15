import torch
import numpy as np
import matplotlib.pyplot as plt

# =====================
# CẤU HÌNH
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOMAIN = [0, 4]  # Ω = [0,4] × [0,4]
X_STAR = torch.tensor([0.0, 0.0], device=DEVICE)  # Điểm quan sát x*

# Định nghĩa vật cản - 2 hình tròn
CIRCLES = [
    {"center": [2.5, 2.5], "radius": 1.0},  # Hình tròn 1
    {"center": [2.5, 1.5], "radius": 0.5},  # Hình tròn 2
]

# =====================
# HÀM VẬT CẢN φ(x)
# =====================
def phi_circle(x, y, center, radius):
    """Signed Distance Function (SDF) cho hình tròn"""
    return radius - torch.sqrt((x - center[0])**2 + (y - center[1])**2)

def phi(x, y):
    """Hàm vật cản tổng hợp - φ(x) = max(φ₁(x), φ₂(x))"""
    phi_values = []
    for circle in CIRCLES:
        phi_values.append(phi_circle(x, y, circle["center"], circle["radius"]))
    return torch.max(torch.stack(phi_values), dim=0)[0]

# =====================
# NGHIỆM GIẢI TÍCH - CÔNG THỨC (2.2)
# ψ(x) = max_{t∈[0,1]} φ(tx)  (với x* = 0)
# =====================
def analytical_solution(x, y):
    """Tính nghiệm giải tích ψ(x) = max_{t∈[0,1]} φ(tx, ty)"""
    t = torch.linspace(0, 1, 1000, device=x.device)
    x_t = t.unsqueeze(1) * x.unsqueeze(0)
    y_t = t.unsqueeze(1) * y.unsqueeze(0)
    phi_values = phi(x_t, y_t)
    return torch.max(phi_values, dim=0)[0]

# =====================
# TÍNH TOÁN TRÊN LƯỚI
# =====================
def compute_solution(resolution=500):
    """Tính nghiệm trên lưới 2D"""
    x = torch.linspace(DOMAIN[0], DOMAIN[1], resolution, device=DEVICE)
    y = torch.linspace(DOMAIN[0], DOMAIN[1], resolution, device=DEVICE)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    print("Đang tính nghiệm giải tích...")
    X_flat, Y_flat = X.reshape(-1), Y.reshape(-1)
    PSI_flat = analytical_solution(X_flat, Y_flat)
    PHI_flat = phi(X_flat, Y_flat)
    
    PSI = PSI_flat.reshape(resolution, resolution).cpu().numpy()
    PHI = PHI_flat.reshape(resolution, resolution).cpu().numpy()
    
    return X.cpu().numpy(), Y.cpu().numpy(), PSI, PHI

# =====================
# VẼ 1 ẢNH DUY NHẤT
# =====================
def plot_single_figure(X, Y, PSI, PHI):
    """Vẽ nghiệm giải tích trên một ảnh duy nhất"""
    
    plt.figure(figsize=(12, 10))
    
    # 1. Contour fill của nghiệm ψ(x) - màu viridis
    levels = np.linspace(PSI.min(), PSI.max(), 50)
    contour_fill = plt.contourf(X, Y, PSI, levels=levels, cmap='viridis', alpha=0.9)
    
    # 2. Đường u = 0 (ranh giới visible/invisible) - màu đỏ đậm
    if PSI.min() < 0 < PSI.max():
        u0_contour = plt.contour(X, Y, PSI, levels=[0], colors='red', linewidths=3)
        # Thêm nhãn "u = 0"
        plt.clabel(u0_contour, inline=True, fontsize=12, fmt='u = 0', colors='red')
    
    # 3. Biên vật cản φ = 0 - màu vàng đứt nét
    plt.contour(X, Y, PHI, levels=[0], colors='yellow', linewidths=2.5, 
                linestyles='--', label='φ = 0')
    
    # 4. Vẽ các hình tròn vật cản (tô màu nhạt để thấy rõ)
    for i, circle in enumerate(CIRCLES):
        center = circle["center"]
        radius = circle["radius"]
        
        # Vẽ hình tròn
        circle_patch = plt.Circle(center, radius, color='orange', alpha=0.3, 
                                  edgecolor='orange', linewidth=2)
        plt.gca().add_patch(circle_patch)
        
        # Đánh dấu tâm
        plt.scatter(center[0], center[1], color='orange', marker='o', s=80, 
                   edgecolor='black', linewidth=1.5, zorder=5)
        plt.annotate(f'C{i+1}', (center[0]+0.15, center[1]+0.15), 
                    fontsize=11, fontweight='bold', color='orange')
    
    # 5. Điểm quan sát x* - màu vàng gold
    plt.scatter(X_STAR[0].cpu(), X_STAR[1].cpu(), 
               color='gold', marker='*', s=600, edgecolor='black', linewidth=2,
               label=f'Observer $x^*$ = (0,0)', zorder=10)
    
    # 6. Thêm một vài contour lines có nhãn để thấy giá trị
    contour_levels = np.linspace(-2, 2, 9)  # Các mức: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    contour_lines = plt.contour(X, Y, PSI, levels=contour_levels, 
                                colors='white', linewidths=1, alpha=0.7, linestyles='-')
    plt.clabel(contour_lines, inline=True, fontsize=9, fmt='%.1f', colors='white')
    
    # 7. Format
    plt.xlim(DOMAIN[0], DOMAIN[1])
    plt.ylim(DOMAIN[0], DOMAIN[1])
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('Nghiệm giải tích u(x,y) của Visibility Problem\n'
              'Vùng tối: u > 0 (invisible), Vùng sáng: u < 0 (visible)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.gca().set_aspect('equal')
    
    # 8. Colorbar
    cbar = plt.colorbar(contour_fill, shrink=0.8)
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
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# =====================
# MAIN
# =====================
def main():
    print("=" * 60)
    print("NGHIỆM GIẢI TÍCH CỦA VISIBILITY PROBLEM")
    print("Công thức: u(x) = max_{t∈[0,1]} φ(tx) với x* = (0,0)")
    print("=" * 60)
    
    # Tính nghiệm
    X, Y, PSI, PHI = compute_solution(resolution=500)
    
    # Thông tin
    print(f"\nTHÔNG TIN:")
    print(f"  min(u) = {PSI.min():.4f}")
    print(f"  max(u) = {PSI.max():.4f}")
    print(f"  % vùng visible (u ≤ 0): {(PSI <= 0).mean()*100:.2f}%")
    print(f"  % vùng invisible (u > 0): {(PSI > 0).mean()*100:.2f}%")
    
    # Vẽ 1 ảnh duy nhất
    print("\nĐang vẽ kết quả...")
    plot_single_figure(X, Y, PSI, PHI)
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)

if __name__ == "__main__":
    main()