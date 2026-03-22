import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class VisibilityWithPointInsideObstacle:
    def __init__(self, x_star):
        self.x_star = torch.tensor(x_star, dtype=torch.float32)
        
        # Vật cản 1: hình vuông trái (x1 < 3)
        self.obs1_center = torch.tensor([2.0, 3.0])
        self.obs1_half = 1.0  # cạnh 2
        
        # Vật cản 2: hình vuông phải (x1 ≥ 3) - chứa x*
        self.obs2_center = torch.tensor([3.0, 3.0])
        self.obs2_half = 0.5  # cạnh 1
        
        # Kiểm tra x* có nằm trong obs2 không
        phi_at_star = self.sdf_square(self.x_star, self.obs2_center, self.obs2_half)
        assert phi_at_star < 0, "x* phải nằm trong vật cản thứ 2"
        print(f"x* = ({x_star[0]}, {x_star[1]}) nằm TRONG hình vuông thứ 2 (φ={phi_at_star:.4f})")
        
    def sdf_square(self, x, center, half):
        """SDF cho hình vuông"""
        dx = torch.abs(x[..., 0] - center[0]) - half
        dy = torch.abs(x[..., 1] - center[1]) - half
        d_out = torch.sqrt(torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2)
        d_in = torch.min(torch.max(dx, dy), torch.tensor(0.0))
        return d_out + d_in
    
    def sdf_combined(self, x):
        """φ(x) = min(φ₁(x), φ₂(x))"""
        phi1 = self.sdf_square(x, self.obs1_center, self.obs1_half)
        phi2 = self.sdf_square(x, self.obs2_center, self.obs2_half)
        return torch.min(phi1, phi2)
    
    def compute_psi(self, x):
        """
        ψ(x) = max_{t∈[0,1]} φ(x* + t(x-x*))
        """
        # Tạo các điểm trên tia từ x* đến x
        t = torch.linspace(0, 1, 1000, device=x.device)
        x_t = self.x_star.unsqueeze(0) + t.unsqueeze(1) * (x - self.x_star)
        
        # Tính φ tại các điểm trên tia
        phi_t = self.sdf_combined(x_t)
        
        # Lấy giá trị lớn nhất dọc theo tia
        psi_val = torch.max(phi_t)
        
        return psi_val
    
    def is_visible(self, x):
        """Kiểm tra điểm x có nhìn thấy từ x* không"""
        psi_val = self.compute_psi(x)
        return psi_val <= 0
    
    def visualize(self, resolution=200):
        """Vẽ bản đồ tầm nhìn"""
        x = torch.linspace(0, 4, resolution)
        y = torch.linspace(0, 4, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.ravel(), Y.ravel()], dim=1)
        
        # Tính ψ cho từng điểm
        print("Đang tính toán ψ(x)...")
        psi_values = []
        for i, p in enumerate(points):
            psi_values.append(self.compute_psi(p).item())
            if (i + 1) % 5000 == 0:
                print(f"  Đã xử lý {i+1}/{len(points)} điểm")
        
        psi_grid = torch.tensor(psi_values).reshape(resolution, resolution).numpy()
        
        # Vẽ kết quả
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Hình 1: ψ(x)
        ax = axes[0]
        im = ax.contourf(X.numpy(), Y.numpy(), psi_grid, levels=50, cmap='RdYlBu_r')
        ax.contour(X.numpy(), Y.numpy(), psi_grid, levels=[0], colors='black', linewidths=2)
        
        # Vẽ hai hình vuông - tạo mới cho axes này
        rect1_1 = Rectangle((1, 2), 2, 2, fill=False, edgecolor='red', linewidth=2)
        rect1_2 = Rectangle((2.5, 2.5), 1, 1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect1_1)
        ax.add_patch(rect1_2)
        
        # Điểm quan sát
        ax.scatter(self.x_star[0], self.x_star[1], color='gold', marker='*', s=300, 
                  edgecolor='black', label='Observer x*', zorder=10)
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Visibility Map ψ(x)')
        ax.legend()
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='ψ(x)')
        
        # Hình 2: Vùng nhìn thấy
        ax2 = axes[1]
        visible = (psi_grid <= 0)
        
        # Tạo colormap tùy chỉnh
        cmap = plt.matplotlib.colors.ListedColormap(['darkred', 'lightgreen'])
        im2 = ax2.contourf(X.numpy(), Y.numpy(), visible, levels=[-0.5, 0.5, 1.5], 
                          colors=['darkred', 'lightgreen'], alpha=0.7)
        
        # Vẽ hai hình vuông - tạo mới cho axes này
        rect2_1 = Rectangle((1, 2), 2, 2, fill=False, edgecolor='red', linewidth=2)
        rect2_2 = Rectangle((2.5, 2.5), 1, 1, fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect2_1)
        ax2.add_patch(rect2_2)
        
        ax2.scatter(self.x_star[0], self.x_star[1], color='gold', marker='*', s=300,
                   edgecolor='black', zorder=10)
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_title('Visible Region (ψ(x) ≤ 0)')
        ax2.set_aspect('equal')
        
        # Thêm chú thích
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightgreen', alpha=0.7, label='Visible'),
                          Patch(facecolor='darkred', alpha=0.7, label='Not Visible')]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Thống kê
        total_points = resolution * resolution
        visible_points = np.sum(visible)
        visible_percent = 100 * visible_points / total_points
        
        print("\n" + "="*60)
        print("THỐNG KÊ")
        print("="*60)
        print(f"Tổng số điểm: {total_points}")
        print(f"Điểm nhìn thấy: {visible_points}")
        print(f"Tỷ lệ nhìn thấy: {visible_percent:.2f}%")
        print(f"Min ψ(x): {psi_grid.min():.4f}")
        print(f"Max ψ(x): {psi_grid.max():.4f}")
        
        return psi_grid

# Sử dụng
if __name__ == "__main__":
    x_star = (3.2, 3.0)  # Nằm trong hình vuông thứ 2
    vis = VisibilityWithPointInsideObstacle(x_star)

    # Kiểm tra một số điểm
    test_points = [
        (3.2, 3.0, "Observer"),
        (3.8, 3.0, "Bên phải (ra ngoài)"),
        (3.0, 3.0, "Tâm hình vuông 2"),
        (2.0, 3.0, "Tâm hình vuông 1"),
        (1.0, 3.0, "Bên trái"),
        (3.2, 3.8, "Phía trên"),
        (3.2, 2.2, "Phía dưới"),
        (3.5, 3.5, "Góc trên phải"),
        (2.5, 3.5, "Góc trên giữa"),
    ]

    print("\n" + "="*60)
    print("KIỂM TRA TẦM NHÌN TỪ ĐIỂM QUAN SÁT TRONG VẬT CẢN")
    print("="*60)

    for x, y, name in test_points:
        point = torch.tensor([x, y], dtype=torch.float32)
        psi = vis.compute_psi(point)
        visible = vis.is_visible(point)
        print(f"{name:<20}: ({x}, {y}) → ψ = {psi:.4f} → {'NHÌN THẤY' if visible else 'KHÔNG NHÌN THẤY'}")

    # Vẽ bản đồ
    print("\nĐang vẽ bản đồ tầm nhìn...")
    vis.visualize(resolution=150)  # Giảm resolution để chạy nhanh hơn