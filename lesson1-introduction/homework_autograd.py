"""
Домашнее задание 2.1-2.3: Автоматическое дифференцирование в PyTorch.
"""
import torch
import math


# ============================================================
# 2.1 Простые вычисления с градиентами (8 баллов)
# ============================================================
def task_2_1():
    """
    f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
    Аналитические градиенты:
        df/dx = 2x + 2yz
        df/dy = 2y + 2xz
        df/dz = 2z + 2xy
    """
    x = torch.tensor(1.5, requires_grad=True)
    y = torch.tensor(-2.0, requires_grad=True)
    z = torch.tensor(0.5, requires_grad=True)

    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
    print(f"f(x,y,z) = {f.item():.4f}")

    f.backward()

    # Аналитический расчёт
    df_dx_analytic = 2 * x.item() + 2 * y.item() * z.item()
    df_dy_analytic = 2 * y.item() + 2 * x.item() * z.item()
    df_dz_analytic = 2 * z.item() + 2 * x.item() * y.item()

    print(f"df/dx: autograd = {x.grad.item():.4f}, аналитика = {df_dx_analytic:.4f}")
    print(f"df/dy: autograd = {y.grad.item():.4f}, аналитика = {df_dy_analytic:.4f}")
    print(f"df/dz: autograd = {z.grad.item():.4f}, аналитика = {df_dz_analytic:.4f}")

    # Проверка
    assert math.isclose(x.grad.item(), df_dx_analytic, rel_tol=1e-6)
    assert math.isclose(y.grad.item(), df_dy_analytic, rel_tol=1e-6)
    assert math.isclose(z.grad.item(), df_dz_analytic, rel_tol=1e-6)
    print("Градиенты совпадают с аналитическими. ✅")


# ============================================================
# 2.2 Градиент функции потерь (9 баллов)
# ============================================================
def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """MSE = (1/n) * Σ(y_pred - y_true)^2."""
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Формы не совпадают: {y_pred.shape} vs {y_true.shape}")
    n = y_pred.numel()
    return ((y_pred - y_true) ** 2).sum() / n


def task_2_2():
    """
    y_pred = w * x + b, MSE = (1/n) * Σ(y_pred - y_true)^2
    Аналитика:
        dMSE/dw = (2/n) * Σ((w*x+b - y_true) * x)
        dMSE/db = (2/n) * Σ((w*x+b - y_true))
    """
    torch.manual_seed(0)
    x = torch.randn(10, dtype=torch.float64)          # признаки
    y_true = 3.0 * x + 2.0 + 0.1 * torch.randn(10, dtype=torch.float64)

    w = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    y_pred = w * x + b
    loss = mse_loss(y_pred, y_true)
    print(f"MSE loss = {loss.item():.6f}")

    loss.backward()

    # Аналитический расчёт
    n = x.numel()
    residual = (w.item() * x + b.item() - y_true)
    dMSE_dw_analytic = (2.0 / n) * (residual * x).sum().item()
    dMSE_db_analytic = (2.0 / n) * residual.sum().item()

    print(f"dMSE/dw: autograd = {w.grad.item():.6f}, аналитика = {dMSE_dw_analytic:.6f}")
    print(f"dMSE/db: autograd = {b.grad.item():.6f}, аналитика = {dMSE_db_analytic:.6f}")

    assert math.isclose(w.grad.item(), dMSE_dw_analytic, rel_tol=1e-9)
    assert math.isclose(b.grad.item(), dMSE_db_analytic, rel_tol=1e-9)
    print("Градиенты MSE совпадают с аналитическими. ✅")


# ============================================================
# 2.3 Цепное правило (8 баллов)
# ============================================================
def task_2_3():
    """
    f(x) = sin(x^2 + 1)
    Аналитика: df/dx = cos(x^2 + 1) * 2x
    """
    x = torch.tensor(2.0, requires_grad=True)

    f = torch.sin(x ** 2 + 1)
    print(f"f(x) = sin(x^2+1) = {f.item():.6f}")

    # Способ 1: backward()
    f.backward()
    grad_backward = x.grad.item()

    # Способ 2: torch.autograd.grad
    x2 = torch.tensor(2.0, requires_grad=True)
    f2 = torch.sin(x2 ** 2 + 1)
    grad_autograd = torch.autograd.grad(f2, x2)[0].item()

    # Аналитика
    grad_analytic = math.cos(x.item() ** 2 + 1) * 2 * x.item()

    print(f"df/dx (backward):       {grad_backward:.6f}")
    print(f"df/dx (autograd.grad):  {grad_autograd:.6f}")
    print(f"df/dx (аналитика):      {grad_analytic:.6f}")

    assert math.isclose(grad_backward, grad_analytic, rel_tol=1e-6)
    assert math.isclose(grad_autograd, grad_analytic, rel_tol=1e-6)
    print("Все три значения совпадают. ✅")


if __name__ == "__main__":
    print("=" * 60)
    print("2.1 Простые вычисления с градиентами")
    print("=" * 60)
    task_2_1()

    print("\n" + "=" * 60)
    print("2.2 Градиент функции потерь (MSE)")
    print("=" * 60)
    task_2_2()

    print("\n" + "=" * 60)
    print("2.3 Цепное правило")
    print("=" * 60)
    task_2_3()