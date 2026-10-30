"""
Домашнее задание 3.1-3.4: Сравнение производительности CPU vs CUDA.
"""
import time
import torch


# ============================================================
# 3.1 Подготовка данных 
# ============================================================
def create_tensors():
    """Создаёт три больших тензора заданных размеров."""
    torch.manual_seed(42)
    shapes = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]
    tensors = []
    for shape in shapes:
        t = torch.randn(*shape)
        print(f"Создан тензор {shape}, numel = {t.numel()}, "
              f"~{t.numel() * 4 / 1024**2:.1f} МБ")
        tensors.append(t)
    return tensors, shapes


# ============================================================
# 3.2 Функция измерения времени 
# ============================================================
def measure_time_cpu(fn, *args, warmup: int = 2, repeat: int = 5) -> float:
    """
    Измеряет среднее время выполнения функции на CPU (time.time()).
    Возвращает время в миллисекундах.
    """
    # Warmup (прогрев)
    for _ in range(warmup):
        fn(*args)
    start = time.time()
    for _ in range(repeat):
        fn(*args)
    return (time.time() - start) / repeat * 1000.0


def measure_time_gpu(fn, *args, warmup: int = 2, repeat: int = 5) -> float:
    """
    Измеряет среднее время выполнения функции на GPU через torch.cuda.Event.
    Возвращает время в миллисекундах.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        fn(*args)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / repeat  # уже в мс


# ============================================================
# 3.3 Сравнение операций 
# ============================================================
def compare_operations(tensors, shapes):
    """Сравнивает время операций CPU и GPU, выводит таблицу."""
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device_gpu.type == "cuda"
    print(f"\nУстройство GPU: {device_gpu} | доступно: {use_gpu}\n")

    # Операции, которые будем тестировать (для каждой формы тензора)
    operations = {
        "matmul": lambda t: torch.matmul(t, t.transpose(-1, -2)),
        "add":    lambda t: t + t,
        "mul":    lambda t: t * t,
        "transpose": lambda t: t.transpose(-1, -2),
        "sum":    lambda t: t.sum(),
    }

    # Заголовок таблицы
    header = f"{'Форма':<20}{'Операция':<12}{'CPU (мс)':>10}{'GPU (мс)':>10}{'Ускорение':>12}"
    print(header)
    print("-" * len(header))

    results = []
    for tensor, shape in zip(tensors, shapes):
        tensor_cpu = tensor.to("cpu")
        tensor_gpu = tensor.to(device_gpu) if use_gpu else None

        for op_name, op_fn in operations.items():
            # CPU
            cpu_ms = measure_time_cpu(op_fn, tensor_cpu)
            # GPU
            gpu_ms = measure_time_gpu(op_fn, tensor_gpu) if use_gpu else float("nan")
            speedup = (cpu_ms / gpu_ms) if (use_gpu and gpu_ms > 0) else float("nan")

            results.append((shape, op_name, cpu_ms, gpu_ms, speedup))

            gpu_str = f"{gpu_ms:>10.3f}" if use_gpu else f"{'N/A':>10}"
            sp_str = f"{speedup:>11.2f}x" if use_gpu else f"{'N/A':>12}"
            print(f"{str(shape):<20}{op_name:<12}{cpu_ms:>10.3f}{gpu_str}{sp_str}")

        print("-" * len(header))

    return results


# ============================================================
# 3.4 Анализ результатов 
# ============================================================
def analyze_results(results):
    """Печатает текстовый анализ полученных результатов."""
    print("\n" + "=" * 70)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    # Наибольшее ускорение по операциям (среднее по формам)
    from collections import defaultdict
    speedups = defaultdict(list)
    for shape, op, cpu_ms, gpu_ms, speedup in results:
        if speedup == speedup:  # не NaN
            speedups[op].append(speedup)

    if speedups:
        print("\n1) Среднее ускорение по операциям (GPU vs CPU):")
        for op, vals in sorted(speedups.items(),
                               key=lambda kv: -sum(kv[1]) / len(kv[1])):
            avg = sum(vals) / len(vals)
            print(f"   {op:<12}: {avg:6.2f}x  (замеров: {len(vals)})")

    print("""
2) Какие операции получают наибольшее ускорение на GPU?
   — Наибольшее ускорение обычно получают «тяжёлые» операции с высокой
     арифметической интенсивностью: матричное умножение (matmul), а также
     поэлементные операции над большими тензорами (add, mul). GPU имеет
     тысячи ядер и хорошо распараллеливает эти вычисления.

3) Почему некоторые операции могут быть медленнее на GPU?
   — Для мелких тензоров или «лёгких» операций (transpose, sum) накладные
     расходы на запуск CUDA-ядер и синхронизацию могут превышать выигрыш
     от параллелизма. Транспонирование в PyTorch — это, по сути, изменение
     strides (без копирования), поэтому CPU-версия очень быстрая, а GPU
     не даёт преимущества. Также сказывается время на передачу данных
     CPU↔GPU.

4) Как размер матриц влияет на ускорение?
   — Чем больше тензор, тем выше ускорение на GPU (при достаточной
     арифметической интенсивности), т.к. амортизируются накладные расходы
     на запуск ядер и передачу данных. Для маленьких тензоров GPU может
     быть даже медленнее.

5) Что происходит при передаче данных между CPU и GPU?
   — Передача идёт по шине PCIe (или NVLink), это «узкое место»:
     ~10–25 ГБ/с против сотен ГБ/с у видеопамяти. Поэтому в реальных
     задачах важно минимизировать число копирований: данные нужно
     один раз перенести на GPU, выполнить все операции там, а результат
     вернуть на CPU. В нашем бенчмарке время передачи не входит в замер
     (тензоры уже на устройстве), но в реальном коде его нужно учитывать.
""")


if __name__ == "__main__":
    print("=" * 70)
    print("3.1 Подготовка данных")
    print("=" * 70)
    tensors, shapes = create_tensors()

    print("\n" + "=" * 70)
    print("3.3 Сравнение операций CPU vs GPU")
    print("=" * 70)
    results = compare_operations(tensors, shapes)

    print("\n" + "=" * 70)
    print("3.4 Анализ результатов")
    print("=" * 70)
    analyze_results(results)