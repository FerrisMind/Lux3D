<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>Rust workspace для clean-room 3D-инференса, инспекции контрактов и каноникализации весов.</b><br>
  CUDA-first runtime для Pi3, Pi3X, TripoSR, VGGT, TripoSG и TRELLIS.2.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D / candle-3d</h1>

## Демо

https://github.com/user-attachments/assets/0a9f766f-68d3-482a-9c94-c55d0c355808

## Что это?

Lux3D / candle-3d — Rust 2024 workspace, который запускает модели 3D-реконструкции
и генерации на [Candle](https://github.com/FerrisMind/candle) с CUDA-first
runtime и fallback-бэкендами Vulkan/WGPU. Предоставляет:

- единый CLI для инспекции контрактов, нормализации весов, GGUF-квантизации,
  инференса и конвертации O-Voxel;
- воспроизводимые canonical-пакеты весов с manifest и checksums;
- clean-room порты референсного препроцессинга, сэмплинга и экспорта мешей;
- OpenAI-style асинхронный HTTP API для генерации point cloud и мешей.

Каждое семейство моделей сохраняет лицензию upstream. Перед
перераспространением весов проверьте условия в [docs/models.md](docs/models.md).

## Поддерживаемые модели

| Семейство | Задача | Выход | Upstream |
|-----------|--------|-------|----------|
| [`pi3`](docs/models.md#pi3-and-pi3x) | Multi-view point cloud | PLY / XYZ | [yyfz/Pi3](https://github.com/yyfz/Pi3) |
| [`pi3x`](docs/models.md#pi3-and-pi3x) | Multi-view point cloud + VO | PLY / XYZ | [yyfz/Pi3](https://github.com/yyfz/Pi3) |
| [`triposr`](docs/models.md#triposr) | Single-image mesh | GLB / OBJ / PLY / STL | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) |
| [`vggt`](docs/vggt.md) | Multi-view geometry + COLMAP | PLY / XYZ / COLMAP / GLB | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| [`triposg`](docs/models.md#triposg) | Image-to-mesh, scribble, surface encode | GLB / OBJ / PLY / STL | [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) |
| [`trellis2`](docs/trellis2.md) | Image-to-3D, текстурирование, O-Voxel | GLB / OBJ / PLY / STL | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |

Canonical-пакеты публикуются под `oxide-lab/*` на Hugging Face и автоматически
скачиваются в пользовательский кэш. `oxide-lab/TRELLIS.2-4B` зарезервирован, но
ещё не опубликован, поэтому для TRELLIS.2 нужен явный `--model-path`.

## Быстрый старт

### Требования

- Rust 1.85+ с поддержкой `edition = "2024"`
- Python 3.x для скриптов нормализации и оценки
- NVIDIA GPU для проверенного CUDA-инференса; Vulkan и WGPU поддерживаются, но
  экспериментальны (см. [experimental backends](docs/inference.md#experimental-backends-vulkan-and-wgpu))

### Сборка

```powershell
cargo build --release -p lux3d-cli --features cuda
```

Сборка по умолчанию — CPU-only. GPU-бэкенды включаются Cargo-фичами: `cuda`,
`vulkan`, `wgpu`, `metal`.

### Инспекция и подготовка весов

```powershell
# Показать контракт, ожидаемый семейством
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposg

# Нормализовать raw checkpoint в canonical-пакет
cargo run -p lux3d-cli -- weights normalize `
  --repo-root <runtime-root> `
  --raw-model-dir <raw-triposg-dir> `
  --output-dir <canonical-triposg-dir> triposg
```

Подробно о весах: [canonical packages and quantization](docs/weights.md).

### Генерация

```powershell
# Изображение -> меш (canonical-пакет TripoSG скачается автоматически)
cargo run -p lux3d-cli --features cuda -- run triposg `
  --source .\input.png --output .\out.glb

# Изображение -> текстурированный GLB (явный пакет TRELLIS.2; RMBG-2.0 отдельно)
cargo run -p lux3d-cli --features cuda -- run trellis2 `
  --model-path <canonical-trellis2-dir> `
  --rmbg-model-path <canonical-rmbg2-dir> `
  --source .\input.png --output .\out.glb
```

Все команды семейств, флаги и правила форматов вывода:
[docs/inference.md](docs/inference.md).

### HTTP-сервер

```powershell
cargo run -p lux3d-server --features cuda -- --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
```

Эндпоинты, поля запросов и SSRF-политика: [docs/server.md](docs/server.md).

## Документация

| Документ | Содержание |
|----------|------------|
| [docs/models.md](docs/models.md) | Семейства моделей, пайплайны, upstream, лицензии |
| [docs/inference.md](docs/inference.md) | Команды `run` для всех семейств, форматы вывода, экспериментальные бэкенды, O-Voxel |
| [docs/vggt.md](docs/vggt.md) | VGGT: препроцессинг, COLMAP-экспорт, трекинг, GLB-сцена, дампы тензоров |
| [docs/trellis2.md](docs/trellis2.md) | TRELLIS.2: варианты пайплайна, сэмплинг, постобработка, запекание текстур, текстурирование мешей |
| [docs/weights.md](docs/weights.md) | Canonical-пакеты, нормализация, GGUF-квантизация, измеренная точность |
| [docs/server.md](docs/server.md) | HTTP API, флаги сервера, workflow задач, встраивание роутера |
| [docs/performance.md](docs/performance.md) | Бенчмарки бэкендов, warm-loop, поведение памяти |
| [FEATURES.md](FEATURES.md) | Инвентаризация фич и user stories |
| [E2E.md](E2E.md) | Матрица end-to-end сценариев и запуск |
| [E2E-RESULTS.md](E2E-RESULTS.md) | Последние результаты real-weight e2e |
| [PERF-NOTES.md](PERF-NOTES.md) | Журнал исследований производительности и оставшиеся рычаги |

## Производительность

Измеренное время инференса (`[stage] infer`: препроцессинг + нейросетевой проход,
без загрузки весов) на NVIDIA GeForce RTX 3060 12 GiB, Windows (WDDM):

| Модель | CUDA | Vulkan | WGPU | Vulkan/CUDA | WGPU/CUDA |
|---|---:|---:|---:|---:|---:|
| pi3 (5 кадров, 518×518) | **4.82 с** | 7.83 с | 13.96 с | 1.62× | 2.90× |
| pi3x (6 кадров, 518×518) | **7.00 с** | 11.10 с | 31.84 с | 1.59× | 4.55× |
| triposr (одно изображение) | **1.23 с** | 3.36 с | 4.89 с | 2.73× | 3.98× |
| vggt (3 кадра, 518×518) | **6.76 с** | 10.96 с | 19.34 с | 1.62× | 2.86× |

TripoSG preview (`--steps 2 --num-tokens 1024 --flash-octree-depth 5`): infer
6.84 с CUDA / 8.56 с Vulkan / 13.13 с WGPU при пике ~8.3–8.7 GiB. TRELLIS.2
`1024_cascade` с текстурой — 438 с на CUDA при пике 7.8 GiB.

Полные таблицы, VRAM, warm-loop и известные ограничения:
[docs/performance.md](docs/performance.md).

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `crates/lux3d-core` | Ядро runtime, контракты, геометрия/экспорт, валидация весов |
| `crates/lux3d-cli` | CLI для инспекции, нормализации и запуска инференса |
| `crates/lux3d-server-core` | Встраиваемый Axum-роутер и HTTP API |
| `crates/lux3d-server` | Standalone HTTP-сервер |
| `docs/` | Гайды: models, inference, VGGT, TRELLIS.2, weights, server, performance |
| `scripts/` | Оценка квантизации, сравнение геометрии, конвертация keypoint-весов |
| [`tools/python_baseline`](tools/python_baseline/README.md) | Python-инструменты паритета и нормализации весов |
| `.github/assets` | Ассеты документации |

## Системные требования

- Rust 1.85 или новее; Cargo с поддержкой `edition = "2024"`.
- Python 3.x для baseline-инструментов и нормализации.
- CUDA-совместимая NVIDIA GPU для проверенного инференса. Референсная тестовая
  карта — RTX 3060 12 GiB.
- Vulkan и WGPU через форк [FerrisMind/candle](https://github.com/FerrisMind/candle);
  проверены на корректность и стабильность памяти, но медленнее CUDA.
- Metal собирается в CI; паритет инференса не проверяется.
- Для полного референсного прогона TripoSG (50 шагов) рекомендуется 16 GiB+ VRAM.
  TRELLIS.2 на `1024_cascade` с текстурой достигает пика ~7.8 GiB.

## Лицензия

Код и документация репозитория — Apache-2.0. Upstream-веса и
каноникализированные артефакты сохраняют оригинальные лицензии и ограничения;
перед перераспространением см. [docs/models.md](docs/models.md#licensing).

## Благодарности

Построено на [FerrisMind/candle](https://github.com/FerrisMind/candle). Реализации
моделей следуют [Pi3](https://github.com/yyfz/Pi3),
[TripoSR](https://github.com/VAST-AI-Research/TripoSR),
[VGGT](https://github.com/facebookresearch/vggt),
[TripoSG](https://github.com/VAST-AI-Research/TripoSG) и
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2).
