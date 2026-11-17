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

## Содержание

- [Что это?](#что-это)
- [Ключевые возможности](#ключевые-возможности)
- [Демо](#демо)
- [Структура репозитория](#структура-репозитория)
- [Быстрый старт](#быстрый-старт)
- [Производительность](#производительность)
- [Системные требования](#системные-требования)
- [Лицензия](#лицензия)

## Что это?

Lux3D / candle-3d — это Rust 2024 workspace, который предоставляет:

- Candle-first runtime для инференса
- инспекцию контрактов поддерживаемых семейств моделей
- нормализацию и валидацию canonical weights
- экспорт геометрии в point cloud и mesh форматы

Поддерживаемые семейства моделей:

- `pi3`
- `pi3x`
- `triposr`
- `vggt` — VGGT-1B: multi-view point cloud, depth и camera инференс с опциональным
  COLMAP-экспортом в feedforward или track-режиме (без bundle adjustment), depth-развёрткой
  в point cloud, трекингом по явным запросам или автоматическим выбором запросов stage A
  и экспортом demo-style GLB-сцены. Оригинальные веса CC-BY-NC-4.0 (non-commercial);
  коммерческий checkpoint распространяется отдельно.
- `triposg` — TripoSG: image-to-mesh инференс (DINOv2-large encoder, 21-блочный rectified-flow
  DiT, SDF VAE decoder/encoder, flash decoder + clean-room DiffDMC по умолчанию), плюс
  TripoSG-scribble с двойным conditioning (`--mode scribble`), батчевые прогоны
  `--num-shapes` по нескольким изображениям и референсный VAE surface-encode
  (`--surface-input`). Лицензия MIT; BriaRMBG 1.4 для RGB-входа поставляется отдельным
  пакетом `rmbg-1.4` под лицензией BRIA.
- `trellis2` — TRELLIS.2-4B: image-to-3D инференс (DINOv3 ViT-L/16 encoder, SS flow DiT,
  shape/texture SLAT flow-трансформеры, FlexiDualGrid VAE, O-Voxel dual-grid меш с PBR-атрибутами
  вершин) плюс пайплайн текстурирования существующих мешей (`--texture-mesh`).
  Референсный `1024_cascade` pipeline по умолчанию: 12 шагов на стадию, guidance 7.5
  (для текстуры 1.0), seed 42. Код и model card — MIT; встроенный DINOv3 — отдельная gated-модель Meta.
  RGB-вход проходит референсное удаление фона RMBG-2.0/BiRefNet — отдельный пакет
  `rmbg-2.0` под лицензией CC-BY-NC-4.0; при валидном alpha-канале RMBG пропускается.

## Ключевые возможности

- `lux3d-core` реализует контракты, загрузку runtime, экспорт и проверку весов.
- `lux3d-cli` предоставляет команды `inspect`, `weights normalize`, `weights quantize`,
  `run` и команды работы с o_voxel-объёмами `ovox mesh2ovox` / `ovox2mesh` / `ovox2glb`.
- `lux3d-server-core` предоставляет OpenAI-style async HTTP API для генерации Pi3, Pi3X, TripoSR, VGGT, TripoSG и TRELLIS.2, а также текстурирования мешей TRELLIS.2 (`POST /v1/meshes/texturings`).
- Python baseline tooling описан в [`tools/python_baseline/README.md`](https://github.com/oxide-lab/Lux3d/blob/main/tools/python_baseline/README.md).
- Лицензирование по каждому семейству моделей можно проверить через CLI перед распространением артефактов.
- **Экспериментально:** инференс на Vulkan и WGPU через `--device vulkan` / `--device wgpu` (сборка с `--features vulkan` / `--features wgpu`). CUDA — основной бэкенд; для путей Pi3/Pi3X/TripoSR и VGGT есть проверки корректности на Vulkan, а для TripoSG — smoke-проверки Vulkan/WGPU (см. [Производительность](#производительность)). WGPU остаётся экспериментальным — может работать нестабильно, давать неверный результат или не запускаться в зависимости от железа и драйверов.

## Демо

https://github.com/user-attachments/assets/20c5ce6c-7447-41ea-a79c-4b611e74c552

## Структура Репозитория

| Путь | Назначение |
|------|------------|
| `crates/lux3d-core` | Основной runtime, контракты, экспорт и валидация весов |
| `crates/lux3d-cli` | CLI-оболочка для инспекции, нормализации и инференса |
| `crates/lux3d-server-core` | Встраиваемый Axum router и HTTP API |
| `crates/lux3d-server` | Standalone HTTP server binary |
| `tools/python_baseline` | Python-скрипты для parity и canonical-weight normalization |
| `.github/assets` | Документационные ассеты проекта |

## Быстрый Старт

### Проверка Workspace

```powershell
cargo metadata --no-deps
cargo run -p lux3d-cli -- --help
```

### Инспекция Контрактов

```powershell
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> pi3
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> pi3x
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposr
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> vggt
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposg
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> trellis2
```

### Нормализация Canonical Weights

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --output-dir <canonical-pi3-dir> pi3
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --output-dir <canonical-pi3x-dir> pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --output-dir <canonical-triposr-dir> triposr
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-vggt-dir> --output-dir <canonical-vggt-dir> vggt
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposg-dir> --output-dir <canonical-triposg-dir> triposg
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-trellis2-dir> --output-dir <canonical-trellis2-dir> trellis2

# Вспомогательный пакет RMBG-2.0 (CC-BY-NC): напрямую через упаковщик.
python -I tools/python_baseline/normalize_weights.py --family rmbg2 `
  --raw-model-dir <raw-RMBG-2.0-dir> --output-dir <canonical-rmbg2-dir>
```

Нормализация TRELLIS.2 упаковывает плоский layout `microsoft/TRELLIS.2-4B` (семь пар
`ckpts/*.json` + `*.safetensors`, внешний разреженный structure-decoder `ss_dec` и
конфиг/модели DINOv3) в canonical-пакет, который ожидают `Trellis2Pipeline::load` и сервер.
Используемый только для текстурирования shape encoder в объединённый canonical-файл весов не
попадает: `--texture-mesh` загружает его из сырого
`ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` или из
`external/shape_enc/model.safetensors` рядом с canonical `model.gguf`
(см. заметки по TRELLIS.2 в разделе запуска инференса).

### Квантование Весов в GGUF

`weights quantize` упаковывает canonical `model.safetensors` в `model.gguf`.
Квантуются только rank-2 матрицы `*.weight`; нормы, bias, эмбеддинги, свёртки
и строки, не кратные блоку dtype, остаются в fallback dtype. Квантованные
linear-слои идут через `QMatMul` (integer dot-product ядра), остальное
де-квантуется по требованию.

Конвертер встраивает соседний `resolved_config.json` в metadata GGUF, поэтому
`model.gguf` может быть **самостоятельным пакетом**: положите его одного в
каталог и передайте этот каталог в `--model-path`. Загрузчик синтезирует
manifest/checksums и читает конфиг из GGUF. Если `model.gguf` лежит рядом с
полным canonical-пакетом, файлы пакета по-прежнему проверяются, а веса берутся
из GGUF. Rank-0 тензоры (тренировочные буферы вроде BatchNorm
`num_batches_tracked`) опускаются — при инференсе они не используются.
`--model-path` принимает и сам `.gguf`-файл, поэтому кванты можно раскладывать
плоско (`TripoSG-1.5B-Q8_0.gguf`, `VGGT-1B-Q8_0.gguf`, ...).

`general.architecture` берётся из resolved config: сначала top-level
`architecture`, затем `family`, затем имя модели (`triposg`, `vggt`, `rmbg`).
Например, RMBG пишет `BriaRMBG`, TripoSG — `triposg`, VGGT (в конфиге нет
обоих полей) — `vggt`. `lux3d.family` всегда хранит family рантайма.

```powershell
# Q8_0 веса, F16 fallback (эталонное качество по умолчанию)
cargo run -p lux3d-cli -- weights quantize triposg `
  --source <canonical-triposg-dir>/model.safetensors `
  --output <gguf-triposg-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# K-кванты: меньше файл, блоки по 256
cargo run -p lux3d-cli -- weights quantize vggt `
  --source <canonical-vggt-dir>/model.safetensors `
  --output <gguf-vggt-dir>/model.gguf --dtype q4_k --fallback-dtype f16

# Pi3, Pi3X и TripoSR используют тот же конвертер.
cargo run -p lux3d-cli -- weights quantize pi3 `
  --source <canonical-pi3-dir>/model.safetensors `
  --output <gguf-pi3-dir>/model.gguf --dtype q8_0 --fallback-dtype f16
cargo run -p lux3d-cli -- weights quantize pi3x `
  --source <canonical-pi3x-dir>/model.safetensors `
  --output <gguf-pi3x-dir>/model.gguf --dtype q8_0 --fallback-dtype f16
cargo run -p lux3d-cli -- weights quantize triposr `
  --source <canonical-triposr-dir>/model.safetensors `
  --output <gguf-triposr-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# TRELLIS.2: квантуются только линейные слои flow-DiT (путь `QMatMul`);
# тензоры DINOv3/VAE/декодеров, 3D-свёртки, нормы и смещения остаются F16
# (их загрузчики читают плотные тензоры). F16-сборка строится без префиксов.
cargo run -p lux3d-cli -- weights quantize trellis2 `
  --source <canonical-trellis2-dir>/model.safetensors `
  --output trellis2-q8_0.gguf --dtype q8_0 --fallback-dtype f16 `
  --prefix ss_flow. --prefix shape_dit_512. --prefix shape_dit_1024. `
  --prefix tex_dit_512. --prefix tex_dit_1024.

# TripoSG-scribble: квантуется ТОЛЬКО DiT с двойным cross-attention (текстовый
# энкодер, VAE и DINOv2 остаются float). --source — каталог сырого чекпойнта
# `VAST-AI/TripoSG-scribble` (shard transformer находится автоматически);
# FAMILY — `triposg`. Запись в `<checkpoint>/transformer/model.gguf` включает
# авто-подхват; можно также передать сам `.gguf` в `--model-path`.
# У GGUF нет встроенного resolved config (в сыром чекпойнте его нет), поэтому
# это DiT-override, а не standalone-пакет.
cargo run -p lux3d-cli -- weights quantize triposg --scribble `
  --source <raw-TripoSG-scribble-dir> `
  --output <raw-TripoSG-scribble-dir>/transformer/model.gguf `
  --dtype q8_0 --fallback-dtype f16

# RMBG-1.4 — вспомогательный пакет: отдельный GGUF, F16, те же правила standalone.
cargo run -p lux3d-cli -- weights quantize --aux-family rmbg `
  --source <canonical-rmbg-dir>/model.safetensors `
  --output <gguf-rmbg-dir>/model.gguf --dtype f16 --fallback-dtype f16

# RMBG-2.0/BiRefNet для TRELLIS.2 — тоже отдельный auxiliary-пакет.
# F16 — референсный дефолт по точности (alpha max |diff| = 1 на валидационном
# изображении); Q8_0 сжимает 0.82 GiB до 0.24 GiB при alpha max |diff| = 7.
cargo run -p lux3d-cli -- weights quantize --aux-family rmbg2 `
  --source <canonical-rmbg2-dir>/model.safetensors `
  --output <gguf-rmbg2-f16-dir>/model.gguf --dtype f16 --fallback-dtype f16
cargo run -p lux3d-cli -- weights quantize --aux-family rmbg2 `
  --source <canonical-rmbg2-dir>/model.safetensors `
  --output <gguf-rmbg2-q8_0-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# Запуск только по GGUF-каталогам; safetensors/manifest/checksums не нужны.
cargo run -p lux3d-cli -- run triposg --device cuda `
  --model-path <gguf-triposg-dir> --rmbg-model-path <gguf-rmbg-dir> `
  --source <image> --output out.obj
cargo run -p lux3d-cli -- run trellis2 --device cuda `
  --model-path <gguf-trellis2-dir> --rmbg-model-path <gguf-rmbg2-f16-dir> `
  --source <image> --output out.glb

# TripoSG-scribble сам подхватывает соседний `transformer/model.gguf`;
# standalone DiT `.gguf` можно передать прямо в `--model-path`. Float-веса
# текстового энкодера/VAE/DINOv2 берутся из LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR
# (по умолчанию `models/TripoSG-scribble`).
cargo run -p lux3d-cli -- run triposg --device cuda --mode scribble `
  --prompt "a cat with wings" --model-path models/TripoSG-scribble `
  --source <scribble.png> --output out.glb
cargo run -p lux3d-cli -- run triposg --device cuda --mode scribble `
  --prompt "a cat with wings" --model-path <scribble-dit-q8_0.gguf> `
  --source <scribble.png> --output out.glb

# `--all-2d` квантует и rank-2 тензоры без суффикса `.weight`;
# `--prefix transformer.` (можно повторять) ограничивает набор для отладки.
```

Поддерживаемые `--dtype`: `f16`, `f32`, `bf16`, `q8_0`, `q4_0`, `q4_1`,
`q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`. `q8_1`/`q8_k` —
активационные GGML-типы, как веса они отклоняются. GGUF без встроенного
конфига (при конвертации не было соседнего `resolved_config.json`) годится
только как дополнение к полному canonical-пакету; standalone-загрузка его
отклоняет.

Замеры на canonical-пакетах (`RTX 3060 12 GiB`; TripoSG
`--steps 2 --num-tokens 1024 --flash-octree-depth 5`, VGGT/Pi3/Pi3X 3 кадра,
TripoSR 1 изображение при `--mc-resolution 256`):

| модель | вариант | размер GGUF | квантовано матриц |
|---|---|---|---|
| TripoSG | canonical F32 | 7.40 GiB | — |
| TripoSG | q8_0 | 1.97 GiB | 527 |
| TripoSG | q6_k | 1.52 GiB | 525 |
| TripoSG | q5_k | 1.28 GiB | 525 |
| TripoSG | q4_k | 1.05 GiB | 525 |
| VGGT | canonical F32 | 4.68 GiB | — |
| VGGT | q8_0 | 1.33 GiB | 384 |
| VGGT | q6_k | 1.09 GiB | 331 |
| VGGT | q5_k | 0.95 GiB | 331 |
| VGGT | q4_k | 0.82 GiB | 331 |
| Pi3 | canonical F32 | 3.57 GiB | — |
| Pi3 | f16 | 1.79 GiB | все тензоры |
| Pi3 | q8_0 | 0.95 GiB | 318 |
| Pi3 | q6_k | 0.74 GiB | 318 |
| Pi3 | q5_k | 0.62 GiB | 318 |
| Pi3 | q4_k | 0.51 GiB | 318 |
| Pi3X | canonical F32 | 5.07 GiB | — |
| Pi3X | f16 | 2.53 GiB | все тензоры |
| Pi3X | q8_0 | 1.37 GiB | 466 |
| Pi3X | q6_k | 1.06 GiB | 466 |
| Pi3X | q5_k | 0.90 GiB | 466 |
| Pi3X | q4_k | 0.74 GiB | 466 |
| TripoSR | canonical F32 | 1.56 GiB | — |
| TripoSR | f16 | 0.78 GiB | все тензоры |
| TripoSR | q8_0 | 0.42 GiB | 244 |
| TripoSR | q6_k | 0.33 GiB | 235 |
| TripoSR | q5_k | 0.27 GiB | 235 |
| TripoSR | q4_k | 0.23 GiB | 235 |
| TRELLIS.2-4B | canonical (смесь BF16/F16/F32) | 15.07 GiB | — |
| TRELLIS.2-4B | f16 | 14.50 GiB | все тензоры |
| TRELLIS.2-4B | q8_0 | 8.87 GiB | 1074 линейных слоёв DiT |
| TRELLIS.2-4B | q6_k | 7.41 GiB | 1070 |
| TRELLIS.2-4B | q5_k | 6.61 GiB | 1070 |
| TRELLIS.2-4B | q4_k | 5.86 GiB | 1070 |
| TripoSG-scribble DiT | чекпойнт F16 | 3.13 GiB | — |
| TripoSG-scribble DiT | q8_0 | 1.67 GiB | 308 матриц |
| TripoSG-scribble DiT | q6_k | 1.29 GiB | 307 |
| TripoSG-scribble DiT | q5_k | 1.08 GiB | 307 |
| TripoSG-scribble DiT | q4_k | 0.88 GiB | 307 |

F16 — чистое преобразование точности, не блочное квантование: все тензоры
хранятся в 16 битах, линейные слои идут обычным путём без `QMatMul`. Цена —
примерно вдвое больше Q8_0, зато результат сохраняется: на CUDA/Vulkan/WGPU/CPU
дрейф не больше 0.034% / 0.042% / 0.0014% CD (Pi3/Pi3X/TripoSR) при
F@1% = 1.000 на каждом бэкенде. Сборка:
`--dtype f16 --fallback-dtype f16`.

Квантование TRELLIS.2-4B — только DiT по политике: пять flow-трансформеров
(`ss_flow`, `shape_dit_512/1024`, `tex_dit_512/1024`) грузят линейные слои
через путь `QMatMul`, а DINOv3 (`image_cond`), разреженные VAE-декодеры и
плотный structure-декодер читают плотные тензоры (3D-свёртки, нормы), поэтому
эти стадии остаются F16 во всех вариантах. GGUF-контейнер candle ограничивает
ранг тензора четырьмя, поэтому 3D-свёртки хранятся сплющенными, и загрузчик
возвращает им форму. K-кванты пропускают четыре матрицы `input_layer`, длина
строки которых не делится на блок 256; Q8_0 дополнительно пропускает ещё одну
(строка длины 8).

У TripoSR Q8_0 квантует 244 матрицы; K-кванты используют блок 256, поэтому
девять матриц с длиной строки не кратной 256 остаются в F16.

RMBG-1.4 (вспомогательный): F32 0.16 GiB → F16 0.08 GiB; записано 686
тензоров, 112 rank-0 тренировочных буферов опущено.

TripoSG-scribble следует той же DiT-only политике: квантуется только
трансформер с двойным cross-attention, а CLIP-энкодер, VAE SDF-декодер и
DINOv2-энкодер скетча остаются в сырых float safetensors. Эталон — нативный
F16 DiT чекпойнта, поднятый до F32 (`--dtype` по умолчанию `f32`), с теми же
seed/шагами/изображением. Дрейф scribble (CUDA, RTX 3060;
`cat_with_wings.png`, промпт "a cat with wings", 16 шагов, 512 токенов,
dense/hierarchical depth 8, seed 42; эталон 80 109 вершин / 159 976 граней):

| вариант | CD rel | F@0.5% | F@1% | voxel IoU | normal | вершин | граней | max drift | mean drift |
|---|---|---|---|---|---|---|---|---|---|
| q8_0 | 0.461% | 0.626 | 0.907 | 0.268 | 0.925 | 82 145 | 164 044 | 0.0669 | 0.0131 |
| q6_k | 0.603% | 0.515 | 0.812 | 0.204 | 0.908 | 84 865 | 169 440 | 0.1043 | 0.0171 |
| q5_k | 0.557% | 0.523 | 0.857 | 0.203 | 0.920 | 76 732 | 153 204 | 0.0654 | 0.0158 |
| q4_k | 0.744% | 0.444 | 0.742 | 0.167 | 0.893 | 83 340 | 166 368 | 0.1421 | 0.0211 |

Max/mean drift — существующие nearest-neighbour метрики
`compare_geometry.py` (`max`, `chamfer_l1_mean` в единицах меша); voxel IoU
показывает изменение топологии (число вершин/граней marching cubes меняется
до 6%). CFG-distilled scribble DiT заметно чувствительнее к квантованию, чем
image-модель TripoSG: только Q8_0 удерживает F@1% выше 0.90, а все K-кванты
существенно меняют топологию поверхности. Для scribble предпочтителен Q8_0;
Q6_K — компактный fallback, Q5_K/Q4_K — preview-уровень. DiT GGUF — это
override, а не standalone-пакет: в сырых чекпойнтах нет
`resolved_config.json`, поэтому остальные веса всегда грузятся из каталога
чекпойнта.

### Метрики Качества Квантования

`scripts/compare_geometry.py` считает отклонение от float-эталона:
симметричный Chamfer-L1 (относительно диагонали bbox эталона), F-score@tau,
voxel IoU по поверхности, normal consistency для мешей; pointmap/depth PSNR,
AbsRel, Chamfer и camera-pose AUC для тензорных дампов VGGT.
`scripts/eval_quantization.py` прогоняет эталон и все квантованные пакеты и
печатает таблицу.

```powershell
# Дрейф меша (TripoSG/TripoSR)
python scripts/compare_geometry.py mesh <reference.obj> <quantized.obj>

# VGGT: сохранить сырые тензоры рядом с PLY и сравнить
cargo run -p lux3d-cli -- run vggt --model-path <pkg> --source <frames> --output <out.ply> --dump-npz <out.npz>
python scripts/compare_geometry.py vggt <reference.npz> <quantized.npz>

# Серия прогонов по нескольким квантованным пакетам
python scripts/eval_quantization.py triposg --binary target/release/lux3d-cli.exe `
  --reference <canonical-triposg-dir> --variant q8_0=<dir-q8_0> --variant q4_k=<dir-q4_k> `
  --source <input-image> --device cuda --output-dir out/eval -- --steps 2 --num-tokens 1024 --flash-octree-depth 5

# Тот же цикл работает для point-cloud моделей и TripoSR.
python scripts/eval_quantization.py pi3x --binary target/release/lux3d-cli.exe `
  --reference <canonical-pi3x-dir> --variant q8_0=<dir-q8_0> `
  --source <frames-dir> --device cuda --output-dir out/eval-pi3x
python scripts/eval_quantization.py triposr --binary target/release/lux3d-cli.exe `
  --reference <canonical-triposr-dir> --variant q4_k=<dir-q4_k> `
  --source <input-image> --device cuda --output-dir out/eval-triposr `
  --mc-resolution 256 --mc-threshold 25.0

# TRELLIS.2 (варианты — standalone `.gguf`-файлы, не каталоги)
python scripts/eval_quantization.py trellis2 --binary target/release/lux3d-cli.exe `
  --reference <canonical-trellis2-dir> --variant q8_0=trellis2-q8_0.gguf `
  --source <input-image> --device cuda --output-dir out/eval-trellis2 -- `
  --pipeline-type 512 --steps 12 --texture false --seed 42

# TripoSG-scribble (варианты — DiT `.gguf`; каталог сырого эталона
# экспортируется как LUX3D_TRIPOSG_SCRIBBLE_MODEL_DIR для variant-прогонов)
python scripts/eval_quantization.py triposg --scribble `
  --reference <raw-TripoSG-scribble-dir> --variant q8_0=<scribble-dit-q8_0.gguf> `
  --source <scribble.png> --device cuda --output-dir out/eval-scribble -- `
  --mode scribble --prompt "a cat with wings"
```

Дрейф меша TripoSG (2 шага, 1024 токена, depth 5; эталон — F32 той же
подложки):

| вариант | backend | CD rel | F@1% | voxel IoU | VRAM (дельта) | время |
|---|---|---|---|---|---|---|
| q8_0 | CUDA | 0.026% | 0.996 | 0.994 | 3.1 GiB | 10.4 s |
| q8_0 | Vulkan | 0.184% | 0.978 | 0.970 | 3.9 GiB | 13.3 s |
| q6_k | CUDA | 0.135% | 0.988 | 0.988 | — | — |
| q6_k | Vulkan | 0.134% | 0.995 | 0.995 | — | — |
| q5_k | CUDA | 0.155% | 0.993 | 0.994 | — | — |
| q5_k | Vulkan | 0.224% | 0.980 | 0.984 | — | — |
| q4_k | CUDA | 0.253% | 0.973 | 0.953 | 2.2 GiB | 10.0 s |
| q4_k | Vulkan | 0.180% | 0.982 | 0.972 | 2.3 GiB | 12.7 s |
| F32 | CUDA | — | — | — | 8.6 GiB | 14.9 s |
| F32 | Vulkan | — | — | — | 8.8 GiB | 17.1 s |

Двухшаговый сэмплинг чувствителен к входу. На жёстком RGB-входе
(`mesh-input-rgb.png`, с путём через RMBG) Q5_K и Q4_0 ушли до ~1.4% Chamfer,
тогда как Q4_K остался на 0.23%, а Q8_0 — на 0.06%. Эти выбросы повторяемы
на данном входе, но не воспроизводятся на alpha-входе из таблицы, поэтому
Q5_K/Q4_0 считаем входозависимыми; для деплоя — Q8_0/Q6_K/Q4_K.

VGGT (3 кадра; эталон — F32 тензоры той же подложки; pose AUC по 2
относительным парам кадров):

| вариант | backend | pointmap PSNR | depth PSNR | depth AbsRel | CD rel | pose AUC@5 | VRAM (дельта) |
|---|---|---|---|---|---|---|---|
| q8_0 | CUDA | 53.1 dB | 48.1 dB | 0.0038 | 0.274% | 1.0 | 4.7 GiB |
| q8_0 | Vulkan | 55.9 dB | 53.5 dB | 0.0020 | 0.182% | 1.0 | 4.4 GiB |
| q8_0 | WGPU | 55.4 dB | 51.0 dB | 0.0027 | 0.193% | 1.0 | — |
| q6_k | CUDA | 45.7 dB | 44.8 dB | 0.0053 | 0.571% | 1.0 | — |
| q6_k | Vulkan | 46.5 dB | 49.5 dB | 0.0026 | 0.413% | 0.5 | — |
| q5_k | CUDA | 34.7 dB | 35.9 dB | 0.0144 | 1.54% | 0.0 | — |
| q5_k | Vulkan | 36.4 dB | 35.0 dB | 0.0180 | 1.39% | 1.0 | — |
| q4_k | CUDA | 24.7 dB | 23.5 dB | 0.0683 | 3.50% | 0.0 | 4.2 GiB |
| q4_k | Vulkan | 26.9 dB | 25.8 dB | 0.0511 | 2.84% | 0.0 | 3.7 GiB |
| F32 | CUDA | — | — | — | — | — | 6.7 GiB |
| F32 | Vulkan | — | — | — | — | — | 7.3 GiB |

Point-cloud Pi3/Pi3X и меш TripoSR (детали CUDA; эталон — F32 той же подложки;
Pi3/Pi3X 3 кадра, TripoSR `mesh-input.png` при 256):

| семейство | вариант | CD rel | F@0.5% | F@1% | voxel IoU | normal |
|---|---|---|---|---|---|---|
| Pi3 | f16 | 0.034% | 1.000 | 1.000 | — | — |
| Pi3 | q8_0 | 0.166% | 0.990 | 0.99995 | — | — |
| Pi3 | q6_k | 0.139% | 0.985 | 0.99907 | — | — |
| Pi3 | q5_k | 0.851% | 0.469 | 0.706 | — | — |
| Pi3 | q4_k | 0.813% | 0.475 | 0.696 | — | — |
| Pi3X | f16 | 0.037% | 1.000 | 1.000 | — | — |
| Pi3X | q8_0 | 0.081% | 0.99999 | 1.000 | — | — |
| Pi3X | q6_k | 0.873% | 0.129 | 0.598 | — | — |
| Pi3X | q5_k | 2.279% | 0.000 | 0.000 | — | — |
| Pi3X | q4_k | 0.789% | 0.274 | 0.719 | — | — |
| TripoSR | f16 | 0.001% | 1.000 | 1.000 | 0.998 | 1.000 |
| TripoSR | q8_0 | 0.036% | 0.9997 | 0.99994 | 0.920 | 0.993 |
| TripoSR | q6_k | 0.050% | 0.9998 | 0.99995 | 0.886 | 0.991 |
| TripoSR | q5_k | 0.058% | 0.9996 | 0.99985 | 0.863 | 0.990 |
| TripoSR | q4_k | 0.129% | 0.9964 | 0.99985 | 0.682 | 0.978 |

Те же квантованные пакеты на всех бэкендах, каждый сравнивается с F32-прогоном
того же бэкенда. F32-выходы совпадают между бэкендами в пределах 0.04% CD при
F@1% = 1.000, поэтому разброс ниже — поведение квантования, а не шум эталона:

| семейство | вариант | CD rel CUDA / Vulkan / WGPU / CPU | F@1% CUDA / Vulkan / WGPU / CPU |
|---|---|---|---|
| Pi3 | q8_0 | 0.166 / 0.172 / 0.259 / 0.251% | 0.99995 / 1.000 / 0.9999 / 0.9999 |
| Pi3 | q6_k | 0.139 / 0.455 / 0.546 / 0.555% | 0.99907 / 0.966 / 0.919 / 0.895 |
| Pi3 | q5_k | 0.851 / 0.190 / 0.240 / 0.209% | 0.706 / 0.995 / 0.986 / 0.990 |
| Pi3 | q4_k | 0.813 / 1.191 / 0.964 / 1.016% | 0.696 / 0.535 / 0.663 / 0.644 |
| Pi3X | q8_0 | 0.081 / 0.071 / 0.086 / 0.155% | 1.000 / 1.000 / 1.000 / 1.000 |
| Pi3X | q6_k | 0.873 / 1.818 / 1.181 / 1.613% | 0.598 / 0.070 / 0.237 / 0.089 |
| Pi3X | q5_k | 2.279 / 3.383 / 2.819 / 2.749% | 0.000 / 0.000 / 0.000 / 0.000 |
| Pi3X | q4_k | 0.789 / 0.328 / 0.383 / 0.370% | 0.719 / 0.947 / 0.930 / 0.938 |
| TripoSR | q8_0 | 0.036 / 0.025 / 0.032 / 0.028% | 0.9999 / 1.000 / 1.000 / 0.9999 |
| TripoSR | q6_k | 0.050 / 0.052 / 0.056 / 0.038% | 0.9999 / 1.000 / 0.9999 / 1.000 |
| TripoSR | q5_k | 0.058 / 0.055 / 0.066 / 0.061% | 0.9998 / 0.9999 / 0.9999 / 0.9999 |
| TripoSR | q4_k | 0.129 / 0.131 / 0.132 / 0.136% | 0.9998 / 0.9997 / 0.9997 / 0.9997 |

Voxel IoU у TripoSR тоже не зависит от бэкенда: 0.920/0.944/0.930/0.939 для
Q8_0 и 0.682/0.675/0.675/0.663 для Q4_K (CUDA/Vulkan/WGPU/CPU), так что
предупреждение о топологии ниже справедливо везде. F16 в таблицу не включён,
потому что это не блочное квантование, но он держит F@1% = 1.000 на всех
четырёх бэкендах при CD ≤ 0.034% (Pi3), ≤ 0.042% (Pi3X) и ≤ 0.0014%
(TripoSR); у TripoSR voxel IoU 0.997–0.998, normal consistency ≥ 0.9996.

Q8_0 — единственный равномерно безопасный вариант: F@1% ≥ 0.9999 для всех трёх
семейств на всех четырёх бэкендах. Pi3 Q6_K отличен на CUDA (0.99907), но
падает до 0.895–0.966 на CPU/Vulkan/WGPU, поэтому это выбор только для CUDA;
Pi3 Q5_K/Q4_K зависят от бэкенда, в продакшн их не стоит. Pi3X требует
Q8_0 — Q6_K/Q5_K/Q4_K разваливаются, хотя ошибка хранения весов растёт
монотонно. Чувствительный модуль — image `encoder.`: если оставить его в F16,
Q6_K восстанавливает F@1% с 0.598 до 0.99996, но пакет получается 1.40 GiB
против 1.37 GiB у полностью Q8_0, так что Q8_0 строго лучше. TripoSR самый
устойчивый: Q8_0–Q4_K держат F@1% ≥ 0.9997 на каждом бэкенде. Но Q4_K теряет
топологию поверхности (voxel IoU 0.66–0.68 против 0.92–0.94 у Q8_0), поэтому
при важности топологии выбирайте Q6_K/Q8_0.

TripoSG и VGGT: TripoSG укладывается в ~0.25% Chamfer даже на Q4_K, поэтому
Q4_K/Q5_K для него практичны. VGGT чувствителен к позе: Q8_0 почти без
потерь, Q6_K — приемлемый запасной вариант, Q5_K — на границе, Q4_K
непригоден (pointmap и поза камеры разваливаются). Это совпадает с опубликованным
результатом: наивное 4-битное квантование разрушает реконструкцию VGGT-подобных
моделей. Перед выкаткой младшего бита прогоните
`--dump-npz` и скрипты выше на своём валидационном наборе. RMBG-1.4 — маленькая
matting-сеть, держим её отдельным F16 GGUF: размер вдвое меньше, дрейф маски
не измеряется.

Дрейф меша TRELLIS.2-4B (CUDA, RTX 3060 12 GiB; пайплайн 512, 12 шагов на
сэмплер, seed 42, текстуры выключены, `mesh-input.png`; эталон — canonical
F32 на той же подложке). F16 практически точен; выход O-Voxel dual-grid
дискретен, поэтому квантование переключает часть воксельных оккупаций —
voxel IoU измеряет именно дрейф топологии, а не позиции:

| вариант | CD rel | F@0.5% | F@1% | voxel IoU | normal | вершин (реф. 914,470) | загрузка | инференс |
|---|---|---|---|---|---|---|---|---|
| f16 | 0.0027% | 0.9999 | 1.000 | 0.996 | 0.993 | 914,157 | 0.03 s | 142.8 s |
| q8_0 | 0.187% | 0.9139 | 0.9835 | 0.623 | 0.904 | 943,514 | 0.03 s | 98.7 s |
| q6_k | 0.199% | 0.9181 | 0.9813 | 0.609 | 0.905 | 952,678 | 0.03 s | 100.5 s |
| q5_k | 0.274% | 0.8839 | 0.9729 | 0.445 | 0.890 | 893,909 | 0.03 s | 96.3 s |
| q4_k | 0.311% | 0.8331 | 0.9625 | 0.400 | 0.885 | 887,078 | 0.03 s | 95.7 s |

Квантованные линейные слои DiT ещё и быстрее: 12-шаговый пайплайн 512 идёт за
~96–101 с при Q8_0 и ниже против 135 с на F32 и 143 с на F16 — ядра `QMatMul`
не делают проход деквантования. (Исключение — WGPU: его packed-q8 matmul
примерно в 10 раз медленнее dense f32 GEMM, поэтому Lux3D один раз
деквантует все packed-веса при загрузке; `LUX3D_WGPU_DEQUANT_WEIGHTS=0`
возвращает packed-путь.) Производственные варианты — Q8_0/Q6_K
(F@1% ≥ 0.98, нормали ≥ 0.90); Q5_K/Q4_K — для превью. Путь `1024_cascade`
проверен от конца до конца на квантованных весах Q4_K (2-шаговый прогон с
текстурами: SS DiT, shape DiT 512/1024 и texture DiT грузятся через
`QMatMul` и декодируются).

Проверка бэкендов на реальных весах (один вход, seed 42): geometry-only прогон
512 проходит на CPU, CUDA, Vulkan и WGPU; полный `1024_cascade` с текстурами
проходит на CUDA за 438 с с **пиком 7.8 GiB** на RTX 3060 12 GiB. GGUF-варианты
F16 и Q8_0 работают на всех четырёх тестируемых бэкендах. K-кванты
(Q4_K/Q5_K/Q6_K) работают на CPU, CUDA и Vulkan, но падают на WGPU при таком
размере модели: dispatch Q8K-раундтрипа активаций в candle превышает лимит
65 535 workgroup для MLP шириной 8 192; до исправления ядра candle безопасный
квантованный выбор для WGPU — Q8_0. На Vulkan дополнительно потребовался
Lux3D-workaround для неограниченного `index_select` в candle при >65 535 строк
(разреженные VAE-gather'ы достигают ~500k строк на разрешении 512): рантайм
режет gather на чанки лимитного размера, после чего Vulkan совпадает с CPU/CUDA
по геометрии и детерминирован от запуска к запуску.

Замеры q8_0 512 geometry-only (RTX 3060 12 GiB, release CLI, один шаг
сэмплера): CUDA 22.0 с, Vulkan 55.9 с, WGPU 97.6 с (было 97 / 129 / 303 с).
На WGPU основной выигрыш дали ragged-F16 GEMM'ы разреженного VAE, ушедшие в
register-tiled ядро, и разворачивание аккумуляторов WGPU-GEMM в именованные
скаляры (динамически индексированные массивы Naga клала в local memory).
Оставшийся разрыв Vulkan — задержка submission/fence в WDDM: разреженный VAE
и DiT выдают десятки тысяч мелких dispatch'ей, блокирующие drain'ы занимают
~32 с при сумме GPU-ядер в несколько секунд. WGPU дополнительно платит ~7 мс
на compute pass и считает DiT в точном F32, тогда как CUDA исполняет q8 на
int8-ядрах; reduced-precision coop-matrix для TRELLIS.2 не включается, потому
что дискретная O-Voxel occupancy меняет топологию меша.

### Запуск Инференса

```powershell
# Pi3 -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3
cargo run -p lux3d-cli -- run pi3 --model-path <canonical-pi3-dir> --source <input-sequence> --output <output-file.ply>

# Pi3X core -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3x
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-sequence> --conditions <conditions-file> --output <output-file.ply>

# Pi3X VO -> PLY с явной директорией canonical package в layout как 3d/canonical-weights/pi3x
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-video> --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output <output-file.ply>

# TripoSR -> GLB с явной директорией canonical package в layout как 3d/canonical-weights/triposr
cargo run -p lux3d-cli -- run triposr --model-path <canonical-triposr-dir> --source <input-image> --mc-resolution 256 --mc-threshold 25.0 --output <output-file.glb>

# VGGT -> PLY (world points фильтруются по confidence > 1.0 по умолчанию)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --output <output-file.ply>

# Автоскачивание canonical package VGGT (oxide-lab/VGGT-1B)
cargo run -p lux3d-cli -- run vggt --source <input-sequence> --output <output-file.ply>

# VGGT pad-препроцессинг, f16-веса, последовательные чанки по 2 кадра, COLMAP feedforward
# экспорт + point cloud из depth рядом с основным confidence-filtered PLY
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --preprocess-mode pad --vggt-dtype f16 --batch-size 2 --colmap-dir <sparse-dir> --depth-points <depth.ply> --output <output-file.ply>

# VGGT square-1024 препроцессинг (официальный путь demo_colmap.py), feedforward COLMAP-экспорт
# с point-head источником точек и seeded случайным лимитом точек
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --preprocess-mode square --colmap-dir <sparse-dir> --colmap-point-source pointmap --colmap-seed 42 --output <output-file.ply>

# VGGT из одного файла изображения (сцена из одного кадра)
cargo run -p lux3d-cli -- run vggt --source <image.png> --output <output-file.ply>

# VGGT явные track-запросы -> JSON треков
cargo run -p lux3d-cli -- run vggt --source <input-sequence> --track-queries '[{"frame":0,"x":259,"y":259}]' --track-output tracks.json --output <output-file.ply>

# VGGT автоматический выбор запросов stage A (DINOv2-reg ranking + FPS-точки) и track-based
# COLMAP-экспорт без BA (COLMAP-ready вход для внешнего BA)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --track-auto --query-frame-num 8 --max-query-pts 4096 --colmap-dir <sparse-dir> --colmap-mode track --output <output-file.ply>

# VGGT demo-style GLB-сцена (confidence-filtered point cloud + опциональные конусы камер)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --glb <scene.glb> --glb-conf-percentile 50 --glb-show-cam --output <output-file.ply>

# TripoSG -> GLB (по умолчанию: 50 rectified-flow шагов, CFG 7.0, seed 42, flash decoder depth 9)
cargo run -p lux3d-cli --features cuda -- run triposg --rmbg-model-path <canonical-rmbg-dir> --model-path <canonical-triposg-dir> --source <input-image> --output <output-file.glb>
# Flash decoder включён по умолчанию; отключить и вернуться к dense/hierarchical MC:
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --flash-decoder false --dense-octree-depth 8 --hierarchical-octree-depth 9 --output <output-file.glb>
# Быстрый preview: 2 шага, flash depth 5
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --steps 2 --flash-decoder true --flash-octree-depth 5 --output <preview.glb>

# Автоскачивание canonical package TripoSG (oxide-lab/TripoSG); RMBG 1.4 — отдельный пакет
cargo run -p lux3d-cli --features cuda -- run triposg --source <input-image> --output <output-file.glb>

# Батчевый TripoSG: повторяемый --source (по изображению на источник) и --num-shapes на промпт;
# общий batch = источники * --num-shapes, выходы получают суффиксы _0, _1, ...
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <image-a.png> --source <image-b.png> --num-shapes 2 --output <output.glb>

# Референсный VAE surface encode TripoSG (scripts/inference_vae.py): .npy с keyed dict
# {"surface_points": [N,3], "surface_normals": [N,3]}; image/DINO/denoise пропускаются
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --surface-input <surface.npy> --output <output.glb>

# TripoSG-scribble: двойной conditioning CLIP-промпта и scribble-изображения (сырой
# checkpoint VAST-AI/TripoSG-scribble или квантованный DiT .gguf); --prompt обязателен,
# --scribble-conf/--prompt-conf переопределяют веса ветвей (0.4 / 1.0)
cargo run -p lux3d-cli --features cuda -- run triposg --mode scribble --prompt "a cat with wings" --model-path models/TripoSG-scribble --source <scribble.png> --scribble-conf 0.4 --prompt-conf 1.0 --output <output.glb>

# Опции референсного post-process/загрузки: `--faces <n>` (целевое число граней после merge-close +
# meshopt quadric simplification — clean-room аппроксимация upstream pymeshlab; 0 = выкл.,
# upstream-эквивалент `-1`; no-op при текущем числе граней или больше), `--timesteps` (явное
# расписание denoise, заменяет `--steps`), `--bounds` (окно SDF-запроса `min_x,...,max_z` в
# нормализованном фрейме модели; скаляр `b` = симметричный куб `[-b, b]`), `--dtype` (`f32`
# референс; для `f16` есть load/DINO/VAE smoke-покрытие; `bf16` принимается, но не проверен)
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --faces 50000 --timesteps 999,777,500,250,0 --bounds "-1.1,-1.1,-1.1,1.1,1.1,1.1" --dtype f16 --output <output.glb>

# TRELLIS.2 -> текстурированный GLB + PBR-sidecar из canonical package
# (по умолчанию: pipeline 1024_cascade, 12 шагов на сэмплер, guidance 7.5, seed 42, текстура включена).
# RGB-входу нужен отдельный пакет RMBG-2.0; соседний каталог `rmbg-2.0`/`RMBG-2.0`
# рядом с --model-path находится автоматически.
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --output <output-file.glb>
# Прогон без текстур (без texture SLAT / tex VAE / PBR-sidecar):
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --texture false --output <output-file.glb>
# Быстрый pipeline 512 с переопределениями:
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --pipeline-type 512 --steps 12 --seed 42 --output <preview.glb>
# Сырой каталог checkpoint'ов microsoft/TRELLIS.2-4B (pipeline.json + ckpts/) принимается как есть:
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <raw-TRELLIS.2-4B-dir> --rmbg-model-path <raw-RMBG-2.0-dir> --source <input-image> --output <output-file.glb>
# Резидентный режим стадий (upstream low_vram=False): все загруженные стадии остаются на устройстве
# на весь прогон вместо дефолтной загрузки/выгрузки по стадиям; те же выходы, быстрее, выше пик VRAM
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --no-low-vram --output <output-file.glb>
# Вернуть pre-remesh cleanup-цепочку или настроить band/project дефолтного narrow-band remesh:
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --no-remesh --output <output-file.glb>
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --remesh-band 1 --remesh-project 0.5 --output <output-file.glb>
# Переопределение размера запекаемой текстуры (по умолчанию 2048; CLI принимает 64..=8192 кратное 8;
# `--no-texture-export` полностью пропускает UV unwrap + bake, `--no-uv-unwrap` — только unwrap):
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <canonical-trellis2-dir> --rmbg-model-path <canonical-rmbg2-dir> --source <input-image> --texture-size 2048 --output <output-file.glb>

# Текстурирование существующего меша (upstream Trellis2TexturingPipeline):
# --source — condition-изображение, --texture-mesh принимает .glb/.obj/.ply/.stl, выход обязательно .glb.
# Shape encoder берётся из сырого `ckpts/shape_enc_next_dc_f16c32_fp16.safetensors`
# или из `external/shape_enc/model.safetensors` в каталоге canonical/GGUF-пакета.
# `--texture-resolution` 512|1024|1536 (по умолчанию 1024; 1536 использует conditioning/DiT 1024).
cargo run -p lux3d-cli --features cuda -- run trellis2 --model-path <raw-TRELLIS.2-4B-dir> --rmbg-model-path <raw-RMBG-2.0-dir> --source <condition-image> --texture-mesh <input-mesh.glb> --texture-resolution 1536 --output textured.glb

# Автоскачивание canonical packages из Hugging Face в пользовательский cache
cargo run -p lux3d-cli -- run pi3 --source <input-sequence> --output <output-file.ply>
```

**Форматы вывода:** mesh-команды (`triposr`, `triposg`, `trellis2`) выбирают writer по расширению: `.glb` (основной/по умолчанию), `.obj`, `.ply`, `.stl`. `.stl` бинарный и не хранит цвета вершин, поэтому цветной вывод `trellis2` в `.stl` теряет покраску; writer текстурированного GLB работает только с `.glb` (`--texture-mesh` требует `.glb` на выходе). Point-cloud команды (`pi3`, `pi3x`, `vggt`) — `.ply` (основной) или `.xyz`. HTTP mesh API по умолчанию отдаёт GLB, point-cloud API — PLY; формат задаётся `options.output_format` (`glb`/`obj`/`ply`/`stl` и `ply`/`xyz` соответственно). Текстурированный `.glb`-прогон `trellis2` встраивает запечённые PNG base-color + metallic-roughness с UV и нормалями (parity с upstream `to_glb`) и дополнительно пишет пару sidecar-файлов `<stem>.pbrvox.bin` + `<stem>.pbr.json` рядом с выводом: PBR-атрибуты вершин (6 каналов) на разреженной воксельной сетке не помещаются в форматы GLB/OBJ/PLY/STL, поэтому sidecar сохраняет их без потерь. `.obj`/`.ply`/`.stl` выводы `trellis2` текстуры не встраивают — остаётся меш с цветами вершин (`.ply`), только с позициями (`.obj`) или без цветов (`.stl`). HTTP-сервер маппит postprocess- и texture-export-поля TRELLIS.2 (`postprocess`, `decimation_target`, `fill_hole_perimeter`, `min_component_area`, `remesh`, `remesh_band`, `remesh_project`, `texture`, `texture_size`): `trellis2`-задачи с `.glb` на выходе пишут GLB со встроенными текстурами (плюс PBR-sidecar'ы в workspace задачи; sidecar'ы не раздаются через `/content`). Отдельный endpoint `POST /v1/meshes/texturings` текстурирует существующий меш и всегда пишет `trellis2-textured.glb` (без PBR-sidecar); `--texture-mesh` в CLI работает так же (и требует `.glb` на выходе). `--return-latent` и CLI-only поверхности TripoSG (`--surface-input`, `--mode scribble`, повторяемый `--source`, `--num-shapes`) не имеют HTTP-полей.

> **Заметки по VGGT:** `--source` принимает каталог кадров (`.png`/`.jpg`/`.jpeg`), видеофайл (`.mp4`/`.mov`/`.avi`/`.mkv`) или одно изображение (`.png`/`.jpg`/`.jpeg`/`.webp`/`.bmp`/`.tif`/`.tiff` — сцена из одного кадра). `--preprocess-mode` выбирает `crop` (upstream-дефолт: resize по ширине 518, центральный crop строк выше 518, затем white-pad до максимума в батче), `pad` (длинная сторона 518, центральный white-pad до квадрата 518x518, без потери пикселей — используйте, когда важна композиция) или `square` (официальный путь `demo_colmap.py`: black-pad до квадрата 1024x1024, вход модели — его bilinear-уменьшение до 518x518, а официальные координаты `x1/y1/x2/y2/width/height` управляют COLMAP-экспортом в исходном разрешении). `--vggt-dtype f32|f16|bf16` задаёт dtype загрузки сети (`f32` — референс; публичные выходы остаются f32; отдельный флаг от TripoSG `--dtype`) — F16/BF16 подключены, но автоматический parity-набор только F32/CUDA, а BF16 не проверен. `--batch-size N` режет список кадров на последовательные чанки по N кадров и гоняет каждый как отдельный B=1 проход: это ограничивает пик VRAM, но НЕ единый attention-проход по всем кадрам — межкадровое внимание видит только кадры своего чанка, поэтому результаты чанкинга отличаются от полного прохода, если сцена занимает несколько чанков (по умолчанию 1; 0 отклоняется; значение >= числа кадров — обычный одиночный проход). `--colmap-dir` пишет бинарную sparse-модель COLMAP (`cameras.bin`/`images.bin`/`points3D.bin`) плюс соседний `points.ply`. Дефолтный `--colmap-mode feedforward` повторяет feedforward-ветку upstream `demo_colmap.py`: одна PINHOLE-камера на изображение, источник точек `--colmap-point-source depth` (по умолчанию; гейт `depth_conf >= --colmap-conf-threshold`, по умолчанию 5.0) или `pointmap` (point-head `world_points`), лимит `--colmap-max-points` (по умолчанию 100000). Без `--colmap-seed` лимит оставляет детерминированную равномерную подвыборку; с seed — случайная выборка (распределённо эквивалентна numpy `randomly_limit_trues`, но не бит-идентична). Интринсики пересчитываются в исходное разрешение; для каталога сохраняются имена кадров, для видео/одного изображения пишется `image_N`; прогон печатает статистику экспорта. `--colmap-mode track` пишет track-based реконструкцию `batch_np_matrix_to_pycolmap` (SIMPLE_PINHOLE-камеры, много-кадровые треки, гейт `vis > --vis-thresh` по умолчанию 0.2, ошибка репроекции `< --max-reproj-error` по умолчанию 8.0, `--shared-camera` и порог `--colmap-min-inlier-per-frame` по умолчанию 64 на кадр); режим требует `--colmap-dir` и `--track-auto` и намеренно НЕ выполняет bundle adjustment — это COLMAP-ready вход для внешнего BA/gsplat. `--depth-points <path>` разворачивает depth-карты в world-space point cloud PLY/XYZ (`depth_conf > --depth-conf-threshold`, по умолчанию 1.0). `--track-queries '<json>'` трекает явные query-точки (`{"frame":0,"x":F,"y":F}` или массивы `[0,x,y]`; пиксельные координаты во входном разрешении модели, x/y в [0, 518); query-кадр должен быть 0 — поверните клип так, чтобы нужный кадр шёл первым) и пишет `{"coords": [[[x,y],...],...], "visibility": [[...],...], "confidence": [[...],...]|null}` в `--track-output` (по умолчанию `<source-stem>_tracks.json`), плюс `"query_frames"` для auto-режима, с числом итераций уточнения `--track-iters` (по умолчанию 4; 0 отклоняется). `--track-auto` заменяет явные запросы стадией A трекинга: DINOv2 ViT-B/14-with-registers CLS-сходство + FPS ранжируют `--query-frame-num` кадров (по умолчанию 8; кадр 0 добавляется первым, если его нет), FPS выбирает до `--max-query-pts` точек на query-кадр (по умолчанию 4096), точки проходят confidence-гейт как в `track_predict.py` (conf > 1.2, когда выживает больше 512); нужен safetensors DINOv2-reg в `models/dinov2_reg4/model.safetensors` или `LUX3D_DINOV2_REG_MODEL_PATH`. Stage A НЕ включает официальные ALIKED/SuperPoint keypoint extractors и стадии fine tracker B/C, поэтому автоматические треки грубее официального BA-входа. `--track-queries` и `--track-auto` взаимоисключающие, а `--batch-size` > 1 вместе с любым трекингом отклоняется (track-проход идёт по всем кадрам сразу). `--track-points-ply <path>` пишет по одной цветной вершине на запрос (world-точка кадра 0 в пикселе запроса). `--glb <scene.glb>` пишет demo-gradio GLB-сцену из `predictions_to_glb`: confidence-filtered цветные точки (отбрасывается `--glb-conf-percentile` процентов точек с наименьшей уверенностью, по умолчанию 50; 0 отключает percentile-гейт и оставляет все точки с `conf > 1e-5`) из `--glb-point-source depth` (по умолчанию) или `pointmap`, плюс конусы камер по кадрам с `--glb-show-cam`; сцена выравнивается по первой камере с официальной OpenGL-конвенцией ±y/z. `--dump-npz` сохраняет сырые тензоры инференса: исторические ключи (`world_points`, `world_points_conf`, `depth`, `depth_conf`, `extrinsics`, `intrinsics`, `pose_enc`, `pose_enc_list` `[K, S, 9]`, по записи на итерацию уточнения камеры) плюс официальные demo-ключи `images`, `extrinsic`, `intrinsic` и `world_points_from_depth`; работает вместе с track/COLMAP/depth/GLB-флагами.

> **Заметки по TRELLIS.2:** `--pipeline-type` принимает `512`, `1024`, `1024_cascade` (референсный дефолт) или `1536_cascade` (каскад с целевой сеткой 1536 и жёстким стопом по бюджету токенов на 1024; каскад 1024 всегда декодирует на 1024). Диспетчеризация вариантов и цикл бюджета токенов 1536→…→1024 покрыты тестами, но реального end-to-end прогона parity для `1536_cascade` на боевых весах в этом дереве нет. `--steps` задаёт число шагов всем трём сэмплерам; постадийные override'ы (`--ss-steps/--shape-steps/--tex-steps`, `--ss-guidance/--shape-guidance/--tex-guidance`, `--ss-rescale/--shape-rescale/--tex-rescale`, `--ss-interval/--shape-interval/--tex-interval` в формате `lo,hi`) заменяют дефолты соответствующей стадии (`resolve_stage_params`: `steps` стадии выигрывает у `--steps`). `--num-samples N` повторяет структуру дро (draw structure) upstream `run(num_samples)` (`trellis2_image_to_3d.py:208, 253`): ОДИН совместный тензор шума `[N, 8, R, R, R]` для sparse structure и ОДИН совместный тензор шума на каждую SLAT-стадию по объединённым координатам всех сэмплов (оба из `--seed`), один общий conditioning-проход и по-сэмпловые сэмплинг SLAT/декод/экспорт; выходные файлы и латенты получают суффиксы `_0`, `_1`, ... (`N = 1` сохраняет одиночный путь бит-в-бит). RNG Lux3D не бит-идентичен `torch.randn`, поэтому дро распределённо эквивалентны референсу, а не value-идентичны; `--return-latent <path>` пишет `shape_latent`/`tex_latent`/`coords`/`resolution`/`lr_shape_latent`/`meta` в `.npz` (поверхность upstream `return_latent`). Референсный препроцессинг воспроизведён полностью: LANCZOS-уменьшение длинной стороны, alpha-путь при непустом alpha-канале RGBA, иначе RMBG-2.0/BiRefNet строит маску (отсутствие весов RMBG — ошибка, а не opaque-fallback), затем bbox по alpha>0.8, квадратный crop с Pillow-округлением, premultiply и resize. Постобработка меша включена по умолчанию и повторяет shipped-точки входа `to_glb` (`example.py:43`, `app.py:503`): общая заливка дыр (`--fill-hole-perimeter`, по умолчанию 3e-2; 0 = выкл.), затем narrow-band dual-contouring remesh в каноническом фрейме `[-0.5, 0.5]^3` с `scale = (resolution + 3 * band) / resolution` (`--remesh-band`, по умолчанию 1 ячейка сетки; `--remesh-project`, по умолчанию 0 = выкл.) и один проход simplify до `--decimation-target` (по умолчанию 1000000; 0 = оставить remesh-топологию). `--no-remesh` возвращает прежнюю cleanup-цепочку (агрессивный simplify x3, ремонт дублированных/неманифолдных граней, удаление мелких компонент через `--min-component-area`, по умолчанию 1e-5; 0 = оставить все, унификация ориентации, финальный simplify) байт-в-байт; `--no-postprocess` отключает весь проход. Текстурированные GLB после remesh получают `doubleSided = false` (upstream `postprocess.py:303`), меши без remesh/pass-through сохраняют `true`. Remesh — clean-room порт narrow-band dual contouring, cleanup/decimation — clean-room порты на simplifier'е meshopt, а не cumesh, поэтому топология может отличаться от референса. При включённом `--texture` (по умолчанию) стадия экспорта текстурированного GLB повторяет upstream `to_glb` UV unwrap + PBR bake: `--texture-size <n>` задаёт размер запекаемой текстуры (по умолчанию 2048; CLI принимает 64..=8192, upstream-референс допускает только 1024..4096 шагом 1024, а ядро дополнительно требует кратность 8), и выходной `.glb` встраивает PNG base-color + metallic-roughness с TEXCOORD UV, area-weighted нормалями и upstream-конверсией осей glTF `(x, y, z) -> (x, z, -y)`. `--no-texture-export` пропускает всю стадию, `--no-uv-unwrap` — только xatlas unwrap (оба варианта откатываются к мешу с цветами вершин; `.obj`/`.ply`/`.stl` текстуры не встраивают). xatlas unwrap + bake идут на CPU — измерено ~42 с unwrap + ~9 с bake для сферы в 80k треугольников при 2048 (release-сборка, это дерево), стоимость растёт как `--texture-size` в квадрате. Непокрытые тексели атласа заполняются TELEA-аппроксимацией (итеративный onion-peel ближайшего валидного текселя), а не референсным `cv2.inpaint`; прогон печатает статистику `[bake]` (charts, atlases, atlas size, texture size, valid/inpainted texels). Меш остаётся поверхностью O-Voxel dual-grid, а пара sidecar-файлов `<stem>.pbrvox.bin` + `<stem>.pbr.json` — lossless-записью PBR по вокселям. Стадии весов загружаются последовательно (~7.8 GiB измеренный пик на `1024_cascade` при резидентных текстурных стадиях); инференс на CPU работает, но медленный. `--no-low-vram` включает upstream-режим `low_vram=False` и держит все загруженные стадии резидентными на устройстве (один набор загрузки, без выгрузки между стадиями); выходы идентичны дефолтному режиму при том же seed, пик VRAM выше, а не поместившийся резидентный набор завершается ошибкой с именем флага, стадии и лимита памяти устройства вместо тихого fallback. Автоскачивание из `oxide-lab/TRELLIS.2-4B` зарезервировано, но репозиторий ещё не опубликован, поэтому передавайте явный `--model-path` (canonical-пакет или сырой каталог checkpoint'ов).

> **Режим текстурирования TRELLIS.2:** `--texture-mesh <file>` текстурирует существующий меш `.glb`/`.obj`/`.ply`/`.stl` по condition-изображению из `--source` (upstream `Trellis2TexturingPipeline`) вместо image-to-3D. Меш препроцессится в `[-0.5, 0.5]^3` (центрирование/масштаб до 0.99999 + y/z swap; `--no-preprocess-mesh` требует, чтобы вход уже был в этом фрейме), вокселизуется в flexible dual grid при `--texture-resolution` (512, 1024 или 1536, по умолчанию 1024; 1536 вокселизует/кодирует на полной сетке 1536, но использует conditioning 1024 и `tex_dit_1024`), кодирует сырой shape SLAT через `shape_enc_next_dc_f16c32_fp16`, сэмплирует `tex_dit_512`/`tex_dit_1024` с override'ами `--tex-*`, декодирует с синтезированными occupancy-guide'ами и запекает UV-атлас. Core API `texture_mesh` умеет запекать прямо по переданным per-corner UV (референсная ветка UV-reuse в `postprocess_mesh`, без запуска xatlas), но CLI `--texture-mesh` и HTTP-задача текстурирования загружают меш по пути и не передают его входные UV, поэтому эти поверхности всегда переразворачивают атлас; cleanup/decimation/remesh не запускаются (UV-швы дублируют вершины, zero-area грани отбрасываются). `--texture-mesh` требует `.glb` на выходе, отклоняет `--pipeline-type`/`--max-num-tokens`/`--num-samples`/`--return-latent`/`--no-postprocess`/`--decimation-target`/`--no-texture-export`/`--no-uv-unwrap`/`--texture false` и пишет только текстурированный GLB (без PBR-sidecar). Применяются только `--tex-*`, `--seed`, `--texture-size`, `--texture-resolution` и `--no-preprocess-image`/`--no-preprocess-mesh`; `--steps`/`--ss-*`/`--shape-*` не отклоняются, но в этом режиме не действуют. Shape encoder берётся из `<model-dir>/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` (сырой layout checkpoint'а), `<model-dir>/external/shape_enc/model.safetensors` (sidecar рядом с merged canonical/GGUF-пакетом) или из ключей `shape_enc.*` в merged safetensors-пакете; работают и сырой каталог `TRELLIS.2-4B`, и layout canonical `model.gguf` + внешние encoder/`external/dinov3`. `--no-preprocess-image` пропускает RMBG/crop и подаёт квадратное исходное изображение напрямую (RMBG-модель при этом не требуется).

> **Экспериментальные бэкенды (Vulkan / WGPU):** выбор устройства через `--device`:
>
> ```powershell
> cargo run -p lux3d-cli --features vulkan -- run triposr --device vulkan --source <input-image> --output <output-file.glb>
> cargo run -p lux3d-cli --features wgpu -- run pi3 --device wgpu --source <input-sequence> --output <output-file.ply>
> cargo run -p lux3d-cli --features vulkan -- run triposg --device vulkan --source <input-image> --steps 2 --num-tokens 1024 --flash-octree-depth 5 --output <preview.glb>
> ```
>
> Эти пути экспериментальные и не проверяются в CI. Инференс может работать нестабильно, выдавать некорректный результат или вовсе не запускаться в зависимости от GPU, драйвера и ОС. Для продакшена используй CUDA.

### Команды o_voxel-объёмов

Группа `ovox` конвертирует текстурные меши в разреженный PBR O-Voxel-объём, который использует TRELLIS.2 (примеры upstream `o-voxel`), и обратно. Это CLI-only команды (HTTP-маршрута нет); каждая печатает JSON-отчёт со статистикой.

```powershell
# Текстурный меш -> разреженный PBR .vxz (по умолчанию 512^3; .npz/.ply тоже принимаются на выходе)
cargo run -p lux3d-cli -- ovox mesh2ovox <input.glb> <output.vxz>

# .vxz/.npz/.ply -> восстановленный .ply с base-color цветами вершин
cargo run -p lux3d-cli -- ovox ovox2mesh <input.vxz> <output.ply> --grid-size 512 --aabb -0.5,-0.5,-0.5,0.5,0.5,0.5

# .vxz/.npz/.ply -> очищенный, UV-unwrapped, текстурированный .glb (дефолты официального
# примера: decimation 100000, texture 2048)
cargo run -p lux3d-cli -- ovox ovox2glb <input.vxz> <output.glb> --grid-size 512 --texture-size 2048
```

Флаги `mesh2ovox`: `--grid-size N[,N,N]` (по умолчанию 512), `--voxel-size S[,S,S]`, `--aabb MINX,...,MAXZ`, `--no-normalize` (пропустить официальную нормализацию в unit-cube: центр + `0.99999/max_extent`), `--mip-level-offset` (0.0), `--face-weight` (1.0), `--boundary-weight` (0.2), `--regularization-weight` (1e-2) и `--compression none|deflate` (по умолчанию `deflate`). `.vxz` пишется только с `none` или `deflate`: LZMA-файлы читаются (официальный writer по умолчанию использует LZMA), но запись LZMA не поддерживается, а zstd-сжатые файлы отклоняются с явной ошибкой, потому что в этой сборке нет zstd-декодера. `ovox2mesh`/`ovox2glb` требуют `--grid-size` и `--aabb`, потому что `.vxz` не хранит размер сетки и границы (дефолты 512 и `[-0.5, 0.5]^3`). `ovox2glb` дополнительно принимает `--decimation-target` (100000), `--texture-size` (2048), `--fill-hole-perimeter` (3e-2) и `--min-component-area` (1e-5).

### Запуск HTTP-Сервера

```powershell
cargo run -p lux3d-server --features cuda -- --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

TRELLIS.2 RGB-запросам нужен отдельный пакет RMBG-2.0. Передайте его через
`--trellis2-rmbg-model-path <dir>` / `LUX3D_TRELLIS2_RMBG_MODEL_PATH` или положите
`rmbg-2.0` (либо сырой `RMBG-2.0`) рядом с явным `--model-path`. Для TripoSG
`--rmbg-model-path` / `LUX3D_RMBG_MODEL_PATH` остаётся отдельным пакетом RMBG-1.4;
семейства намеренно не делят веса.

Флаги серверного бинарника (`crates/lux3d-server/src/main.rs`): `--host` / `--port`
(по умолчанию `127.0.0.1:8080`), `--model-path` (`LUX3D_MODEL_PATH`), `--cache-dir`
(`LUX3D_CACHE_DIR`), `--rmbg-model-path` (`LUX3D_RMBG_MODEL_PATH`),
`--trellis2-rmbg-model-path` (`LUX3D_TRELLIS2_RMBG_MODEL_PATH`), `--cuda-device` (0),
`--upload-dir`, `--cors-origin` (повторяемый), `--job-ttl-secs` (86400),
`--cleanup-interval-secs` (300), `--access-log` (true), `--access-log-health` (false),
`--access-log-format text|json`, `--request-id-header` (true) и `--metrics` (true).
Бинарник всегда собирается с платформенным дефолтным устройством (CUDA вне macOS,
Metal на macOS metal-сборках); Vulkan/WGPU (и явный выбор Metal) доступны только через
`with_device_backend` встраиваемого router builder'а.

Готовые релизы (`.github/workflows/release.yml`): Windows + Ubuntu × cpu/cuda/vulkan.
Имена по образцу llama.cpp (`lux3d-<tag>-bin-…`). CUDA-сборки не содержат библиотек NVIDIA:
скачайте соответствующий архив `cudart-lux3d-<tag>-…` и распакуйте рядом с бинарниками
(в той же директории, что и `.exe`; `$ORIGIN` на Linux). Хосту нужен только драйвер NVIDIA.

Generation-endpoint'ы принимают `multipart/form-data` (или JSON) и возвращают `202 Accepted` с async job object:

| Endpoint | Models | Output |
|----------|--------|--------|
| `POST /v1/point-clouds/generations` | `pi3`, `pi3x`, `vggt` | PLY |
| `POST /v1/meshes/generations` | `triposr`, `triposg`, `trellis2` | GLB |
| `POST /v1/meshes/texturings` | `trellis2` | GLB (встроенные текстуры) |

`POST /v1/meshes/texturings` текстурирует существующий меш по одному condition-изображению (только TRELLIS.2). Multipart-поля: `source` (condition-изображение), `mesh` (`.glb`/`.obj`/`.ply`/`.stl`), опциональные `model` (по умолчанию `trellis2`) и `options`. JSON-поля: `source_url`/`source_file_id`/`source_base64`/`source_filename`, `mesh_url`/`mesh_file_id`/`mesh_base64`/`mesh_filename`, опциональные `model`, `options`, `webhook_url`; нужен хотя бы один вариант из `source*` и один из `mesh*`. `options` принимает `output_format` (только `glb`; по умолчанию `glb`), `seed` (42), `texture_resolution` (512/1024/1536; по умолчанию 1024), `texture_size` (2048), `tex_steps`, `tex_guidance_strength`, `tex_guidance_rescale`, `tex_guidance_interval` (`[lo, hi]`), `tex_rescale_t`, `preprocess_image` и `preprocess_mesh` (оба по умолчанию true). Задача пишет ровно один `trellis2-textured.glb` (запечённый PBR-атлас встроен, PBR-sidecar нет).

Остальные маршруты: `GET /` и `GET /health`; `GET /metrics` (Prometheus; включён по умолчанию с `--metrics`, при отключении маршрут возвращает 503); `GET`/`POST /v1/files`, `GET`/`DELETE /v1/files/{id}` и `GET /v1/files/{id}/content` (загрузите один раз и передавайте `source_file_id`/`mesh_file_id`); `GET /v1/generations`, `GET`/`DELETE /v1/generations/{id}`, `POST /v1/generations/{id}/cancel` и `GET /v1/generations/{id}/stream` (SSE-обновления статуса); `POST /v1/models/reload`, `POST /v1/models/unload` и `POST /v1/models/status`; `GET /v1/system/info` и `POST /v1/system/doctor`. Swagger UI отдаётся на `/docs`, схема — на `/api-doc/openapi.json` (feature `swagger-ui` включён по умолчанию).

Общие поля:

- `model` — обязателен для point clouds (`pi3`, `pi3x` или `vggt`), опционален для meshes (по умолчанию `triposr`; укажите `triposg` для TripoSG или `trellis2` для TRELLIS.2)
- `source` — обязательная загрузка (`.zip`-архив кадров, видео (`.mp4`/`.mov`/`.avi`/`.mkv`), изображение (`.png`/`.jpg`/`.jpeg`/`.webp`) или уже распакованный каталог кадров)
- `conditions` — опциональный файл условий Pi3X
- `options` — опциональная JSON-строка (или JSON-объект в JSON-запросе) с параметрами семейства.
  - TripoSG: `steps`, `guidance_scale`, `seed`, `num_tokens`, `use_flash_decoder`, `flash_octree_depth`, `dense_octree_depth`, `hierarchical_octree_depth`, `query_chunk_size`, `timesteps`, `bounds` (`[min_x,min_y,min_z,max_x,max_y,max_z]`), `faces`, `output_format` (`glb` по умолчанию, `obj`, `ply`, `stl`). CLI-only поверхности TripoSG (`--surface-input`, `--mode scribble`, повторяемый `--source`, `--num-shapes`) не имеют полей запроса.
  - TRELLIS.2: `steps` (все три Euler-сэмплера), `seed`, `pipeline_type` (`512`/`1024`/`1024_cascade`/`1536_cascade`), `max_num_tokens`, `texture`, `output_format` (`glb` по умолчанию, `obj`, `ply`), `num_samples` (значения > 1 используют ту же joint-draw структуру, что и CLI, и пишут выходы `_0`, `_1`, ...; `/content` отдаёт sample 0), `texture_size`, postprocess-поля (`postprocess`, `decimation_target`, `fill_hole_perimeter`, `min_component_area`, `remesh`, `remesh_band`, `remesh_project`) и постадийные override'ы `ss_steps`/`ss_guidance_strength`/`ss_guidance_rescale`/`ss_guidance_interval`/`ss_rescale_t` и соответствующие `shape_*`/`tex_*`. `--return-latent` остаётся CLI-only; текстурирование существующего меша — через `POST /v1/meshes/texturings`.
  - Point clouds: `output_format` (`ply` по умолчанию, `xyz`), плюс `interval`, `vo`, `chunk_size`, `overlap`, `conf_threshold`, `inject_condition`. Поверхность VGGT `--preprocess-mode`/`--batch-size`/`--colmap-dir`/`--depth-points`/`--track-queries`/`--track-auto`/`--glb` — CLI-only.

Workflow:

```powershell
# 1. Создать задачу генерации меша
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=triposr" `
  -F "source=@input.png" `
  -F "options={\"mc_resolution\":256,\"mc_threshold\":25.0}"

# Задача TripoSG (готовый файл скачивается как triposg.glb)
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=triposg" `
  -F "source=@input.png" `
  -F "options={\"steps\":50,\"use_flash_decoder\":true,\"flash_octree_depth\":9}"

# Задача TRELLIS.2 (готовый файл скачивается как trellis2.glb)
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=trellis2" `
  -F "source=@input.png" `
  -F "options={\"steps\":12,\"seed\":42,\"pipeline_type\":\"1024_cascade\",\"texture\":true}"

# Текстурирование существующего меша (готовый файл скачивается как trellis2-textured.glb):
# source = condition-изображение, mesh = меш .glb/.obj/.ply/.stl
curl -X POST http://127.0.0.1:8080/v1/meshes/texturings `
  -F "model=trellis2" `
  -F "source=@condition.png" `
  -F "mesh=@input.glb" `
  -F "options={\"texture_resolution\":1024,\"texture_size\":2048,\"tex_steps\":12}"

# 2. Опрос статуса задачи
curl http://127.0.0.1:8080/v1/generations/gen_...

# 3. Скачивание готового ассета
curl http://127.0.0.1:8080/v1/generations/gen_.../content --output mesh.glb
```

Встраивание router'а в другое Axum-приложение:

```rust
use lux3d_server_core::{
    lux3d_for_server_builder::Lux3dForServerBuilder,
    lux3d_server_router_builder::Lux3dServerRouterBuilder,
};

let lux3d = Lux3dForServerBuilder::new().build()?;
let routes = Lux3dServerRouterBuilder::new()
    .with_lux3d(lux3d)
    .build()
    .await?;
let app = axum::Router::new().nest("/api/lux3d", routes);
```

## Производительность

Замеры инференса (`[stage] infer`: препроцессинг + нейросетевой проход, одна итерация на процесс, без учёта загрузки весов) на NVIDIA GeForce RTX 3060 12 GiB, Windows (WDDM), форк candle `wgpu/vulkan` rev `efb506c9`, candle-3d (2026-09-06):

| модель | CUDA | Vulkan | WGPU | Vulkan/CUDA | WGPU/CUDA |
|---|---:|---:|---:|---:|---:|
| pi3 (5 кадров, 518×518) | **4.82 с** | 7.83 с | 13.96 с | 1.62× | 2.90× |
| pi3x (6 кадров, 518×518) | **7.00 с** | 11.10 с | 31.84 с | 1.59× | 4.55× |
| triposr (одно изображение) | **1.23 с** | 3.36 с | 4.89 с | 2.73× | 3.98× |

VGGT-1B добавлен позже и измерен отдельно (форк rev `f94d26eb`, 2026-11-11): 3 кадра 518×518,
camera + DPT depth + DPT point heads. RTX 3060 12 GiB: CUDA **6.76 с**, Vulkan 10.96 с (1.62×),
WGPU 19.34 с, CPU 107.66 с. Пик CUDA VRAM 6.52 GiB на 3 кадрах (Vulkan 6.1–7.4 GiB, WGPU
6.8 GiB по повторным прогонам); на NVIDIA L4 23 GiB — 7.98 GiB на 3 кадрах
(4.19 с infer) и 8.30 GiB на 12 кадрах (26.21 с infer).

TripoSG измерен по стадиям на RTX 3060 12 GiB (Windows/WDDM, release build, одно изображение
`test-assets/mesh-input.png`):

| Стадия | CUDA | Vulkan | WGPU |
|---|---:|---:|---:|
| load | 7.12 с | 6.68 с | 6.60 с |
| infer (preprocess + DINO + denoise) | 6.84 с | 8.56 с | 13.13 с |
| flash SDF grid + DiffDMC + trimesh processing | 0.73 с | 0.26 с | 0.45 с |
| export | 0.00 с | 0.00 с | 0.01 с |
| пик VRAM | 8.29 GiB | 8.36 GiB | 8.65 GiB |

Все три колонки — одна команда (`--steps 2 --num-tokens 1024 --flash-octree-depth 5`).
TripoSG держит все три подсети (DINOv2, DiT, VAE) в памяти — 7.4 GiB весов, — поэтому память
определяется чекпоинтом. После исправления дефолтов памяти (Vulkan: transient pool 0 / inflight
grace 2; WGPU: блоки аллокатора 2 MiB) Vulkan укладывается в 0.1 GiB от CUDA, WGPU — в ~0.4 GiB
(бывают колебания ±0.5 GiB из-за асинхронного staging).
12 GiB хватает для preview с 2048 токенами (Vulkan: load 6.56 с, infer 12.86 с, пик 9.13 GiB),
но для эталонного прогона на 50 шагов / depth 9 по-прежнему рекомендуется 16 GiB.

Тёплый criterion (10 сэмплов, установившийся режим — протокол candle-bench, одно сквозное окно измерений): vulkan 3.14 с / 4.90 с на итерацию (1.16×/1.20× от CUDA 2.72 с / 4.09 с), wgpu 8.21 с / 24.0 с (3.0×/5.9×). Самый крупный одиночный рычаг — слитый кернел LayerNorm+RoPE (`rope_layernorm.comp`/`.wgsl` в форке): один диспатч на q/k вместо медленного пути LayerNorm, обе окружающие копии и цепочка rope из 8 опов на каждом rope-вызове — wgpu pi3 в тёплом цикле 54.9 с → 8.21 с.

Корректность: выходы всех комбинаций бэкенд × модель сверены с эталонными мешами CUDA (дельты bbox/центра/среднего в пределах 1% от max extent, число вершин/граней в пределах 0.5% — все PASS); слитый кернел дополнительно покрыт юнит-тестом с точным совпадением (разница 0.0) против составной формы на обоих бэкендах.

Поведение памяти при повторном инференсе (цикл из 10 итераций в одном процессе):

- **CUDA** — потребление плоское; буфер attention scores декодера ограничен 768 MiB чанкингом по query-оси (`LUX3D_MAX_SDPA_SCORES_BYTES`) — убран ~4 GiB транзиент, который раньше фрагментировал пул CUDA и замедлял 12-гигабайтные карты ~в 100 раз.
- **Vulkan** — плоско; удержание памяти ограничено inflight-бюджетом (256 MiB × grace-полоса 2), транзиенты по умолчанию не пулятся (`CANDLE_VK_INFLIGHT_GRACE`, `CANDLE_VK_POOL_MAX_BYTES=0`); пул можно включить для latency-first прогонов. Капы батчей настроены по flush-reason профилированию: transfer-байты 512 MiB (`CANDLE_VK_MAX_BATCH_TRANSFER_BYTES`), descriptor sets 8× от диспатчей — закрытие батча на каждой большой копии активаций стоило ~4-9 ms WDDM fence-signal латентности на каждый сабмит.
- **WGPU** — 10-итерационный бенчмарк pi3x проходит с пиком VRAM 10.4 GiB и нулём ошибок (раньше OOM на ~3-й итерации); free pool / recycle backlog / in-flight удержание ограничены по байтам (`CANDLE_WGPU_POOL_MAX_BYTES`, `CANDLE_WGPU_INFLIGHT_MAX_BYTES`), а блоки аллокатора 2 MiB убирают большую часть слака чекпоинта (`CANDLE_WGPU_BLOCK_BYTES`).

Известные разрывы: оставшийся тёплый разрыв Vulkan/CUDA (1.16–1.20×) — структурный WDDM-налог на сабмит (~4-9 ms на сабмит при строго зависимой цепи операций) — свойство платформы, кодом больше не достижимо; Linux без WDDM закрыл бы его. WGPU отстаёт на 3.0–5.9× в тёплом цикле (было 13.9–19.5×) — качество WGSL GEMM плюс те же межpass-зазоры WDDM. Criterion-харнесс для бенчмарков целых моделей — в `crates/lux3d-core/benches/` (одна пара (бэкенд, модель) на процесс: `LUX3D_BENCH_DEVICE=… LUX3D_BENCH_MODEL=… cargo bench -p lux3d-core --bench bench_main --features vulkan,wgpu`).

## Системные Требования

- Rust 1.85 или новее
- Cargo с поддержкой `edition = "2024"`
- Python 3.x для baseline tooling и нормализации
- NVIDIA GPU с CUDA для проверенных инференс-запусков
- **Бэкенды Vulkan и WGPU** (через форк [FerrisMind/candle](https://github.com/FerrisMind/candle) `wgpu/vulkan`): корректность и стабильность памяти проверены на RTX 3060 (см. [Производительность](#производительность)); медленнее CUDA, WGPU — существенно. Пик TripoSG на обоих бэкендах в пределах ~0.4 GiB от CUDA, так что набор весов 7.9 GB оставляет ~3 GiB запаса на карте 12 GiB.
- Бэкенд Metal на macOS в этом репозитории не тестировался, но теоретически может поддерживаться.
- Директории canonical model package, переданные через `--model-path`, либо автоматически скачанные из Hugging Face в пользовательский cache
- TripoSG требует примерно 8.3–8.7 GiB на preview с 1024 токенами (9.1 GiB с 2048 токенами) на всех трёх бэкендах; для эталонного прогона на 50 шагов рекомендуется 16 GiB+.
- TRELLIS.2 загружает stage-checkpoint'ы последовательно и достигает ~7.8 GiB пика при резидентных текстурных стадиях на прогоне `1024_cascade` (прогоны без текстур легче); карта 12 GiB вмещает его с запасом ~4 GiB, инференс на CPU работает, но медленный. `--no-low-vram` держит все стадии резидентными и поэтому требует больше пиковой VRAM (при нехватке — явная ошибка с лимитом памяти, без тихого fallback). Сервер загружает только canonical-пакет; сырой каталог checkpoint'ов `microsoft/TRELLIS.2-4B` — loader только для CLI.

## Лицензия

Код и документация этого репозитория лицензированы по Apache 2.0. См. [LICENSE](https://github.com/oxide-lab/Lux3d/blob/main/LICENSE).

Upstream-модельные артефакты и canonical weights сохраняют свои исходные лицензии и ограничения на использование. Перед распространением проверяй условия для конкретного семейства модели.

**VGGT-1B:** оригинальный код и checkpoint `facebook/VGGT-1B` — CC-BY-NC-4.0 (non-commercial). Для коммерческого использования нужен отдельно распространяемый `facebook/VGGT-1B-Commercial`. Canonical package для `--model-path`/автоскачивания публикуется как `oxide-lab/VGGT-1B`; сырой upstream checkpoint остаётся в `facebook/VGGT-1B`. Нативный track head поддерживает явные query-точки и автоматический выбор запросов stage A (DINOv2-reg ranking кадров + FPS); официальные стадии fine tracker B/C (ALIKED/SuperPoint, LightGlue-подобное уточнение) не включены (см. заметки по VGGT в разделе запуска инференса).

**TripoSG:** upstream-код и веса — MIT. Сырой checkpoint лежит в `VAST-AI/TripoSG`; canonical package для `--model-path`/автоскачивания публикуется как `oxide-lab/TripoSG`. Upstream `load_image` использует BriaRMBG 1.4, когда нет валидной alpha-маски; Lux3D реализует этот fallback полностью. Веса RMBG не входят в MIT-пакет TripoSG: они нормализованы в отдельный canonical package `rmbg-1.4` (лицензия BRIA, source-available/non-commercial; для коммерческого использования нужна лицензия BRIA) и подключаются через `--rmbg-model-path` или соседнюю папку `rmbg-1.4`. Upstream pipeline по умолчанию использует flash decoder (`use_flash_decoder=True`, `flash_octree_depth=9`); Lux3D реализует тот же flash SDF-планировщик и clean-room Rust DiffDMC. Код и таблицы Diso (CC BY-NC 4.0) не копируются, поэтому на редких ambiguous-ячейках топология может отличаться; non-flash путь остаётся доступен через `--flash-decoder false`. Также реализован полный VAE encode API (FPS-сэмплирование, `vae.quant`, распределение mean/logvar/std), хотя image-to-3D pipeline вызывает только `vae.decode`.

**TRELLIS.2:** upstream-код (`microsoft/TRELLIS.2`) и model card `microsoft/TRELLIS.2-4B` лицензированы под MIT; референс зеркалируется в `candle-3d-refs/TRELLIS.2`. Получите веса в upstream-репозитории и нормализуйте командой `weights normalize ... trellis2` перед запуском. Встроенный DINOv3 ViT-L/16 — отдельная gated-модель Meta со своей лицензией; проверьте её условия независимо, доступ к DINOv3 может требоваться отдельно от TRELLIS.2. Референсное удаление фона RMBG-2.0 (BiRefNet/Swin-L) реализовано и используется для RGB-входа; его веса не входят в MIT-пакет TRELLIS.2. Нормализуйте их под auxiliary-семейством `rmbg2` и передавайте `--rmbg-model-path` (либо положите `rmbg-2.0` рядом с `--model-path`). RMBG-2.0 лицензирован CC-BY-NC-4.0 (non-commercial); коммерческое использование — по условиям BRIA.
