<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
</p>

---

<p align="center">
  <b>Workspace Rust para inferência 3D clean-room, inspeção de contratos e canonização de pesos.</b><br>
  Runtime CUDA-first para Pi3, Pi3X, TripoSR, VGGT, TripoSG e TRELLIS.2.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D / candle-3d</h1>

## Demo

https://github.com/user-attachments/assets/0a9f766f-68d3-482a-9c94-c55d0c355808

## O que é isto?

Lux3D / candle-3d é um workspace Rust 2024 que executa modelos de reconstrução e
geração 3D em [Candle](https://github.com/FerrisMind/candle) com runtime
CUDA-first e fallbacks Vulkan/WGPU. Ele oferece:

- um único CLI para inspeção de contratos, normalização de pesos, quantização
  GGUF, inferência e conversão O-Voxel;
- pacotes de pesos canônicos reproduzíveis com manifest e checksums;
- portes clean-room do pré-processamento, amostragem e exportação de malhas de
  referência;
- uma API HTTP assíncrona no estilo OpenAI para geração de point clouds e malhas.

Cada família de modelos mantém a licença do upstream. Revise os termos por
família em [docs/models.md](docs/models.md) antes de redistribuir pesos.

## Modelos suportados

| Família | Tarefa | Saída | Upstream |
|---------|--------|-------|----------|
| [`pi3`](docs/models.md#pi3-and-pi3x) | Point cloud multi-view | PLY / XYZ | [yyfz/Pi3](https://github.com/yyfz/Pi3) |
| [`pi3x`](docs/models.md#pi3-and-pi3x) | Point cloud multi-view + VO | PLY / XYZ | [yyfz/Pi3](https://github.com/yyfz/Pi3) |
| [`triposr`](docs/models.md#triposr) | Malha a partir de uma imagem | GLB / OBJ / PLY / STL | [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) |
| [`vggt`](docs/vggt.md) | Geometria multi-view + COLMAP | PLY / XYZ / COLMAP / GLB | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| [`triposg`](docs/models.md#triposg) | Image-to-mesh, scribble, surface encode | GLB / OBJ / PLY / STL | [VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) |
| [`trellis2`](docs/trellis2.md) | Image-to-3D, texturização, O-Voxel | GLB / OBJ / PLY / STL | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |

Os pacotes canônicos são publicados em `oxide-lab/*` no Hugging Face e baixados
automaticamente para o cache do usuário. `oxide-lab/TRELLIS.2-4B` está reservado
mas ainda não foi publicado, então TRELLIS.2 exige um `--model-path` explícito.

## Início rápido

### Requisitos

- Rust 1.85+ com suporte a `edition = "2024"`
- Python 3.x para scripts de normalização e avaliação
- GPU NVIDIA para inferência CUDA verificada; Vulkan e WGPU são suportados, mas
  experimentais (veja [experimental backends](docs/inference.md#experimental-backends-vulkan-and-wgpu))

### Build

```powershell
cargo build --release -p lux3d-cli --features cuda
```

Builds padrão são CPU-only. Backends GPU são features do Cargo: `cuda`, `vulkan`,
`wgpu`, `metal`.

### Inspecionar e preparar pesos

```powershell
# Mostrar o contrato esperado por uma família
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposg

# Normalizar um checkpoint bruto no pacote canônico
cargo run -p lux3d-cli -- weights normalize `
  --repo-root <runtime-root> `
  --raw-model-dir <raw-triposg-dir> `
  --output-dir <canonical-triposg-dir> triposg
```

Guia completo de pesos: [pacotes canônicos e quantização](docs/weights.md).

### Gerar

```powershell
# Imagem -> malha (baixa automaticamente o pacote canônico TripoSG)
cargo run -p lux3d-cli --features cuda -- run triposg `
  --source .\input.png --output .\out.glb

# Imagem -> GLB texturizado (pacote TRELLIS.2 explícito; RMBG-2.0 é separado)
cargo run -p lux3d-cli --features cuda -- run trellis2 `
  --model-path <canonical-trellis2-dir> `
  --rmbg-model-path <canonical-rmbg2-dir> `
  --source .\input.png --output .\out.glb
```

Todos os comandos por família, flags e regras de formato de saída:
[docs/inference.md](docs/inference.md).

### Servidor HTTP

```powershell
cargo run -p lux3d-server --features cuda -- --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
```

Endpoints, campos de requisição e política SSRF: [docs/server.md](docs/server.md).

## Documentação

| Documento | Conteúdo |
|-----------|----------|
| [docs/models.md](docs/models.md) | Famílias, resumos de pipeline, upstream e licenças |
| [docs/inference.md](docs/inference.md) | Comandos `run` de todas as famílias, formatos de saída, backends experimentais, O-Voxel |
| [docs/vggt.md](docs/vggt.md) | VGGT: pré-processamento, exportação COLMAP, tracking, cena GLB, dumps de tensores |
| [docs/trellis2.md](docs/trellis2.md) | TRELLIS.2: variantes de pipeline, amostragem, pós-processamento, bake de textura, texturização de malhas |
| [docs/weights.md](docs/weights.md) | Pacotes canônicos, normalização, quantização GGUF, fidelidade medida |
| [docs/server.md](docs/server.md) | API HTTP, flags do servidor, fluxo de jobs, embed do router |
| [docs/performance.md](docs/performance.md) | Benchmarks por backend, warm-loop, comportamento de memória |
| [docs/features.md](docs/features.md) | Inventário de features e user stories |
| [docs/e2e.md](docs/e2e.md) | Matriz de cenários end-to-end e execução |
| [docs/e2e-results.md](docs/e2e-results.md) | Resultados e2e mais recentes com pesos reais |
| [docs/performance-notes.md](docs/performance-notes.md) | Log de investigação de performance e alavancas restantes |

## Desempenho

Tempo de inferência medido (`[stage] infer`: pré-processamento + passe neural,
sem carregar pesos) em uma NVIDIA GeForce RTX 3060 12 GiB, Windows (WDDM):

| Modelo | CUDA | Vulkan | WGPU | Vulkan/CUDA | WGPU/CUDA |
|---|---:|---:|---:|---:|---:|
| pi3 (5 frames, 518×518) | **4.82 s** | 7.83 s | 13.96 s | 1.62× | 2.90× |
| pi3x (6 frames, 518×518) | **7.00 s** | 11.10 s | 31.84 s | 1.59× | 4.55× |
| triposr (imagem única) | **1.23 s** | 3.36 s | 4.89 s | 2.73× | 3.98× |
| vggt (3 frames, 518×518) | **6.76 s** | 10.96 s | 19.34 s | 1.62× | 2.86× |

TripoSG preview (`--steps 2 --num-tokens 1024 --flash-octree-depth 5`): infer
6.84 s CUDA / 8.56 s Vulkan / 13.13 s WGPU com pico de ~8.3–8.7 GiB. TRELLIS.2
`1024_cascade` com textura roda em 438 s na CUDA com pico de 7.8 GiB.

Tabelas completas, VRAM, warm-loop e lacunas conhecidas:
[docs/performance.md](docs/performance.md).

## Estrutura do repositório

| Caminho | Propósito |
|---------|-----------|
| `crates/lux3d-core` | Runtime central, contratos, geometria/exportação e validação de pesos |
| `crates/lux3d-cli` | CLI para inspeção, normalização e execução de inferência |
| `crates/lux3d-server-core` | Router Axum embutível e API HTTP |
| `crates/lux3d-server` | Binário do servidor HTTP |
| `docs/` | Guias e referência: models, inference, VGGT, TRELLIS.2, weights, server, performance, features, e2e |
| `scripts/` | Avaliação de quantização, comparação de geometria, conversão de keypoints |
| [`tools/python_baseline`](tools/python_baseline/README.md) | Ferramentas Python de paridade e normalização de pesos |
| `.github/assets` | Assets de documentação |

## Requisitos do sistema

- Rust 1.85 ou mais recente; Cargo com suporte a `edition = "2024"`.
- Python 3.x para ferramentas baseline e normalização.
- GPU NVIDIA compatível com CUDA para inferência verificada. A placa de teste de
  referência é a RTX 3060 12 GiB.
- Vulkan e WGPU via fork [FerrisMind/candle](https://github.com/FerrisMind/candle);
  verificados para correção e estabilidade de memória, mas mais lentos que CUDA.
- Metal é build-checked na CI; a paridade de inferência não é validada.
- 16 GiB+ de VRAM são recomendados para a execução de referência do TripoSG com
  50 steps. TRELLIS.2 em `1024_cascade` com textura atinge ~7.8 GiB de pico.

## Licença

O código e a documentação deste repositório são Apache-2.0. Pesos upstream e
artefatos canonizados mantêm suas licenças e restrições originais; revise
[docs/models.md](docs/models.md#licensing) antes de redistribuir.

## Agradecimentos

Construído sobre [FerrisMind/candle](https://github.com/FerrisMind/candle). As
implementações seguem [Pi3](https://github.com/yyfz/Pi3),
[TripoSR](https://github.com/VAST-AI-Research/TripoSR),
[VGGT](https://github.com/facebookresearch/vggt),
[TripoSG](https://github.com/VAST-AI-Research/TripoSG) e
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2).
