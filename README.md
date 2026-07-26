# Lunaris Dense IRL-2

Implementação de pesquisa de um modelo de linguagem causal, decoder-only e
totalmente denso, construído ao redor de refinamento recorrente IRL-2.

O repositório contém a arquitetura, geração com KV cache, objetivos auxiliares
MTP, testes, perfis de modelo e um pipeline completo de pretraining para uma
GPU, oito GPUs ou múltiplos nós. O mesmo código usa a API `cuda` pública do
PyTorch tanto em NVIDIA/CUDA quanto em AMD/ROCm.

> **Estado do projeto:** implementação experimental destinada a pesquisa.
> Todos os caminhos pequenos possuem testes, mas os hiperparâmetros dos perfis
> grandes ainda precisam ser validados em uma execução real. O código não deve
> ser apresentado como um modelo pretrained ou como resultado reproduzido de
> larga escala antes dessa validação.
>
> **Pipeline v0.4:** o núcleo matemático IRL-2 e os pesos do modelo não foram
> alterados. Esta revisão adiciona retomada de dados independente do world
> size, FSDP2/HSDP 2D, launcher elástico, preflight sem alocação e proteções de
> checkpoint. Esses caminhos novos ainda requerem qualificação multi-GPU.

Veja [`RELEASE_NOTES_v0.4.0.md`](RELEASE_NOTES_v0.4.0.md) para a matriz de
decisões, compatibilidade e os próximos gates de GPU.

## Sumário

- [Visão geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Formulação IRL-2](#formulação-irl-2)
- [MTP acoplado ao IRL-2](#mtp-acoplado-ao-irl-2)
- [Função objetivo](#função-objetivo)
- [Atenção local eficiente](#atenção-local-eficiente)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Requisitos e instalação](#requisitos-e-instalação)
- [Uso mínimo](#uso-mínimo)
- [Dados de treinamento](#dados-de-treinamento)
- [Perfis incluídos](#perfis-incluídos)
- [Treinamento](#treinamento)
- [Configuração completa](#configuração-completa)
- [Distribuição e desempenho](#distribuição-e-desempenho)
- [Checkpoints e retomada](#checkpoints-e-retomada)
- [Validação, métricas e profiling](#validação-métricas-e-profiling)
- [Geração e KV cache](#geração-e-kv-cache)
- [Testes](#testes)
- [Procedimento recomendado para uma execução real](#procedimento-recomendado-para-uma-execução-real)
- [Limitações e trabalho futuro](#limitações-e-trabalho-futuro)
- [Referências](#referências)

## Visão geral

O Lunaris Dense IRL-2 elimina os componentes esparsos da arquitetura MoC
anterior. Não existem router, experts, top-k routing, capacity factor,
mediador ou losses de balanceamento.

Cada bloco executa:

1. RMSNorm;
2. atenção causal GQA com QK-Norm e RoPE;
3. residual de atenção;
4. normalização da âncora;
5. `T` passos de refinamento IRL-2 com pesos SwiGLU compartilhados;
6. adição da correção recorrente ao residual.

O projeto oferece:

- arquitetura decoder-only totalmente densa;
- RMSNorm com estatísticas em FP32;
- Grouped-Query Attention;
- QK-Norm por cabeça;
- RoPE com bases distintas para atenção local e global;
- padrão local/global repetível;
- sliding-window attention block-sparse;
- seleção automática entre FlexAttention e SDPA;
- fallback GQA compatível;
- IRL-2 com full backpropagation through iterations;
- profundidades diferentes para treino e avaliação;
- currículo de profundidade recorrente;
- halting global opcional em avaliação;
- MTP com dois ou mais horizontes configuráveis;
- uma única transição IRL compartilhada entre horizontes MTP;
- loss de consistência shallow-to-deep;
- embeddings ligados ao LM head;
- loss causal em chunks;
- `linear_cross_entropy` fundida quando disponível;
- KV cache global pré-alocado e cache local circular;
- geração greedy, temperature, top-k e top-p;
- DDP, FSDP2 e Distributed Checkpoint;
- execução NVIDIA/CUDA e AMD/ROCm;
- perfis prontos para oito MI300X;
- validação distribuída, JSONL, W&B opcional e profiler.

## Arquitetura

Considere o estado de entrada do bloco \(h_\ell\). A parte de atenção é:

\[
\tilde h_\ell =
h_\ell +
s_\ell^{\mathrm{attn}}\odot
\operatorname{Attn}_\ell(
\operatorname{RMSNorm}(h_\ell)
).
\]

A atenção utiliza GQA. Queries têm `n_heads`, enquanto keys e values usam
`n_kv_heads`. É exigido:

\[
d_{\mathrm{model}}\bmod n_{\mathrm{heads}}=0
\]

e

\[
n_{\mathrm{heads}}\bmod n_{\mathrm{kv\_heads}}=0.
\]

O estado \(\tilde h_\ell\) se torna a âncora normalizada do IRL-2:

\[
x_\ell = \operatorname{RMSNorm}(\tilde h_\ell).
\]

Após o refinamento recorrente \(r_\ell^{(T)}\), a saída é:

\[
h_{\ell+1} = \tilde h_\ell + r_\ell^{(T)}.
\]

Os pesos da célula IRL são compartilhados entre as iterações de uma mesma
camada, mas não entre camadas diferentes.

## Formulação IRL-2

Para cada bloco, o estado recorrente começa em:

\[
r^{(0)}=0.
\]

Em cada iteração:

\[
v^{(t)} =
\operatorname{RMSNorm}(x+r^{(t)})
+ g\odot x
\]

\[
F_\theta(r^{(t)};x)=
\beta\odot W_d\left[
\operatorname{SiLU}(W_gv^{(t)})
\odot W_uv^{(t)}
\right]
\]

\[
r^{(t+1)} =
(1-\lambda)r^{(t)}
+ \lambda F_\theta(r^{(t)};x).
\]

O damping é aprendido, mas permanece dentro de um intervalo:

\[
\lambda =
\lambda_{\min}
+ (\lambda_{\max}-\lambda_{\min})
\sigma(\lambda_{\mathrm{logit}}).
\]

Parâmetros principais:

- `irl_train_iters`: limite de iterações durante o treinamento;
- `irl_eval_iters`: iterações padrão durante avaliação e geração;
- `irl_lambda_min` e `irl_lambda_max`: intervalo do damping;
- `irl_lambda_init`: valor inicial do damping;
- `irl_anchor_gate_init`: gate inicial da âncora;
- `irl_layer_scale_init`: escala inicial da correção;
- `irl_halt_tol`: tolerância opcional para halting em avaliação;
- `irl_halt_percentile`: percentil usado pelo critério de halting.

Halting por token não é utilizado. Interromper tokens individuais dentro de
GEMMs densos normalmente não reduz compute. O código implementa apenas halting
global do batch em avaliação.

## MTP acoplado ao IRL-2

O Multi-Token Prediction é um objetivo de treinamento. O backbone principal
continua sendo um modelo causal next-token.

Para um horizonte \(k\ge2\), são combinados:

- o hidden state final do backbone na posição \(t\);
- o embedding teacher-forced do token \(x_{t+k-1}\);
- um embedding aprendido do horizonte \(q_k\).

A fusão é:

\[
a_t^{(k)} =
\operatorname{RMSNorm}\left(
P_h\operatorname{RMSNorm}(h_t)
+ \sigma(g_{\mathrm{mtp}})\odot
P_e\operatorname{RMSNorm}(E[x_{t+k-1}])
+ q_k
\right).
\]

Uma célula IRL adicional refina \(a_t^{(k)}\). Essa célula, as projeções
\(P_h\) e \(P_e\), o embedding de tokens e o LM head são compartilhados entre
todos os horizontes. Apenas \(q_k\) depende do horizonte.

\[
z_t^{(k)} =
\operatorname{RMSNorm}
\left(
a_t^{(k)}
+ \operatorname{IRL}_{\phi}(a_t^{(k)})
\right)
\]

\[
p(x_{t+k}\mid x_{\le t},x_{t+k-1},k)
=
\operatorname{softmax}(W_{\mathrm{LM}}z_t^{(k)}).
\]

Com `mtp_num_heads: 2`, os horizontes treinados são \(k=2\) e \(k=3\).
O MTP não é executado na geração normal. Os pesos permanecem no checkpoint
para pesquisa futura com self-speculative decoding.

Parâmetros:

- `mtp_num_heads`: quantidade de horizontes adicionais;
- `mtp_irl_iters`: iterações da célula IRL compartilhada do MTP;
- `mtp_ffn_hidden_dim`: largura do SwiGLU auxiliar.

## Função objetivo

A loss total é:

\[
\mathcal L =
\mathcal L_{\mathrm{NTP}}
+ \gamma\mathcal L_z
+ \alpha(s)\mathcal L_{\mathrm{MTP}}
+ \beta(s)\mathcal L_{\mathrm{cons}}.
\]

Os horizontes MTP recebem decaimento geométrico:

\[
\mathcal L_{\mathrm{MTP}}
=
\frac{
\sum_{k=2}^{K}\rho^{k-2}\mathcal L_k
}{
\sum_{k=2}^{K}\rho^{k-2}
}.
\]

A consistência faz a iteração anterior aproximar uma iteração mais profunda,
tratada como professora:

\[
\mathcal L_{\mathrm{cons}}
=
\mathbb E_j
\left[
\frac{
\|r^{(j)}-\operatorname{sg}(r^{(j+1)})\|_2^2
}{
\|\operatorname{sg}(r^{(j+1)})\|_2^2+\epsilon
}
\right].
\]

`sg` representa stop-gradient. A loss causal principal evita que a solução
trivial de zerar todas as correções seja suficiente.

O currículo controla:

- número ativo de iterações IRL;
- peso MTP \(\alpha(s)\);
- quantidade de horizontes MTP;
- peso de consistência \(\beta(s)\).

O primeiro head MTP permanece no grafo durante o warmup de peso zero. Isso
garante que DDP observe todos os parâmetros auxiliares desde o primeiro step.

## Atenção local eficiente

Camadas globais usam o caminho causal do
`torch.nn.functional.scaled_dot_product_attention`.

Camadas locais tentam usar FlexAttention com um `BlockMask` causal:

\[
t-w+1 \le j \le t.
\]

O `BlockMask` é armazenado em cache por:

- comprimento da sequência;
- sliding window;
- device;
- block size.

Isso evita materializar uma máscara booleana completa
`[batch, sequence, sequence]` no caminho comum.

O callable do FlexAttention usa compilação regional: um único grafo fundido é
compilado por formato e reutilizado por todas as camadas locais compatíveis.
Isso funciona mesmo quando `parallel.compile` está desativado e evita tanto o
fallback não fundido quanto a compilação repetida do modelo completo.

O backend é configurado por:

```yaml
attention_backend: "auto"  # auto, sdpa, or flex
flex_block_size: 128
```

Com `auto`, FlexAttention é usado quando:

- o device é CUDA ou ROCm pela API `cuda` do PyTorch;
- a camada é local;
- não existe KV cache;
- não existe `attention_mask` explícita;
- não existem `document_ids`;
- attention dropout é zero durante treino.

Nos demais casos, o modelo usa SDPA e constrói a máscara necessária.
`attention_backend: flex` transforma a ausência dessas condições em erro.
`attention_backend: sdpa` desativa completamente FlexAttention.

Packed documents ainda usam o fallback explícito porque a máscara muda por
batch. Targets NTP e MTP que atravessam documentos são automaticamente
substituídos por `loss_ignore_index`.

## Estrutura do repositório

```text
lunaris_dense_irl2/
├── README.md
├── TRAINING.md
├── pyproject.toml
├── requirements.txt
├── examples/
│   └── smoke_test.py
├── lunaris_dense_irl2/
│   ├── __init__.py
│   ├── attention.py
│   ├── block.py
│   ├── cache.py
│   ├── config.py
│   ├── diagnostics.py
│   ├── generation.py
│   ├── initialization.py
│   ├── irl2.py
│   ├── model.py
│   ├── mtp.py
│   ├── norms.py
│   ├── rope.py
│   └── training/
│       ├── checkpointing.py
│       ├── config.py
│       ├── data.py
│       ├── distributed.py
│       ├── export_checkpoint.py
│       ├── logging.py
│       ├── optimizers.py
│       ├── schedules.py
│       └── train.py
├── profiles/
│   ├── debug.yaml
│   ├── 120m.yaml
│   ├── 350m.yaml
│   └── 950m.yaml
├── training_profiles/
│   ├── debug_cpu.yaml
│   ├── 120m_8x_mi300x.yaml
│   ├── 350m_8x_mi300x.yaml
│   ├── 350m_8x_nvidia.yaml
│   └── 950m_8x_mi300x.yaml
├── scripts/
│   ├── launch_mi300x_8gpu.sh
│   ├── launch_nvidia_8gpu.sh
│   └── launch_multinode.sh
└── tests/
    ├── test_config.py
    ├── test_irl2.py
    ├── test_model.py
    ├── test_optional_paths.py
    └── test_training.py
```

Comentários e docstrings do código-fonte estão em inglês.

## Requisitos e instalação

Requisitos mínimos:

- Python 3.10 ou mais recente;
- PyTorch 2.7 ou mais recente;
- NumPy;
- PyYAML;
- pytest para testes;
- W&B apenas se logging remoto for desejado.

Instale primeiro um build do PyTorch correspondente ao hardware.

### AMD/ROCm

Use uma imagem oficial ROCm/PyTorch compatível com a versão instalada no
cluster. Depois:

```bash
python -m pip install -e ".[test]"
```

### NVIDIA/CUDA

Instale o wheel CUDA adequado e depois:

```bash
python -m pip install -e ".[test]"
```

### Monitoramento opcional

```bash
python -m pip install -e ".[test,monitoring]"
```

Confirme o ambiente:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("accelerator available:", torch.cuda.is_available())
print("CUDA:", torch.version.cuda)
print("ROCm/HIP:", torch.version.hip)
print("devices:", torch.cuda.device_count())
PY
```

No ROCm, `torch.cuda` continua sendo a API pública do PyTorch. O backend
distribuído solicitado é `nccl`, mas a implementação utilizada é RCCL.

## Uso mínimo

```python
import torch

from lunaris_dense_irl2 import (
    LunarisDenseConfig,
    LunarisDenseForCausalLM,
)

config = LunarisDenseConfig.debug(
    mtp_num_heads=2,
    mtp_irl_iters=1,
)
model = LunarisDenseForCausalLM(config)

input_ids = torch.randint(0, config.vocab_size, (2, 32))
output = model(
    input_ids,
    labels=input_ids,
    return_logits=False,
    loss_chunk_size=128,
    mtp_loss_weight=0.2,
    mtp_active_heads=2,
    irl_consistency_coefficient=0.01,
)

print("total:", output.loss)
print("next-token:", output.cross_entropy)
print("mtp:", output.mtp_loss)
print("irl consistency:", output.irl_consistency_loss)

output.loss.backward()
```

`labels` tem o mesmo formato de `input_ids`. O modelo faz o deslocamento causal
internamente. Posições com `loss_ignore_index`, cujo padrão é `-100`, não
contribuem para a loss.

Carregar um perfil:

```python
from lunaris_dense_irl2 import LunarisDenseConfig

config = LunarisDenseConfig.from_yaml("profiles/350m.yaml")
```

## Dados de treinamento

O trainer incluído espera shards NPY:

- um array unidimensional por arquivo;
- dtype inteiro, normalmente `uint32`;
- tokens já produzidos pelo tokenizer;
- nenhum objeto Python ou pickle.

Exemplo:

```python
from pathlib import Path

import numpy as np

destination = Path("data/train")
destination.mkdir(parents=True, exist_ok=True)
tokens = np.asarray([1, 17, 93, 4, 2], dtype=np.uint32)
np.save(destination / "shard_00000.npy", tokens)
```

Estrutura sugerida:

```text
/data/lunaris/
├── train/
│   ├── shard_00000.npy
│   ├── shard_00001.npy
│   └── ...
└── validation/
    ├── shard_00000.npy
    └── ...
```

O dataset:

- abre arquivos com memory mapping;
- nunca carrega o corpus inteiro em RAM;
- não cruza fronteiras de shards;
- produz blocos fixos e não sobrepostos;
- descarta apenas a cauda menor que uma sequência;
- mantém um pequeno cache LRU de mmaps por worker;
- usa uma permutação afim determinística de memória O(1);
- separa amostras entre ranks.

Cada shard deve conter uma sequência de tokens coerente. Se fronteiras exatas
de documentos precisarem ser preservadas, use `document_ids` e `position_ids`
em um dataloader customizado.

Exemplo de packed documents:

```python
position_ids = torch.tensor([[0, 1, 2, 0, 1]])
document_ids = torch.tensor([[0, 0, 0, 1, 1]])

output = model(
    input_ids,
    labels=input_ids,
    position_ids=position_ids,
    document_ids=document_ids,
)
```

## Perfis incluídos

| Perfil | Contexto | Atenção | MTP | Estratégia sugerida |
|---|---:|---|---:|---|
| `debug.yaml` | 256 | local/global | 2 | CPU ou uma GPU |
| `120m.yaml` | 4.096 | global | 2 | DDP |
| `135m_mi300x_qualification.yaml` | 4.096 | `LLLG` | 2 | uma MI300X |
| `350m.yaml` | 8.192 | `LLLG` | 2 | DDP |
| `950m.yaml` | 32.768 | `LLLLLG` | 2 | FSDP2 |

Os nomes 120M, 350M e 950M identificam a escala original do backbone. O MTP
adiciona parâmetros auxiliares compartilhados, portanto a contagem final deve
ser medida com `model.get_num_parameters()` no ambiente real.

Perfis de treinamento:

- `training_profiles/120m_8x_mi300x.yaml`;
- `training_profiles/350m_8x_mi300x.yaml`;
- `training_profiles/950m_8x_mi300x.yaml`;
- `training_profiles/350m_8x_nvidia.yaml`;
- `training_profiles/135m_1x_mi300x_qualification.yaml`;
- `training_profiles/135m_1x_mi300x_100m_continuation.yaml`;
- `training_profiles/135m_1x_mi300x_throughput_probe.yaml`;
- `training_profiles/mi300x_smoke.yaml`;
- `training_profiles/debug_cpu.yaml`.

Os caminhos de dataset e checkpoint nos perfis grandes são exemplos e devem
ser alterados antes do treinamento.

## Treinamento

### Validação rápida em CPU

Crie dados pequenos:

```bash
python - <<'PY'
from pathlib import Path

import numpy as np

for split, seed in (("train", 7), ("validation", 8)):
    path = Path("data/debug") / split
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 257, 4096, dtype=np.uint32)
    np.save(path / "shard_000.npy", tokens)
PY
```

Execute:

```bash
python -m lunaris_dense_irl2.training.train \
  --config training_profiles/debug_cpu.yaml
```

### Uma GPU

Qualquer perfil pode usar uma GPU quando `parallel.strategy` for `none`:

```yaml
parallel:
  strategy: "none"
  precision: "bf16"
  compile: false
  regional_compile: true
```

```bash
python -m lunaris_dense_irl2.training.train \
  --config path/to/profile.yaml
```

Para a qualificação reproduzível em uma única MI300X, incluindo download de um
shard, divisão sem vazamento, smoke test, treino, resume e relatório:

```bash
bash scripts/run_mi300x_qualification.sh
```

Consulte [`MI300X_QUALIFICATION.md`](MI300X_QUALIFICATION.md) para escopo,
orçamento de tokens, artefatos e critérios de aceitação. Esse perfil usa o
tokenizer Lunaris completo de 65.536 tokens; ele não é intercambiável com o
perfil original de vocabulário 48.000.

### Oito MI300X

```bash
chmod +x scripts/launch_mi300x_8gpu.sh

./scripts/launch_mi300x_8gpu.sh \
  --config training_profiles/350m_8x_mi300x.yaml
```

O launcher define:

```bash
TORCH_NCCL_HIGH_PRIORITY=1
GPU_MAX_HW_QUEUES=2
NCCL_DEBUG=WARN
TORCH_DISTRIBUTED_DEBUG=OFF
```

### Oito GPUs NVIDIA

```bash
chmod +x scripts/launch_nvidia_8gpu.sh

./scripts/launch_nvidia_8gpu.sh \
  --config training_profiles/350m_8x_nvidia.yaml
```

### Múltiplos nós

Em cada nó:

```bash
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
export GPUS_PER_NODE=8

./scripts/launch_multinode.sh \
  --config training_profiles/950m_8x_mi300x.yaml
```

No segundo nó, use `NODE_RANK=1`. Dataset e checkpoints precisam estar
acessíveis em todos os nós.

Para reinício elástico dos workers, use um rendezvous compartilhado:

```bash
export MIN_NODES=2
export MAX_NODES=2
export GPUS_PER_NODE=8
export RDZV_ENDPOINT=10.0.0.10:29400
export RUN_ID=lunaris-pretrain-001
export MAX_RESTARTS=3

./scripts/launch_elastic_multinode.sh \
  --config path/to/profile.yaml
```

Elasticidade não substitui storage compartilhado: shards e checkpoints
precisam apontar para o mesmo conteúdo em todos os nós.

### Preflight sem GPU

O preflight instancia o modelo no device `meta`, lê apenas os headers NPY e
gera topologia, budget de tokens, contagem de parâmetros, fingerprint da
configuração e um limite inferior de memória persistente:

```bash
lunaris-preflight \
  --config training_profiles/950m_8x_mi300x.yaml \
  --world-size 8 \
  --output preflight_950m_8gpu.json
```

Use `--require-data` no servidor final para tornar shards ausentes um erro. A
estimativa de memória não inclui ativações, buffers de coletivas, workspaces
do compilador ou fragmentação; ela nunca substitui um smoke test real.

### Override para smoke test

```bash
./scripts/launch_mi300x_8gpu.sh \
  --config training_profiles/350m_8x_mi300x.yaml \
  --max-steps 20 \
  --resume none
```

## Configuração completa

O YAML raiz aceita:

```yaml
model_profile: "../profiles/350m.yaml"

model: {}
data: {}
optimizer: {}
schedule: {}
parallel: {}
training: {}
objectives: {}
validation: {}
checkpoint: {}
logging: {}
```

Campos desconhecidos são rejeitados.

### `model`

Principais campos:

```yaml
model:
  vocab_size: 64000
  d_model: 1024
  n_layers: 32
  n_heads: 16
  n_kv_heads: 4
  ffn_hidden_dim: 2048
  max_seq_len: 8192

  attention_pattern: "LLLG"
  local_window: 1024
  attention_backend: "auto"
  flex_block_size: 128

  qk_norm: true
  sandwich_norm: false
  attention_output_gate: false
  tie_word_embeddings: true

  rope_theta_global: 1000000.0
  rope_theta_local: 10000.0

  irl_train_iters: 2
  irl_eval_iters: 2
  irl_lambda_min: 0.2
  irl_lambda_max: 1.0
  irl_lambda_init: 0.9

  mtp_num_heads: 2
  mtp_irl_iters: 1
  mtp_ffn_hidden_dim: 1024

  use_gradient_checkpointing: true
  gradient_checkpointing_policy: "irl"
```

### `data`

```yaml
data:
  train_dir: "/data/lunaris/train"
  validation_dir: "/data/lunaris/validation"
  shard_glob: "*.npy"
  sequence_length: 8192
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 4
  mmap_cache_size: 8
  shuffle: true
  seed: 1337
```

### `optimizer`

```yaml
optimizer:
  name: "adamw"  # adamw or mona
  learning_rate: 0.0004
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  epsilon: 1.0e-8
  fused: "auto"
```

MONA é experimental e utiliza matrizes completas. Por isso, não é permitido
com FSDP2.

### `schedule`

```yaml
schedule:
  name: "wsd"  # wsd, cosine, or constant
  warmup_steps: 2000
  stable_steps: 88000
  decay_steps: 10000
  min_lr_ratio: 0.1
  decay_shape: "cosine"
```

WSD significa warmup-stable-decay.

### `parallel`

```yaml
parallel:
  strategy: "ddp"  # auto, none, ddp, or fsdp2
  precision: "bf16"
  reduce_dtype: "fp32"
  compile: false
  regional_compile: true
  compile_mode: "max-autotune-no-cudagraphs"
  reshard_after_forward: false
  fsdp_replicate_group_size: 1
  ddp_static_graph: false
  deterministic: false
  deterministic_warn_only: false
  timeout_minutes: 30
```

`ddp_static_graph` permanece falso nos perfis MTP porque a quantidade de
horizontes executados muda durante o currículo.

`compile` significa compilação do modelo completo. Ela é mutuamente exclusiva
com `regional_compile`, que é o caminho medido na MI300X. Para HSDP, use
`fsdp_replicate_group_size` maior que 1; `WORLD_SIZE` deve ser divisível por
esse valor.

### `training`

```yaml
training:
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  max_steps: 100000
  max_tokens: null
  seed: 1337

  gradient_clip: 1.0
  gradient_clip_mode: "global"
  adaptive_clip_beta: 0.98
  adaptive_clip_factor: 2.0

  loss_chunk_size: 2048
  loss_backend: "auto"
  z_loss_coefficient: 1.0e-4
  fail_on_nonfinite: true
```

`loss_backend`:

- `auto`: usa a operação fundida quando compatível;
- `chunked`: força a implementação portátil interna;
- `native`: exige `torch.nn.functional.linear_cross_entropy`.

O backend nativo requer `z_loss_coefficient: 0`. Com z-loss positiva, `auto`
seleciona o caminho interno, que também calcula o log-partition.

Em aceleradores, o caminho interno compila apenas o kernel regional
linear+CE+z-loss. O último chunk é preenchido com labels ignorados para manter
um formato estático e permitir que CE principal e horizontes MTP reutilizem o
mesmo grafo. Chunks muito pequenos criam muitos GEMMs de vocabulário; para a
MI300X e vocabulário 65.536, `loss_chunk_size: 4096` foi substancialmente mais
rápido que 1024.

### `objectives`

```yaml
objectives:
  irl_start_iters: 1
  irl_ramp_steps: 10000

  irl_consistency_weight: 0.01
  irl_consistency_warmup_steps: 5000
  irl_consistency_ramp_steps: 5000

  mtp_loss_weight: 0.2
  mtp_warmup_steps: 2000
  mtp_ramp_steps: 8000
  mtp_horizon_interval: 4000
  mtp_horizon_decay: 0.7
```

Se o modelo possuir MTP, `mtp_loss_weight` precisa ser positivo. Para treinar
sem MTP, use `model.mtp_num_heads: 0`.

### `validation`

```yaml
validation:
  interval: 500
  batches: 50
  at_start: false
```

### `checkpoint`

```yaml
checkpoint:
  directory: "checkpoints/lunaris_350m"
  interval: 1000
  keep_last: 3
  async_save: true
  resume: "auto"
  save_final: true
```

### `logging`

```yaml
logging:
  interval: 10
  jsonl_path: "checkpoints/lunaris_350m/metrics.jsonl"
  wandb_project: null
  wandb_entity: null
  wandb_run_name: null

  profile_steps: 0
  profile_wait: 1
  profile_warmup: 1
  profile_directory: "traces"
  peak_tflops_per_gpu: null
```

### `checkpoint`

```yaml
checkpoint:
  directory: "/shared/checkpoints/lunaris"
  interval: 1000
  keep_last: 3
  async_save: true
  resume: "auto"
  save_final: true
  min_free_space_gib: 100.0
```

`min_free_space_gib` faz todos os ranks falharem antes de iniciar um save
quando o filesystem compartilhado está abaixo da reserva configurada.

## Distribuição e desempenho

### Batch global

\[
B_{\mathrm{global}} =
B_{\mathrm{micro}}
\times N_{\mathrm{acumulação}}
\times N_{\mathrm{ranks}}.
\]

Tokens por atualização:

\[
N_{\mathrm{tokens/step}} =
B_{\mathrm{global}}
\times L_{\mathrm{sequência}}.
\]

### DDP

Use DDP quando modelo, ativações e estados do otimizador couberem em cada GPU.
Para 120M e 350M, essa normalmente é a primeira opção em MI300X.

### FSDP2

Use FSDP2 quando replicação completa se tornar cara. O modelo aplica
`fully_shard` em cada bloco e depois na raiz.

No perfil 950M/MI300X:

```yaml
reshard_after_forward: false
```

Isso mantém parâmetros materializados até o backward, usando mais memória para
evitar outro all-gather. Se ocorrer OOM, altere para `true`.

### HSDP

Em múltiplos nós, HSDP cria uma malha 2D: replica no eixo 0 e fragmenta no eixo
1. Em 2 nós com 8 GPUs por nó:

```yaml
parallel:
  strategy: "fsdp2"
  fsdp_replicate_group_size: 2
```

Isso produz mesh `[2, 8]`: os parâmetros são fragmentados dentro de grupos de
8 ranks e replicados entre 2 grupos. O melhor formato depende da topologia de
rede; em um único nó mantenha o valor 1 até medir o contrário.

### Precisão

- BF16 é o padrão recomendado;
- FP16 utiliza GradScaler;
- FP32 é destinado a depuração;
- reductions distribuídas podem permanecer em FP32.

FP8 não é habilitado automaticamente.

### `torch.compile`

FlexAttention e a loss linear em chunks são compilados regionalmente por
padrão no acelerador. Essa estratégia preserva grafos pequenos e reutilizáveis.

`parallel.compile: true` compila o modelo inteiro antes do wrapper distribuído
e desliga o caminho regional. Use essa opção somente depois de um probe
representativo: IRL, MTP e loops de chunks podem gerar muitos grafos
especializados e um autotune inicial caro. Todos os perfis fornecidos agora
usam compilação regional com `parallel.compile: false`.

### Gradient checkpointing

Políticas:

- `none`: nenhuma recomputação;
- `block`: checkpoint do bloco completo;
- `irl`: checkpoint somente da parte recorrente.

`irl` é recomendado porque evita recalcular a atenção e recomputa as operações
recorrentes/pointwise.

### Otimizadores

AdamW é o caminho principal.

MONA é uma opção de pesquisa:

- matrizes internas usam MONA;
- embeddings, heads, vetores e escalas usam AdamW;
- horizon embeddings permanecem no AdamW;
- não é compatível com FSDP2.

### Gradient clipping

- `global`: clipping global padrão;
- `adaptive_group`: grupos de atenção, IRL, MTP, embeddings/head e escalas;
- `none`: mede o norm sem limitar.

Adaptive group clipping é restrito a parâmetros replicados e não deve ser
usado com FSDP2.

## Checkpoints e retomada

O pipeline utiliza PyTorch Distributed Checkpoint.

Cada checkpoint inclui:

- model state;
- optimizer state;
- GradScaler quando aplicável;
- step;
- microbatches consumidos;
- cursor global de amostras;
- tokens vistos;
- melhor validation loss;
- RNG de Python, NumPy e PyTorch;
- configuração resolvida.

`resume: auto` seleciona somente checkpoints com marcador `_SUCCESS`.

O cursor global de amostras é independente do world size. Um checkpoint salvo
com 8 ranks pode ser retomado com 4 sem repetir o prefixo já consumido. DCP
redistribui modelo e otimizador; como a quantidade de streams mudou, os RNGs
locais de dropout são reseeded de forma determinística e a retomada deixa de
ser bitwise idêntica. Com a mesma topologia, cada RNG por rank é restaurado.

Retomar:

```bash
./scripts/launch_mi300x_8gpu.sh \
  --config training_profiles/350m_8x_mi300x.yaml \
  --resume auto
```

Exportar um checkpoint distribuído para o formato portátil:

```bash
lunaris-export \
  --checkpoint checkpoints/lunaris_350m/step_00100000 \
  --output exports/lunaris_350m
```

O export contém:

```text
exports/lunaris_350m/
├── config.json
└── model.pt
```

Carregar:

```python
from lunaris_dense_irl2 import LunarisDenseForCausalLM

model = LunarisDenseForCausalLM.from_pretrained("exports/lunaris_350m")
```

## Validação, métricas e profiling

Métricas do trainer:

- total loss;
- next-token cross entropy;
- z-loss;
- MTP loss;
- IRL consistency loss;
- perplexity NTP;
- gradient norm;
- learning rate;
- iterações IRL ativas;
- heads MTP ativos;
- pesos auxiliares ativos;
- tokens por segundo;
- memória alocada, reservada e peak;
- TFLOPs e MFU estimados quando peak teórico é configurado;
- validation loss e validation perplexity.

JSONL é gravado apenas pelo rank zero. W&B é opcional.

Ativar profiler:

```yaml
logging:
  profile_steps: 5
  profile_wait: 1
  profile_warmup: 1
  profile_directory: "traces"
```

As traces são separadas por rank e podem ser abertas em ferramentas
compatíveis com Chrome Trace.

## Geração e KV cache

```python
model.eval()

generated = model.generate(
    input_ids,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    n_iters=2,
)
```

Com `temperature=0.0`, a geração é greedy.

O cache:

- mantém buffers globais pré-alocados para camadas globais;
- usa buffers circulares limitados pela sliding window em camadas locais;
- armazena posições absolutas para RoPE;
- não é permitido durante treinamento;
- não suporta packed documents.

Prompts de um mesmo batch devem ter o mesmo comprimento e não utilizar padding.

## Testes

Instale:

```bash
python -m pip install -e ".[test]"
```

Execute:

```bash
pytest
```

Smoke test:

```bash
python examples/smoke_test.py
```

Os testes cobrem:

- validação de configuração;
- shapes, loss e gradientes;
- equivalência entre loss materializada e chunked;
- recorrência IRL com pesos compartilhados;
- consistência IRL;
- MTP e gradientes de todos os parâmetros auxiliares;
- isolamento de packed documents;
- targets que cruzam documentos;
- equivalência entre cache e forward completo;
- geração;
- gradient checkpointing;
- halting;
- sampler distribuído;
- retomada do sampler após mudança de world size;
- resolução de meshes FSDP/HSDP;
- preflight no device meta;
- memory-mapped dataset;
- WSD;
- currículo IRL/MTP;
- MONA;
- treinamento CPU;
- checkpoint, resume e export.

## Procedimento recomendado para uma execução real

1. Execute `lunaris-preflight --require-data` para validar configuração,
   shards, budget e mesh sem alocar o modelo.
2. Instale e execute todos os testes no mesmo build PyTorch/ROCm ou CUDA que
   será usado no cluster.
3. Ajuste os caminhos de dataset e checkpoint compartilhado.
4. Rode 20 steps com `--resume none`.
5. Confirme losses finitas, gradient norm e uso de memória.
6. Verifique que `mtp_loss` e `irl_consistency_loss` são registradas.
7. Execute save, encerramento e resume com a mesma topologia.
8. Teste uma retomada curta com topologia diferente antes de habilitar o
   launcher elástico.
9. Ative o profiler por poucos steps.
10. Compare DDP e FSDP2 quando ambos couberem.
11. Compare `attention_backend: auto` e `sdpa`.
12. Faça uma ablação curta com:
    - MTP desligado;
    - MTP ligado;
    - consistência desligada;
    - uma versus duas iterações IRL.
13. Só depois aumente o orçamento de tokens.

Configuração mínima para bring-up:

```bash
./scripts/launch_mi300x_8gpu.sh \
  --config training_profiles/350m_8x_mi300x.yaml \
  --max-steps 20 \
  --resume none
```

Não reutilize um checkpoint sem MTP em um modelo com MTP usando strict load.
Também não troque AdamW por MONA no meio do mesmo experimento.

## Limitações e trabalho futuro

Não estão incluídos:

- Mixture of Experts;
- MoC;
- router ou mediador;
- Multi-Latent Attention;
- Mamba ou SSM;
- Gated DeltaNet;
- solver implícito;
- Jacobian-Free Backpropagation;
- normalização espectral rígida;
- Context Parallel;
- Tensor Parallel;
- Pipeline Parallel;
- FP8 automático;
- speculative decoding com os heads MTP;
- multimodalidade.

Context Parallel é a próxima extensão relevante para contexto 32k quando uma
sequência individual não couber confortavelmente. FP8 deve ser adicionado
somente após benchmark e validação numérica independentes para ROCm e CUDA.

FlexAttention permanece uma API de evolução rápida no PyTorch. O fallback
SDPA existe para compatibilidade, mas contexto 32k com máscara local densa pode
consumir muita memória.

## Reprodutibilidade

O trainer registra a configuração completamente resolvida e salva RNG por
rank. O sampler é determinístico para:

- mesmo dataset;
- mesmos shards;
- mesma seed;
- mesmo cursor global de amostras.

Alterar world size após resume preserva o stream global de dados, o modelo e o
otimizador via DCP. A ordem futura é repartida entre a nova quantidade de
ranks; RNGs locais são reseeded, portanto a trajetória numérica não é bitwise
idêntica.

Resultados também podem variar entre versões de PyTorch, ROCm, CUDA, Triton,
drivers e kernels de atenção.

## Referências

- [PyTorch FlexAttention](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html)
- [PyTorch Scaled Dot Product Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch Linear Cross Entropy](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.linear_cross_entropy.html)
- [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch asynchronous Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
- [PyTorch Context Parallel](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [AMD MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [Curriculum-based Multi-Token Prediction, 2025](https://arxiv.org/abs/2505.22757)
- [Better & Faster LLMs via Multi-Token Prediction, 2024](https://arxiv.org/abs/2404.19737)
- [MONA, 2026](https://arxiv.org/abs/2605.26842)
- [AGGC, 2026](https://arxiv.org/abs/2601.11864)
- [Muon is Scalable for LLM Training, 2025](https://arxiv.org/abs/2502.16982)

## Uso interno

Este repositório não inclui uma licença open source. Para um repositório privado
e interno, mantenha o acesso restrito e adicione uma licença explícita apenas se
o projeto for distribuído para terceiros.
