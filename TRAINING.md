# Treinamento em larga escala

O trainer foi escrito diretamente para a arquitetura densa com IRL-2. Ele não importa nem preserva métricas, routers, experts ou losses auxiliares do MoC anterior.

## Recursos

- uma GPU, DDP ou FSDP2;
- HSDP 2D para fragmentação intra-grupo e replicação entre grupos;
- `torchrun` em um nó ou múltiplos nós;
- launcher elástico com restart a partir do último checkpoint completo;
- NVIDIA/NCCL e AMD/RCCL pela mesma API `cuda` do PyTorch;
- BF16 por padrão, FP16 com loss scaling e FP32 para depuração;
- acumulação de gradientes sem sincronização nos microbatches intermediários;
- gradient checkpointing por bloco ou somente no IRL-2;
- FlexAttention com máscara local block-sparse em treino sem padding;
- `torch.compile` aplicado antes da camada distribuída;
- AdamW fused quando suportado;
- MONA experimental para DDP;
- clipping global ou clipping adaptativo por grupos funcionais;
- warmup-stable-decay, cosine ou constant LR;
- linear cross entropy fundida quando suportada pelo PyTorch instalado;
- fallback causal e z-loss em chunks, sem manter todos os logits do vocabulário;
- MTP acoplado a uma célula IRL compartilhada;
- currículo conjunto de profundidade IRL, horizontes MTP e consistência;
- shards NPY memory-mapped;
- sampler distribuído determinístico com memória O(1);
- cursor global de dados que sobrevive a mudanças de world size;
- prefetch H2D em stream separado;
- Distributed Checkpoint com resharding;
- checkpoint assíncrono, retenção e retomada automática;
- preflight no device meta sem alocar o modelo;
- validação distribuída ponderada por tokens;
- console, JSONL, W&B opcional e PyTorch Profiler;
- throughput, memória, perplexidade, TFLOPs e MFU estimado.

## Instalação

Instale primeiro a versão de PyTorch correspondente ao hardware.

Em uma imagem ROCm:

```bash
python -m pip install -e ".[monitoring]"
```

Em uma imagem CUDA, o comando do pacote é o mesmo. A diferença está no wheel do PyTorch previamente instalado na imagem.

Confirme o ambiente:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda API available:", torch.cuda.is_available())
print("CUDA:", torch.version.cuda)
print("ROCm/HIP:", torch.version.hip)
print("devices:", torch.cuda.device_count())
PY
```

No ROCm, `torch.cuda` continua sendo a API pública usada pelo PyTorch. O backend distribuído é solicitado como `nccl`, mas a implementação usada é RCCL.

## Dados

Cada arquivo deve ser um array NPY unidimensional com dtype inteiro:

```python
import numpy as np

tokens = np.asarray([1, 17, 93, 4, 2], dtype=np.uint32)
np.save("data/train/shard_00000.npy", tokens)
```

Estrutura sugerida:

```text
/data/lunaris/
├── train/
│   ├── shard_00000.npy
│   └── shard_00001.npy
└── validation/
    └── shard_00000.npy
```

O dataset:

- nunca carrega o corpus inteiro em RAM;
- não cruza fronteiras de shards;
- descarta apenas a cauda menor que uma sequência;
- abre shards sob demanda com um pequeno cache LRU por worker;
- usa uma permutação afim determinística em vez de criar um `randperm` global.

Quando `document_ids` é usado por um pipeline de packing customizado, a loss
ignora automaticamente previsões NTP e MTP que cruzariam a fronteira entre
documentos. Máscaras por documento ainda usam o fallback SDPA explícito; o
caminho FlexAttention mais rápido é selecionado para sequências causais comuns,
sem máscara variável por batch.

O sampler garante que ranks diferentes não recebam a mesma posição do stream.
A retomada restaura um cursor global de amostras, não um contador local de
microbatches. Se o número de GPUs for alterado, modelo e otimizador são
redistribuídos pelo DCP e o prefixo de dados já consumido não é repetido. Os
RNGs locais são reseeded porque a topologia de streams mudou.

## Teste em CPU

Crie alguns shards pequenos:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

for split in ("train", "validation"):
    path = Path("data/debug") / split
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7 if split == "train" else 8)
    np.save(path / "shard_000.npy", rng.integers(0, 257, 4096, dtype=np.uint32))
PY

python -m lunaris_dense_irl2.training.train \
  --config training_profiles/debug_cpu.yaml
```

## Oito MI300X

Edite os caminhos do dataset e do checkpoint no perfil escolhido. Depois:

```bash
./scripts/launch_mi300x_8gpu.sh \
  --config training_profiles/350m_8x_mi300x.yaml
```

Perfis incluídos:

- `120m_8x_mi300x.yaml`: DDP;
- `350m_8x_mi300x.yaml`: DDP;
- `950m_8x_mi300x.yaml`: FSDP2.

DDP normalmente é mais simples e rápido quando o modelo, ativações e estados do otimizador cabem confortavelmente em cada MI300X. FSDP2 passa a ser útil quando a memória replicada ou o tamanho futuro do modelo justificam all-gather/reduce-scatter.

O perfil de 950M mantém `reshard_after_forward: false`. Isso consome mais
memória, mas evita um segundo all-gather antes do backward. Na MI300X de
192&nbsp;GB essa é uma troca geralmente mais interessante; altere para `true`
se o perfil real ultrapassar a memória disponível.

O launcher MI300X define:

```bash
TORCH_NCCL_HIGH_PRIORITY=1
GPU_MAX_HW_QUEUES=2
```

Esses valores seguem a orientação atual da AMD para melhorar a sobreposição entre streams de compute e RCCL. Em nós MI300X com oito GPUs, a própria AMD recomenda preferir coletivas com todas as oito GPUs.

## Oito GPUs NVIDIA

```bash
./scripts/launch_nvidia_8gpu.sh \
  --config training_profiles/350m_8x_nvidia.yaml
```

O código não usa FlashAttention externo, Apex, Transformer Engine ou kernels CUDA escritos à mão. SDPA, BF16, AdamW fused, compile e DCP são selecionados pelo PyTorch instalado, preservando o mesmo trainer em AMD e NVIDIA.

Camadas locais tentam FlexAttention automaticamente. Em ROCm e CUDA o mesmo
`BlockMask` causal/sliding-window é usado; se um build específico não oferecer
FlexAttention, defina `model.attention_backend: sdpa`. O modo `flex` torna a
ausência do backend um erro explícito, enquanto `auto` preserva o fallback.

## Múltiplos nós

Em cada nó, ajuste:

```bash
export NNODES=2
export NODE_RANK=0
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
export GPUS_PER_NODE=8

./scripts/launch_multinode.sh \
  --config training_profiles/950m_8x_mi300x.yaml
```

Use `NODE_RANK=1` no segundo nó. O diretório de checkpoints precisa ser visível por todos os ranks.

Para workers reiniciáveis:

```bash
export MIN_NODES=2
export MAX_NODES=2
export GPUS_PER_NODE=8
export RDZV_ENDPOINT=10.0.0.10:29400
export RUN_ID=lunaris-pretrain-001

./scripts/launch_elastic_multinode.sh \
  --config training_profiles/950m_8x_mi300x.yaml
```

Para HSDP em dois nós de oito GPUs, configure
`parallel.fsdp_replicate_group_size: 2`, produzindo uma mesh `[2, 8]`.

## Preflight sem acelerador

```bash
lunaris-preflight \
  --config training_profiles/950m_8x_mi300x.yaml \
  --world-size 16 \
  --require-data \
  --output preflight.json
```

O relatório inclui contagem de parâmetros via device `meta`, budget de tokens,
mesh FSDP/HSDP, manifest leve dos shards e limite inferior da memória
persistente por rank. Ativações e workspaces exigem um smoke test real.

## Tamanho efetivo do batch

\[
B_{\text{global}} =
B_{\text{micro}}
\times N_{\text{acumulação}}
\times N_{\text{ranks}}
\]

Os tokens por atualização são:

\[
N_{\text{tokens/step}} =
B_{\text{global}}
\times L_{\text{sequência}}
\]

Por exemplo, microbatch 1, acumulação 4, oito GPUs e contexto 8192 produzem 262.144 tokens por atualização.

`training.max_tokens` pode limitar uma execução por orçamento de tokens. Quando presente, o trainer deriva o limite real usando o world size atual, sem ultrapassar `training.max_steps`.

## Checkpoints e retomada

Com `checkpoint.resume: auto`, o trainer seleciona apenas o checkpoint mais recente que contém `_SUCCESS`.

Cada checkpoint contém:

- estado canônico e reshardable do modelo;
- estados dos otimizadores;
- configuração completamente resolvida;
- step, microbatches consumidos, cursor global de amostras e tokens vistos;
- RNG local de Python, NumPy e PyTorch por rank;
- estado do GradScaler no modo FP16.

O save assíncrono só publica `_SUCCESS` depois que todos os ranks finalizaram a escrita. Uma interrupção durante a escrita deixa um diretório incompleto, que não é selecionado pelo modo `auto`.

`checkpoint.min_free_space_gib` oferece uma reserva configurável de storage e
interrompe todos os ranks antes de começar um checkpoint quando o espaço livre
é insuficiente.

Para consolidar um checkpoint distribuído em um pacote portátil de inferência:

```bash
lunaris-export \
  --checkpoint checkpoints/lunaris_350m/step_00100000 \
  --output exports/lunaris_350m
```

O diretório exportado usa o mesmo formato de `save_pretrained`, com `config.json`, `model.pt` e uma cópia dos metadados de treinamento.

## Otimizadores e técnicas de 2026

### Currículo IRL + MTP

Os perfis grandes ativam dois heads auxiliares, que preveem os horizontes 2 e
3. Eles compartilham:

- projeções de fusão;
- uma única célula IRL;
- token embedding e LM head do backbone.

O bloco `objectives` controla a introdução gradual:

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

O treino começa com uma iteração IRL e sobe em estágios até
`model.irl_train_iters`. O primeiro horizonte MTP permanece no grafo mesmo
durante o warmup de peso zero; isso mantém todos os parâmetros auxiliares
participando do DDP. Horizontes posteriores entram no intervalo configurado.

`gradient_checkpointing_policy: irl` é o padrão recomendado: atenção não é
recalculada, enquanto os passos IRL do backbone e do MTP são checkpointados.

### Backend da loss

`training.loss_backend: auto` usa
`torch.nn.functional.linear_cross_entropy` com chunking quando a API estiver
presente. Builds mais antigos usam a implementação interna em chunks. O caminho
nativo requer z-loss igual a zero; com z-loss ativada, `auto` escolhe o fallback
para também calcular o log-partition.

Use `chunked` para forçar o caminho mais compatível ou `native` para exigir a
operação fundida e falhar cedo quando ela não estiver disponível.

Em CUDA/ROCm, o fallback em chunks usa compilação regional do kernel
linear+CE+z-loss e pad seguro com `ignore_index` para que todos os chunks tenham
o mesmo formato. Isso evita recompilar cada horizonte MTP ou posição do loop.
Prefira chunks que produzam GEMMs suficientemente grandes; o perfil MI300X
validado usa 4096 linhas por chunk.

FlexAttention também é compilado regionalmente e reutilizado entre camadas
locais. Evite começar um experimento caro com compilação do modelo inteiro:
faça primeiro um probe, pois o currículo IRL/MTP pode especializar múltiplos
grafos.

### Caminho recomendado

O padrão é AdamW, BF16, clipping global e WSD. Esse caminho é conservador, compatível com FSDP2 e adequado para a primeira validação de uma arquitetura nova.

### MONA experimental

`optimizer.name: mona` ativa um otimizador híbrido:

- matrizes internas usam MONA;
- embeddings, LM head, vetores de norm e escalas usam AdamW;
- gradientes completos são sincronizados por DDP antes da ortogonalização;
- weight decay desacoplado e ajuste de escala por formato da matriz são aplicados.

MONA usa uma EMA das diferenças de gradientes antes do momentum e da aproximação Newton-Schulz do fator polar. A opção é bloqueada com FSDP2 porque operar independentemente nos shards não equivale a ortogonalizar a matriz completa.

Trate MONA como uma ablação de pesquisa. Faça primeiro um sweep pequeno contra AdamW e não retome um experimento AdamW como MONA, ou vice-versa.

### Adaptive group clipping

`training.gradient_clip_mode: adaptive_group` separa atenção, IRL-2, embeddings/head e normas/escalas. Cada grupo recebe um limite baseado no histórico EMA do próprio norm. Essa é uma implementação conservadora inspirada no princípio do AGGC de 2026; ela apenas reduz spikes e não amplifica gradientes pequenos.

Como a interpretação de grupos em DTensor ainda introduziria coletivas adicionais, esse modo é restrito a DDP. O padrão continua sendo clipping global.

### ScheduleFree+

O trabalho ScheduleFree+ de 2026 foi considerado, mas não foi reimplementado parcialmente. O método muda a semântica do otimizador, averaging, avaliação e checkpoint. O trainer oferece WSD, cosine e constant como opções confiáveis; uma integração futura de ScheduleFree+ deve usar a implementação oficial dos autores e testes de equivalência.

## Por que não FP8

FP8 não é ativado automaticamente. MI300X e GPUs NVIDIA recentes possuem caminhos FP8 diferentes, com matrizes de suporte e kernels que variam entre versões. BF16 é o denominador comum confiável. Um plugin FP8 pode ser adicionado depois de validar numericamente cada backend, sem alterar o checkpoint BF16 principal.

## Primeira execução real

Antes de uma execução longa:

1. rode 20–100 steps com o perfil final;
2. confirme que loss, gradient norm e memória estabilizam;
3. ative `logging.profile_steps` por poucos steps;
4. compare DDP e FSDP2 no modelo que cabe em memória;
5. teste save, encerramento e resume;
6. só então aumente o orçamento de tokens.

Os valores de LR e batch dos perfis são pontos de partida, não hiperparâmetros universalmente validados para o IRL-2.

## Referências primárias

- [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch FlexAttention](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html)
- [PyTorch Linear Cross Entropy](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.linear_cross_entropy.html)
- [PyTorch guidance for compile with distributed wrappers](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.where_to_apply_compile.html)
- [AMD MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [Curriculum-based Multi-Token Prediction, 2025](https://arxiv.org/abs/2505.22757)
- [MONA: Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training, 2026](https://arxiv.org/abs/2605.26842)
- [AGGC: Adaptive Group Gradient Clipping for Stabilizing Large Language Model Training, 2026](https://arxiv.org/abs/2601.11864)
- [ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models, 2026](https://arxiv.org/abs/2605.19095)
- [Muon is Scalable for LLM Training, 2025](https://arxiv.org/abs/2502.16982)
