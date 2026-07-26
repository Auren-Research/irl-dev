# Lunaris Dense IRL-2 v0.4.0

Esta versão fortalece o pipeline de pretraining sem alterar o núcleo
arquitetural ou a formulação IRL-2. Checkpoints de modelo continuam usando o
mesmo formato arquitetural; o metadata do trainer passou para a versão 2.

## O que mudou

- compilação regional passou a ser uma escolha explícita e o default;
- compilação do modelo inteiro ficou desativada nos perfis fornecidos;
- FSDP2 aceita mesh 1D e HSDP mesh 2D;
- retomada usa cursor global de amostras independente do world size;
- DCP continua redistribuindo modelo e otimizador entre topologias;
- RNG por rank é restaurado com a mesma topologia e reseeded quando ela muda;
- launcher `torchrun` elástico foi adicionado;
- checkpoint pode exigir uma reserva mínima de espaço livre;
- `lunaris-preflight` inspeciona um plano sem alocar o modelo;
- todos os perfis grandes foram migrados para o caminho regional validado.

## Decisões conservadoras

| Técnica | Estado | Decisão |
|---|---|---|
| BF16 + AdamW | default | baseline de qualidade |
| MTP em currículo | ativo | preservado ao redor do IRL-2 |
| compilação regional | default | caminho medido na MI300X |
| FSDP2 | disponível | recomendado quando DDP replica memória demais |
| HSDP | opt-in | útil em múltiplos nós após medir a rede |
| async DCP | opt-in/default distribuído | reduz tempo no caminho crítico |
| MONA/Muon | experimental | não usar no primeiro pretraining grande |
| FP8 | não habilitado | requer paridade numérica separada por backend |
| Tensor/Context Parallel | adiado | só justificar por escala/contexto real |
| Pipeline Parallel | adiado | API e scheduling ainda elevam a superfície de risco |

Muon possui resultados promissores de eficiência, mas trocar o otimizador
muda a trajetória de treinamento e exige sweep de learning rate e validação de
convergência. FP8 também pode melhorar throughput, porém não deve ser adotado
somente com base em um smoke test curto.

## Validação realizada sem GPU

- suíte completa em PyTorch 2.13 CPU;
- forward e backward da arquitetura, IRL-2 e MTP;
- equivalência da loss chunked;
- checkpoint, resume e export;
- retomada do sampler após mudança de world size;
- resolução de mesh FSDP/HSDP;
- preflight do perfil 950M sem alocar pesos;
- validação estrita dos nove perfis YAML;
- `ruff check` sem erros;
- compilação sintática de todos os módulos e testes.

CPU valida lógica, formatos e invariantes; não valida kernels ROCm/CUDA,
coletivas RCCL/NCCL, memória real, compile ou throughput.

## Próximos gates

### GPU gratuita com tempo limitado

1. executar 5 steps FP32 ou BF16 sem compile;
2. executar 20 steps BF16 com compilação regional;
3. salvar, encerrar e retomar por 5 steps;
4. comparar loss inicial e gradientes com regional compile ligado/desligado;
5. registrar versão de PyTorch, backend, tokens/s e pico de memória.

Esse gate detecta incompatibilidades gerais CUDA/ROCm, mas não qualifica
performance de MI300X.

### Quando houver MI300X novamente

1. repetir o perfil 135M já medido para detectar regressão contra v0.3.2;
2. qualificar uma e oito GPUs separadamente;
3. comparar DDP e FSDP2 quando ambos couberem;
4. testar checkpoint durante carga real e retomada;
5. testar HSDP apenas em múltiplos nós;
6. exigir uma janela longa de losses finitas antes do pretraining principal.

## Referências primárias

- [PyTorch FSDP2/HSDP](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [PyTorch async DCP](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [Pre-Training Curriculum for Multi-Token Prediction](https://arxiv.org/abs/2505.22757)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
