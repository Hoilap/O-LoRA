# O-LoRA

- This repo releases our implementation for the O-LoRA model.
- It is built based on the pretrained T5-large model, and finetuned on our data.

![image_text](https://github.com/cmnfriend/O-LoRA/blob/main/data/O-LoRA.jpg)


## Setup

You can install the required libraries by running 

```
pip install -r requirements.txt
```

You are also required to download the t5-large model from huggingface, put it to the folder named ```initial_model```, and rename the model folder as 't5-large'.

LLaMA2 HF is also supported. You can put your llama2 hf model to the folder named ```initial_model``` and rename the model folder as 'llama'.


## Training and Evaluation

For t5-large:

You can reproduce our experiments of order 1 & 2 & 3 by simply running

order1:

```
bash scripts/order_1.sh> logs_and_outputs/order_1/logs/train_and_infer.log 2>&1 &
```

order2:

```
bash scripts/order_2.sh> logs_and_outputs/order_2/logs/train_and_infer.log 2>&1 &
```

order3:

```
bash scripts/order_3.sh> logs_and_outputs/order_3/logs/train_and_infer.log 2>&1 &
```

The model you have trained will be saved in ```logs_and_outputs/order_1(2 or 3)/outputs```.

The result of each task will be saved in ```logs_and_outputs/order_1(2 or 3)/outputs/TASK_NAME/predict_results.json```.

You can also check the logs during training and infering in  ```logs_and_outputs/order_1(2 or 3)/logs/train_and_infer.log```

For LLaMA2:

order1:

```
bash scripts_llama/order_1.sh> logs_and_outputs_llama/order_1/logs/train_and_infer.log 2>&1 &
```

order2:

```
bash scripts_llama/order_2.sh> logs_and_outputs_llama/order_2/logs/train_and_infer.log 2>&1 &
```

order3:

```
bash scripts_llama/order_3.sh> logs_and_outputs_llama/order_3/logs/train_and_infer.log 2>&1 &
```
## Timestamp
O-lora复现成功，单卡，


这个警告 ProcessGroupNCCL.cpp:4561 ... using GPU 0 to perform barrier as devices used by this process are currently unknown 通常出现在 PyTorch 分布式初始化（DDP）早期，当进程组尚未完全建立或设备映射尚未明确时调用了 barrier()。

在您的代码中，这很可能是因为我在上一步添加的 training_args.main_process_first 上下文管理器内部调用了 barrier()，而此时 DeepSpeed 或 PyTorch 的分布式环境可能还没有完全初始化好设备绑定。

问题分析：
training_args.main_process_first 依赖于 torch.distributed。如果它在 deepspeed 初始化之前被调用，或者在设备（cuda:0, cuda:1）被显式设置之前被调用，就可能触发这个警告，甚至导致死锁（Hang）。

解决方案：
我们需要确保在调用 main_process_first 之前，分布式环境已经正确初始化，并且当前进程已经绑定到了正确的 GPU 设备上。

在 run_uie_lora.py 中，UIETrainingArguments 继承自 Seq2SeqTrainingArguments，而 Seq2SeqTrainingArguments 继承自 TrainingArguments。TrainingArguments 的初始化通常会处理分布式设置。

但是，为了稳妥起见，我们可以显式地设置当前设备的 ID。

让我们检查一下 run_uie_lora.py 的开头部分。

training_args.device 的访问应该会触发设备初始化。

然而，警告信息提示 devices used by this process are currently unknown。这表明 torch.cuda.set_device(local_rank) 可能还没有被调用。

我建议在 main 函数的早期，在解析完参数后，显式调用 torch.cuda.set_device。

此外，为了避免 main_process_first 带来的潜在死锁（如果它内部的 barrier 实现有问题），我们可以用一种更显式、更安全的方式来替代它：

但是 load_dataset 内部有文件锁机制，理论上是安全的。问题在于我们之前加的 main_process_first 可能引入了额外的 barrier。

让我们先尝试显式设置设备，这通常能解决 ProcessGroupNCCL 的那个警告。

同时，我会检查 main_process_first 的位置。

如果 training_args.device 已经被访问过（在 logger.warning 中），那么 training_args 内部应该已经初始化了分布式环境。

让我们尝试在 run_uie_lora.py 中添加 torch.cuda.set_device。

另外，还有一个可能性：deepspeed 启动器会自动设置 LOCAL_RANK 环境变量，但 transformers 的 TrainingArguments 可能需要显式传入 local_rank 参数（虽然通常它能自动检测）。



```
[rank0]: AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 2
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/dengkn/O-LoRA/src/run_uie_lora.py", line 599, in <module>
[rank1]:     main()
[rank1]:   File "/home/dengkn/O-LoRA/src/run_uie_lora.py", line 539, in main
[rank1]:     train_result = trainer.train(resume_from_checkpoint=checkpoint)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/trainer.py", line 1591, in train
[rank1]:     return inner_training_loop(
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/trainer.py", line 1891, in _inner_training_loop
[rank1]:     with self.accelerator.accumulate(model):
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/contextlib.py", line 119, in enter
[rank1]:     return next(self.gen)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/accelerate/accelerator.py", line 1057, in accumulate
[rank1]:     cm_stack.enter_context(contextlib.nullcontext() if self.sync_gradients else self.no_sync(m))
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/contextlib.py", line 448, in enter_context
[rank1]:     result = _cm_type.enter(cm)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/contextlib.py", line 119, in enter
[rank1]:     return next(self.gen)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/accelerate/accelerator.py", line 947, in no_sync
[rank1]:     with context():
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/contextlib.py", line 119, in enter
[rank1]:     return next(self.gen)
[rank1]:   File "/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/runtime/engine.py", line 2397, in no_sync
[rank1]:     assert not self.zero_optimization_partition_gradients(), 
[rank1]: AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 2
```

简单来说： 你在训练配置中开启了 DeepSpeed ZeRO Stage 2，同时又开启了 梯度累积（Gradient Accumulation）。
DeepSpeed ZeRO Stage 2：会对梯度（Gradients）进行切分（Partitioning），分布在不同的 GPU 上以节省显存。
no_sync：这是 PyTorch DDP 的一种机制，用于在梯度累积的前几步暂停多卡间的梯度同步，以减少通信开销。
冲突点：ZeRO Stage 2 要求梯度必须是切分状态，而 no_sync 上下文管理器试图以一种 ZeRO 2 不支持的方式处理梯度。DeepSpeed 为了保证数据正确性，显式地禁止了在 ZeRO 2 下使用 no_sync，从而抛出了这个 AssertionError。


accelerate               0.30.0
deepspeed                0.18.3
transformers             4.34.0


下载了指定deepspeed后解决

```
+ export CUDA_DEVICE_ORDER=PCI_BUS_ID
+ CUDA_DEVICE_ORDER=PCI_BUS_ID
+ export TRANSFORMERS_CACHE=/home/dengkn/.cache/huggingface
+ TRANSFORMERS_CACHE=/home/dengkn/.cache/huggingface
++ shuf -i25000-30000 -n1
+ port=27244
+ CUDA_VISIBLE_DEVICES=0,1
+ deepspeed --master_port 27244 src/run_uie_lora.py --do_train --do_predict --predict_with_generate --model_name_or_path /home/dengkn/N-LoRA/initial_model/llama --data_dir CL_Benchmark --task_config_dir configs/order1_configs/dbpedia --instruction_file configs/instruction_config.json --instruction_strategy single --output_dir logs_and_outputs_llama/order_1/outputs/1-dbpedia --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-03 --num_train_epochs 1 --deepspeed configs/ds_configs/stage2_llama.config --run_name order1_round1 --max_source_length 512 --max_target_length 50 --generation_max_length 50 --add_task_name True --add_dataset_name True --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 10 --evaluation_strategy no --save_strategy no --save_steps 1500 --lamda_1 0.5 --lamda_2 0
[2025-12-18 20:16:22,490] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
[2025-12-18 20:16:24,212] [WARNING] [runner.py:215:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
Detected VISIBLE_DEVICES=0,1: setting --include=localhost:0,1
[2025-12-18 20:16:24,213] [INFO] [runner.py:607:main] cmd = /home/dengkn/miniforge3/envs/aslora/bin/python3.9 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMV19 --master_addr=127.0.0.1 --master_port=27244 --enable_each_rank_log=None src/run_uie_lora.py --do_train --do_predict --predict_with_generate --model_name_or_path /home/dengkn/N-LoRA/initial_model/llama --data_dir CL_Benchmark --task_config_dir configs/order1_configs/dbpedia --instruction_file configs/instruction_config.json --instruction_strategy single --output_dir logs_and_outputs_llama/order_1/outputs/1-dbpedia --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-03 --num_train_epochs 1 --deepspeed configs/ds_configs/stage2_llama.config --run_name order1_round1 --max_source_length 512 --max_target_length 50 --generation_max_length 50 --add_task_name True --add_dataset_name True --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 10 --evaluation_strategy no --save_strategy no --save_steps 1500 --lamda_1 0.5 --lamda_2 0
[2025-12-18 20:16:25,391] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
[2025-12-18 20:16:27,068] [INFO] [launch.py:146:main] WORLD INFO DICT: {'localhost': [0, 1]}
[2025-12-18 20:16:27,068] [INFO] [launch.py:152:main] nnodes=1, num_local_procs=2, node_rank=0
[2025-12-18 20:16:27,068] [INFO] [launch.py:163:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1]})
[2025-12-18 20:16:27,068] [INFO] [launch.py:164:main] dist_world_size=2
[2025-12-18 20:16:27,068] [INFO] [launch.py:168:main] Setting CUDA_VISIBLE_DEVICES=0,1
[2025-12-18 20:16:27,089] [INFO] [launch.py:256:main] process 748089 spawned with command: ['/home/dengkn/miniforge3/envs/aslora/bin/python3.9', '-u', 'src/run_uie_lora.py', '--local_rank=0', '--do_train', '--do_predict', '--predict_with_generate', '--model_name_or_path', '/home/dengkn/N-LoRA/initial_model/llama', '--data_dir', 'CL_Benchmark', '--task_config_dir', 'configs/order1_configs/dbpedia', '--instruction_file', 'configs/instruction_config.json', '--instruction_strategy', 'single', '--output_dir', 'logs_and_outputs_llama/order_1/outputs/1-dbpedia', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '4', '--gradient_accumulation_steps', '8', '--learning_rate', '1e-03', '--num_train_epochs', '1', '--deepspeed', 'configs/ds_configs/stage2_llama.config', '--run_name', 'order1_round1', '--max_source_length', '512', '--max_target_length', '50', '--generation_max_length', '50', '--add_task_name', 'True', '--add_dataset_name', 'True', '--overwrite_output_dir', '--overwrite_cache', '--lr_scheduler_type', 'constant', '--warmup_steps', '0', '--logging_strategy', 'steps', '--logging_steps', '10', '--evaluation_strategy', 'no', '--save_strategy', 'no', '--save_steps', '1500', '--lamda_1', '0.5', '--lamda_2', '0']
[2025-12-18 20:16:27,104] [INFO] [launch.py:256:main] process 748090 spawned with command: ['/home/dengkn/miniforge3/envs/aslora/bin/python3.9', '-u', 'src/run_uie_lora.py', '--local_rank=1', '--do_train', '--do_predict', '--predict_with_generate', '--model_name_or_path', '/home/dengkn/N-LoRA/initial_model/llama', '--data_dir', 'CL_Benchmark', '--task_config_dir', 'configs/order1_configs/dbpedia', '--instruction_file', 'configs/instruction_config.json', '--instruction_strategy', 'single', '--output_dir', 'logs_and_outputs_llama/order_1/outputs/1-dbpedia', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '4', '--gradient_accumulation_steps', '8', '--learning_rate', '1e-03', '--num_train_epochs', '1', '--deepspeed', 'configs/ds_configs/stage2_llama.config', '--run_name', 'order1_round1', '--max_source_length', '512', '--max_target_length', '50', '--generation_max_length', '50', '--add_task_name', 'True', '--add_dataset_name', 'True', '--overwrite_output_dir', '--overwrite_cache', '--lr_scheduler_type', 'constant', '--warmup_steps', '0', '--logging_strategy', 'steps', '--logging_steps', '10', '--evaluation_strategy', 'no', '--save_strategy', 'no', '--save_steps', '1500', '--lamda_1', '0.5', '--lamda_2', '0']
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/fairscale/experimental/nn/offload.py:19: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  return torch.cuda.amp.custom_fwd(orig_func)  # type: ignore
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/fairscale/experimental/nn/offload.py:30: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  return torch.cuda.amp.custom_bwd(orig_func)  # type: ignore
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/fairscale/experimental/nn/offload.py:19: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  return torch.cuda.amp.custom_fwd(orig_func)  # type: ignore
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/fairscale/experimental/nn/offload.py:30: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  return torch.cuda.amp.custom_bwd(orig_func)  # type: ignore
[2025-12-18 20:16:29,874] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-12-18 20:16:29,904] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-12-18 20:16:31,005] [INFO] [comm.py:652:init_distributed] cdb=None
[2025-12-18 20:16:31,005] [INFO] [comm.py:652:init_distributed] cdb=None
[2025-12-18 20:16:31,005] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
12/18/2025 20:16:31 - WARNING - __main__ - Process rank: 0, device: cuda:0, n_gpu: 1distributed training: True, 16-bits training: False
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[WARNING|logging.py:290] 2025-12-18 20:16:31,452 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
12/18/2025 20:16:31 - WARNING - __main__ - Process rank: 1, device: cuda:1, n_gpu: 1distributed training: True, 16-bits training: False
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
[WARNING|logging.py:290] 2025-12-18 20:16:31,578 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.

Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]
Loading checkpoint shards:  33%|███▎      | 1/3 [00:00<00:01,  1.23it/s]
Loading checkpoint shards:  33%|███▎      | 1/3 [00:01<00:02,  1.03s/it]
Loading checkpoint shards:  67%|██████▋   | 2/3 [00:01<00:00,  1.43it/s]
Loading checkpoint shards:  67%|██████▋   | 2/3 [00:01<00:00,  1.29it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:01<00:00,  1.63it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:01<00:00,  1.54it/s]
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/nn/init.py:511: UserWarning: Initializing zero-element tensors is a no-op
  warnings.warn("Initializing zero-element tensors is a no-op")

Loading checkpoint shards: 100%|██████████| 3/3 [00:02<00:00,  1.65it/s]
Loading checkpoint shards: 100%|██████████| 3/3 [00:02<00:00,  1.48it/s]
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/nn/init.py:511: UserWarning: Initializing zero-element tensors is a no-op
  warnings.warn("Initializing zero-element tensors is a no-op")
trainable params: 0 || all params: 6742609920 || trainable%: 0.0
=======================================
trainable params: 4194304 || all params: 6742609920 || trainable%: 0.06220594176090199
-----Gradient checkpointing: False -----
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/accelerate/accelerator.py:446: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
dataloader_config = DataLoaderConfiguration(dispatch_batches=None)
  warnings.warn(
trainable params: 0 || all params: 6742609920 || trainable%: 0.0
=======================================
trainable params: 4194304 || all params: 6742609920 || trainable%: 0.06220594176090199
-----Gradient checkpointing: False -----
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/accelerate/accelerator.py:446: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
dataloader_config = DataLoaderConfiguration(dispatch_batches=None)
  warnings.warn(
Installed CUDA version 12.6 does not match the version torch was compiled with 12.4 but since the APIs are compatible, accepting this combination
Using /home/dengkn/.cache/torch_extensions/py39_cu124 as PyTorch extensions root...
Installed CUDA version 12.6 does not match the version torch was compiled with 12.4 but since the APIs are compatible, accepting this combination
Using /home/dengkn/.cache/torch_extensions/py39_cu124 as PyTorch extensions root...
Emitting ninja build file /home/dengkn/.cache/torch_extensions/py39_cu124/cpu_adam/build.ninja...
Building extension module cpu_adam...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/3] c++ -MMD -MF cpu_adam.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/ops/csrc/includes -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/TH -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/THC -isystem /home/dengkn/miniforge3/envs/aslora/include/python3.9 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX512__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/ops/csrc/adam/cpu_adam.cpp -o cpu_adam.o 
[2/3] c++ -MMD -MF cpu_adam_impl.o.d -DTORCH_EXTENSION_NAME=cpu_adam -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -I/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/ops/csrc/includes -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/TH -isystem /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/include/THC -isystem /home/dengkn/miniforge3/envs/aslora/include/python3.9 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -O3 -std=c++17 -g -Wno-reorder -L/usr/local/cuda/lib64 -lcudart -lcublas -g -march=native -fopenmp -D__AVX512__ -D__ENABLE_CUDA__ -DBF16_AVAILABLE -c /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/deepspeed/ops/csrc/adam/cpu_adam_impl.cpp -o cpu_adam_impl.o 
[3/3] c++ cpu_adam.o cpu_adam_impl.o -shared -lcurand -L/usr/local/cuda/lib64 -L/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o cpu_adam.so
Loading extension module cpu_adam...
Time to load cpu_adam op: 22.258127450942993 seconds
Loading extension module cpu_adam...
Time to load cpu_adam op: 22.287405014038086 seconds
[rank0]:[E1218 20:27:21.426415540 ProcessGroupNCCL.cpp:629] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600081 milliseconds before timing out.
[rank1]:[E1218 20:27:21.433063423 ProcessGroupNCCL.cpp:629] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600088 milliseconds before timing out.
[rank1]:[E1218 20:27:21.484903450 ProcessGroupNCCL.cpp:2168] [PG ID 1 PG GUID 1 Rank 1]  failure detected by watchdog at work sequence id: 1 PG status: last enqueued work: 547, last completed work: -1
[rank0]:[E1218 20:27:21.484910731 ProcessGroupNCCL.cpp:2168] [PG ID 1 PG GUID 1 Rank 0]  failure detected by watchdog at work sequence id: 1 PG status: last enqueued work: 547, last completed work: -1
[rank1]:[E1218 20:27:21.484941868 ProcessGroupNCCL.cpp:667] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[rank0]:[E1218 20:27:21.484955368 ProcessGroupNCCL.cpp:667] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[rank1]:[E1218 20:27:21.484959605 ProcessGroupNCCL.cpp:681] [Rank 1] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank0]:[E1218 20:27:21.484963240 ProcessGroupNCCL.cpp:681] [Rank 0] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank1]:[E1218 20:27:21.484966856 ProcessGroupNCCL.cpp:695] [Rank 1] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E1218 20:27:21.484970331 ProcessGroupNCCL.cpp:695] [Rank 0] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E1218 20:27:21.486305774 ProcessGroupNCCL.cpp:1895] [PG ID 1 PG GUID 1 Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600081 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7b99599e31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x7b9906ffec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x7b99070007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7b99070016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7b9959b425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7b995a69caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7b995a729c6c in /lib/x86_64-linux-gnu/libc.so.6)

[rank1]:[E1218 20:27:21.486311443 ProcessGroupNCCL.cpp:1895] [PG ID 1 PG GUID 1 Rank 1] Process group watchdog thread terminated with exception: [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600088 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7ca805fe31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x7ca7b35fec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x7ca7b36007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7ca7b36016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7ca8061425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7ca806c9caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7ca806d29c6c in /lib/x86_64-linux-gnu/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG ID 1 PG GUID 1 Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600081 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7b99599e31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x7b9906ffec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x7b99070007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7b99070016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7b9959b425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7b995a69caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7b995a729c6c in /lib/x86_64-linux-gnu/libc.so.6)

Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1901 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7b99599e31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xe5c6fc (0x7b9906c5c6fc in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0x145c0 (0x7b9959b425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #3: <unknown function> + 0x9caa4 (0x7b995a69caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #4: <unknown function> + 0x129c6c (0x7b995a729c6c in /lib/x86_64-linux-gnu/libc.so.6)

  what():  [PG ID 1 PG GUID 1 Rank 1] Process group watchdog thread terminated with exception: [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600088 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7ca805fe31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x7ca7b35fec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x7ca7b36007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x7ca7b36016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7ca8061425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7ca806c9caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7ca806d29c6c in /lib/x86_64-linux-gnu/libc.so.6)

Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1901 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7ca805fe31b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xe5c6fc (0x7ca7b325c6fc in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0x145c0 (0x7ca8061425c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #3: <unknown function> + 0x9caa4 (0x7ca806c9caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #4: <unknown function> + 0x129c6c (0x7ca806d29c6c in /lib/x86_64-linux-gnu/libc.so.6)

[2025-12-18 20:27:22,783] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 748089
[2025-12-18 20:27:22,783] [INFO] [launch.py:319:sigkill_handler] Killing subprocess 748090
[2025-12-18 20:27:23,533] [ERROR] [launch.py:325:sigkill_handler] ['/home/dengkn/miniforge3/envs/aslora/bin/python3.9', '-u', 'src/run_uie_lora.py', '--local_rank=1', '--do_train', '--do_predict', '--predict_with_generate', '--model_name_or_path', '/home/dengkn/N-LoRA/initial_model/llama', '--data_dir', 'CL_Benchmark', '--task_config_dir', 'configs/order1_configs/dbpedia', '--instruction_file', 'configs/instruction_config.json', '--instruction_strategy', 'single', '--output_dir', 'logs_and_outputs_llama/order_1/outputs/1-dbpedia', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '4', '--gradient_accumulation_steps', '8', '--learning_rate', '1e-03', '--num_train_epochs', '1', '--deepspeed', 'configs/ds_configs/stage2_llama.config', '--run_name', 'order1_round1', '--max_source_length', '512', '--max_target_length', '50', '--generation_max_length', '50', '--add_task_name', 'True', '--add_dataset_name', 'True', '--overwrite_output_dir', '--overwrite_cache', '--lr_scheduler_type', 'constant', '--warmup_steps', '0', '--logging_strategy', 'steps', '--logging_steps', '10', '--evaluation_strategy', 'no', '--save_strategy', 'no', '--save_steps', '1500', '--lamda_1', '0.5', '--lamda_2', '0'] exits with return code = -6
+ --disable_tqdm True
scripts_llama/order_1.sh: line 45: --disable_tqdm: command not found
+ sleep 5
+ CUDA_VISIBLE_DEVICES=0,1
+ deepspeed --master_port 27244 src/run_uie_lora.py --do_train --do_predict --predict_with_generate --model_name_or_path logs_and_outputs_llama/order_1/outputs/1-dbpedia/adapter --data_dir CL_Benchmark --task_config_dir configs/order1_configs/amazon --instruction_file configs/instruction_config.json --instruction_strategy single --output_dir logs_and_outputs_llama/order_1/outputs/2-amazon --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-04 --num_train_epochs 1 --deepspeed configs/ds_configs/stage2_llama.config --run_name order1_round2 --max_source_length 512 --max_target_length 50 --generation_max_length 50 --add_task_name True --add_dataset_name True --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 10 --evaluation_strategy no --save_strategy no --save_steps 1500 --lamda_1 0.5 --lamda_2 0 --disable_tqdm True
[2025-12-18 20:27:30,944] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
[2025-12-18 20:27:32,639] [WARNING] [runner.py:215:fetch_hostfile] Unable to find hostfile, will proceed with training with local resources only.
Detected VISIBLE_DEVICES=0,1: setting --include=localhost:0,1
[2025-12-18 20:27:32,639] [INFO] [runner.py:607:main] cmd = /home/dengkn/miniforge3/envs/aslora/bin/python3.9 -u -m deepspeed.launcher.launch --world_info=eyJsb2NhbGhvc3QiOiBbMCwgMV19 --master_addr=127.0.0.1 --master_port=27244 --enable_each_rank_log=None src/run_uie_lora.py --do_train --do_predict --predict_with_generate --model_name_or_path logs_and_outputs_llama/order_1/outputs/1-dbpedia/adapter --data_dir CL_Benchmark --task_config_dir configs/order1_configs/amazon --instruction_file configs/instruction_config.json --instruction_strategy single --output_dir logs_and_outputs_llama/order_1/outputs/2-amazon --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-04 --num_train_epochs 1 --deepspeed configs/ds_configs/stage2_llama.config --run_name order1_round2 --max_source_length 512 --max_target_length 50 --generation_max_length 50 --add_task_name True --add_dataset_name True --overwrite_output_dir --overwrite_cache --lr_scheduler_type constant --warmup_steps 0 --logging_strategy steps --logging_steps 10 --evaluation_strategy no --save_strategy no --save_steps 1500 --lamda_1 0.5 --lamda_2 0 --disable_tqdm True
[2025-12-18 20:27:33,718] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
[2025-12-18 20:27:35,366] [INFO] [launch.py:146:main] WORLD INFO DICT: {'localhost': [0, 1]}
[2025-12-18 20:27:35,366] [INFO] [launch.py:152:main] nnodes=1, num_local_procs=2, node_rank=0
[2025-12-18 20:27:35,366] [INFO] [launch.py:163:main] global_rank_mapping=defaultdict(<class 'list'>, {'localhost': [0, 1]})
[2025-12-18 20:27:35,366] [INFO] [launch.py:164:main] dist_world_size=2
[2025-12-18 20:27:35,366] [INFO] [launch.py:168:main] Setting CUDA_VISIBLE_DEVICES=0,1
[2025-12-18 20:27:35,388] [INFO] [launch.py:256:main] process 749021 spawned with command: ['/home/dengkn/miniforge3/envs/aslora/bin/python3.9', '-u', 'src/run_uie_lora.py', '--local_rank=0', '--do_train', '--do_predict', '--predict_with_generate', '--model_name_or_path', 'logs_and_outputs_llama/order_1/outputs/1-dbpedia/adapter', '--data_dir', 'CL_Benchmark', '--task_config_dir', 'configs/order1_configs/amazon', '--instruction_file', 'configs/instruction_config.json', '--instruction_strategy', 'single', '--output_dir', 'logs_and_outputs_llama/order_1/outputs/2-amazon', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '4', '--gradient_accumulation_steps', '8', '--learning_rate', '1e-04', '--num_train_epochs', '1', '--deepspeed', 'configs/ds_configs/stage2_llama.config', '--run_name', 'order1_round2', '--max_source_length', '512', '--max_target_length', '50', '--generation_max_length', '50', '--add_task_name', 'True', '--add_dataset_name', 'True', '--overwrite_output_dir', '--overwrite_cache', '--lr_scheduler_type', 'constant', '--warmup_steps', '0', '--logging_strategy', 'steps', '--logging_steps', '10', '--evaluation_strategy', 'no', '--save_strategy', 'no', '--save_steps', '1500', '--lamda_1', '0.5', '--lamda_2', '0', '--disable_tqdm', 'True']
[2025-12-18 20:27:35,402] [INFO] [launch.py:256:main] process 749022 spawned with command: ['/home/dengkn/miniforge3/envs/aslora/bin/python3.9', '-u', 'src/run_uie_lora.py', '--local_rank=1', '--do_train', '--do_predict', '--predict_with_generate', '--model_name_or_path', 'logs_and_outputs_llama/order_1/outputs/1-dbpedia/adapter', '--data_dir', 'CL_Benchmark', '--task_config_dir', 'configs/order1_configs/amazon', '--instruction_file', 'configs/instruction_config.json', '--instruction_strategy', 'single', '--output_dir', 'logs_and_outputs_llama/order_1/outputs/2-amazon', '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '4', '--gradient_accumulation_steps', '8', '--learning_rate', '1e-04', '--num_train_epochs', '1', '--deepspeed', 'configs/ds_configs/stage2_llama.config', '--run_name', 'order1_round2', '--max_source_length', '512', '--max_target_length', '50', '--generation_max_length', '50', '--add_task_name', 'True', '--add_dataset_name', 'True', '--overwrite_output_dir', '--overwrite_cache', '--lr_scheduler_type', 'constant', '--warmup_steps', '0', '--logging_strategy', 'steps', '--logging_steps', '10', '--evaluation_strategy', 'no', '--save_strategy', 'no', '--save_steps', '1500', '--lamda_1', '0.5', '--lamda_2', '0', '--disable_tqdm', 'True']
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/transformers/utils/generic.py:311: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  torch.utils._pytree._register_pytree_node(
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/fairscale/experimental/nn/offload.py:19: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  return torch.cuda.amp.custom_fwd(orig_func)  # type: ignore
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/fairscale/experimental/nn/offload.py:30: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  return torch.cuda.amp.custom_bwd(orig_func)  # type: ignore
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/fairscale/experimental/nn/offload.py:19: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  return torch.cuda.amp.custom_fwd(orig_func)  # type: ignore
/home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/fairscale/experimental/nn/offload.py:30: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  return torch.cuda.amp.custom_bwd(orig_func)  # type: ignore
[2025-12-18 20:27:38,112] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-12-18 20:27:38,216] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2025-12-18 20:27:39,316] [INFO] [comm.py:652:init_distributed] cdb=None
[2025-12-18 20:27:39,316] [INFO] [comm.py:683:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
[2025-12-18 20:27:39,346] [INFO] [comm.py:652:init_distributed] cdb=None
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
12/18/2025 20:27:39 - WARNING - __main__ - Process rank: 0, device: cuda:0, n_gpu: 1distributed training: True, 16-bits training: False
[rank0]:[W1218 20:27:39.903177285 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
12/18/2025 20:27:39 - WARNING - __main__ - Process rank: 1, device: cuda:1, n_gpu: 1distributed training: True, 16-bits training: False
[rank1]:[W1218 20:27:39.932905430 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. Specify device_ids in barrier() to force use of a particular device, or call init_process_group() with a device_id.

```
关闭offload

```
[rank0]:[E1218 20:43:21.166114205 ProcessGroupNCCL.cpp:629] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600036 milliseconds before timing out.
[rank0]:[E1218 20:43:21.166281407 ProcessGroupNCCL.cpp:2168] [PG ID 1 PG GUID 1 Rank 0]  failure detected by watchdog at work sequence id: 1 PG status: last enqueued work: 547, last completed work: -1
[rank0]:[E1218 20:43:21.166297611 ProcessGroupNCCL.cpp:667] Stack trace of the failed collective not found, potentially because FlightRecorder is disabled. You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.
[rank0]:[E1218 20:43:21.166304091 ProcessGroupNCCL.cpp:681] [Rank 0] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
[rank0]:[E1218 20:43:21.166309339 ProcessGroupNCCL.cpp:695] [Rank 0] To avoid data inconsistency, we are taking the entire process down.
[rank0]:[E1218 20:43:21.167223203 ProcessGroupNCCL.cpp:1895] [PG ID 1 PG GUID 1 Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600036 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7564c37d71b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x756470dfec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x756470e007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x756470e016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7564c39365c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7564c449caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7564c4529c6c in /lib/x86_64-linux-gnu/libc.so.6)

terminate called after throwing an instance of 'c10::DistBackendError'
  what():  [PG ID 1 PG GUID 1 Rank 0] Process group watchdog thread terminated with exception: [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=BROADCAST, NumelIn=131072000, NumelOut=131072000, Timeout(ms)=600000) ran for 600036 milliseconds before timing out.
Exception raised from checkTimeout at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:632 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7564c37d71b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout(std::optional<std::chrono::duration<long, std::ratio<1l, 1000l> > >) + 0x2b4 (0x756470dfec74 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: c10d::ProcessGroupNCCL::watchdogHandler() + 0x890 (0x756470e007d0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #3: c10d::ProcessGroupNCCL::ncclCommWatchdog() + 0x14d (0x756470e016ed in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #4: <unknown function> + 0x145c0 (0x7564c39365c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #5: <unknown function> + 0x9caa4 (0x7564c449caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #6: <unknown function> + 0x129c6c (0x7564c4529c6c in /lib/x86_64-linux-gnu/libc.so.6)

Exception raised from ncclCommWatchdog at /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1901 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x96 (0x7564c37d71b6 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libc10.so)
frame #1: <unknown function> + 0xe5c6fc (0x756470a5c6fc in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so)
frame #2: <unknown function> + 0x145c0 (0x7564c39365c0 in /home/dengkn/miniforge3/envs/aslora/lib/python3.9/site-packages/torch/lib/libtorch.so)
frame #3: <unknown function> + 0x9caa4 (0x7564c449caa4 in /lib/x86_64-linux-gnu/libc.so.6)
frame #4: <unknown function> + 0x129c6c (0x7564c4529c6c in /lib/x86_64-linux-gnu/libc.so.6)

```
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
双卡训练成功，但是推理出现问题




## Citation
```latex
@article{wang2023orthogonal,
  title={Orthogonal Subspace Learning for Language Model Continual Learning},
  author={Wang, Xiao and Chen, Tianze and Ge, Qiming and Xia, Han and Bao, Rong and Zheng, Rui and Zhang, Qi and Gui, Tao and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2310.14152},
  year={2023}
}
```


