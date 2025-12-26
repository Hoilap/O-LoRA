import torch
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback
import os
from transformers.integrations import is_deepspeed_zero3_enabled

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX


def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:

            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions

    return final_predictions


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control


class LrAndGradLogCallback(TrainerCallback):
    """Log actual optimizer lrs, scheduler last lr, and LoRA param update.

    Enable by setting env UIE_LOG_LR=1. Logs every args.logging_steps steps.
    """

    def __init__(self):
        super().__init__()
        self.enabled = os.environ.get("UIE_LOG_LR", "0") == "1"
        self.track_name = None
        self.prev_norm = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled:
            return
        model = kwargs.get("model", None)
        if model is None:
            return
        # pick first trainable LoRA-new param
        for n, p in model.named_parameters():
            if p.requires_grad and ("loranew_B" in n or "lora_" in n):
                self.track_name = n
                self.prev_norm = p.data.float().norm().item()
                logger.info(f"[LR Debug] tracking param: {self.track_name}, init_norm={self.prev_norm:.6f}")
                break

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled:
            return control

        # throttle by logging_steps (and also log very early steps)
        log_now = False
        if args.logging_strategy == IntervalStrategy.STEPS and args.logging_steps:
            if state.global_step <= 5 or state.global_step % args.logging_steps == 0:
                log_now = True
        else:
            log_now = state.global_step <= 5

        if not log_now:
            return control

        optimizer = kwargs.get("optimizer", None)
        lr_scheduler = kwargs.get("lr_scheduler", None)
        model = kwargs.get("model", None)

        # current lrs actually on optimizer param groups
        current_lrs = []
        try:
            if optimizer is not None and hasattr(optimizer, "param_groups"):
                current_lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
        except Exception:
            pass

        # scheduler last lr (represents lr computed for next step in most schedulers)
        scheduled_lrs = None
        try:
            if lr_scheduler is not None and hasattr(lr_scheduler, "get_last_lr"):
                scheduled_lrs = [float(x) for x in lr_scheduler.get_last_lr()]
        except Exception:
            pass

        # detect param update by norm change (only true on steps with optimizer.step())
        updated = None
        norm_val = None
        if model is not None and self.track_name is not None:
            try:
                p = dict(model.named_parameters()).get(self.track_name, None)
                if p is not None:
                    norm_val = p.data.float().norm().item()
                    if self.prev_norm is not None:
                        updated = abs(norm_val - self.prev_norm) > 1e-12
                    self.prev_norm = norm_val
            except Exception:
                pass

        grad_norm = None
        if model is not None and self.track_name is not None:
            try:
                p = dict(model.named_parameters()).get(self.track_name, None)
                if p is not None:
                    if p.grad is not None:
                        grad_norm = p.grad.data.float().norm().item()
                    # [Debug] print grad to check if it is None or all zeros
                    print(f"[Debug] {self.track_name} grad: {p.grad}")
                    '''
                    打印为none, Trainer 的机制限制：在 transformers.Trainer 的主循环中，执行顺序是：
                    loss.backward()（产生梯度）
                    optimizer.step()（利用梯度更新参数）
                    optimizer.zero_grad() （清空梯度为 None）
                    state.global_step += 1
                    调用 on_step_end（你的 Callback 在这里执行）
                    '''
            except Exception:
                pass

        msg = (
            f"[LR Debug] global_step={state.global_step} current_lrs={current_lrs} "
            f"scheduled_lrs={scheduled_lrs} track={self.track_name} norm={norm_val} updated={updated} grad_norm={grad_norm}"
        )
        logger.info(msg)
        # Also print to stdout to ensure visibility regardless of logger settings
        print(msg)

        return control

class UIETrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        print("UIETrainer Initialized")
        super().__init__(*args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        重写优化器创建逻辑，确保所有 requires_grad=True 的参数（包括 loranew_）都被加入。
        """
        print(f">>>> [DEBUG] create_optimizer_and_scheduler is called! steps={num_training_steps} <<<<", flush=True)
        print(f"self.optimizer: {self.optimizer}", flush=True)
        
        # 打印一下，确保你的 loranew 参数确实在里面
        num_loranew = sum(1 for n, p in self.model.named_parameters() if "loranew_" in n and p.requires_grad)
        print(f"--- [Optimizer Debug] Found {num_loranew} 'loranew_' parameters to optimize.")
    
        # 2. 获取优化器参数（从 TrainingArguments 中读取学习率、权重衰减等）
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        
        # 3. 按照标准逻辑处理权重衰减（可选，但建议保留）
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                "weight_decay": 0.0,
            },
        ]

        # 4. 实例化优化器
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)\
        

    def compute_loss_for_debug(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 调用原始的 compute_loss 得到 loss Tensor
        outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss, logits = (outputs[0], outputs[1]) if isinstance(outputs, tuple) else (outputs, None)

        # 2. 检查计算图
        # 使用 self.state.global_step 访问当前的步数
        if self.state.global_step <= 5 or self.state.global_step % self.args.logging_steps == 0:
            print(f"\n[Debug Graph] Global Step: {self.state.global_step}")
            
            # 打印 grad_fn。如果显示为 None，说明从这里计算图就断了
            print(f"[Debug Graph] Loss grad_fn: {loss.grad_fn}")

            # 3. 终极测试：直接测试参数与 Loss 的连通性
            # 遍历模型中所有 requires_grad=True 的参数
            found_trainable = False
            for n, p in model.named_parameters():
                if p.requires_grad and ("loranew_" in n or "lora_" in n):
                    found_trainable = True
                    # 使用 torch.autograd.grad 尝试直接计算梯度
                    # retain_graph=True 是为了不影响后续 Trainer 自己的 backward
                    # allow_unused=True 是为了防止参数没参与计算时报错
                    grads = torch.autograd.grad(loss, p, retain_graph=True, allow_unused=True)[0]
                    
                    if grads is None:
                        print(f"!!! [CRITICAL] 参数 {n} 没参与计算 (grad is None) !!!")
                    else:
                        print(f"--- [SUCCESS] 参数 {n} 连通正常，当前瞬时梯度 norm: {grads.norm().item():.8f}")
            
            if not found_trainable:
                print("!!! [WARNING] 未发现任何 requires_grad=True 的 LoRA 参数 !!!")

        return (loss, logits) if return_outputs else loss
    def _check_grad_fn(self, grad_fn, target_name, depth=0, max_depth=100):
        """递归检查计算图节点（仅作高级调试参考）"""
        if grad_fn is None or depth > max_depth:
            return False
        # 检查该节点是否关联了我们的参数名（这取决于具体算子，不一定都能看到名字）
        # 但通常我们可以通过查看 next_functions 来遍历
        if hasattr(grad_fn, 'next_functions'):
            for next_f, _ in grad_fn.next_functions:
                if next_f is not None:
                    # 只要能一直追踪到 AccumulateGrad 节点，说明图没断
                    if "AccumulateGrad" in str(next_f):
                        return True
                    if self._check_grad_fn(next_f, target_name, depth + 1):
                        return True
        return False

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        ########################### Regularization ##########################
        orthogonal_loss = 0.
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                for name_, param_ in self.model.named_parameters():
                    if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                        orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                        break # target modules have been matched

        # l2-normalization for loranew_A/B
        l2_loss = 0.
        for name, param in self.model.named_parameters():
            if "loranew_" in name:
                l2_loss += torch.norm(param, p=2)

        lamda_1 = self.args.lamda_1
        lamda_2 = self.args.lamda_2

        logger.info(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {lamda_1}; λ2: {lamda_2}")
        loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
        ######################################################################

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        # if args.deepspeed and not self.deepspeed:  因为你使用的是 ZeRO Stage 2 且处于 纯推理模式（Inference），这段手动初始化的代码其实是多余且有害的。
        #     # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
        #     # from the checkpoint eventually
        #     deepspeed_engine, *_ = deepspeed_init(
        #         self, num_training_steps=0, #resume_from_checkpoint=None, # inference=True
        #     )
        #     self.model = deepspeed_engine.module
        #     self.model_wrapped = deepspeed_engine
        #     self.deepspeed = deepspeed_engine


        if args.deepspeed and not self.deepspeed:
            ds_results = deepspeed_init(
                self, num_training_steps=0, #resume_from_checkpoint=None, # inference=True
            )
            first_item = ds_results[0]

            if hasattr(first_item, "module"):
                # ZeRO Stage 3 逻辑：使用 DS Engine
                self.model = first_item.module
                self.model_wrapped = first_item
                self.deepspeed = first_item
            else:
                # ZeRO Stage 2 推理逻辑：
                # DeepSpeed 返回了 DummyOptim，没有接管模型。
                # 此时模型还在 CPU 上，我们需要手动将其移动到当前进程对应的 GPU 设备上。
                self.model = self.model.to(self.args.device)  # <--- 新增这一行！

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        print("Starting evaluation loop...") # Debug print
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                # labels = self._pad_across_processes(labels)  # version update, changed
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                # logits = self._pad_across_processes(logits)  # version update, changed
                logits = self.accelerator.pad_across_processes(logits,dim=1, pad_index=0)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        from transformers.trainer_pt_utils import nested_truncate
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs
        gen_kwargs["synced_gpus"] = True if is_deepspeed_zero3_enabled() else False

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            input_ids=generation_inputs, 
            generation_config=generation_config
        )

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
