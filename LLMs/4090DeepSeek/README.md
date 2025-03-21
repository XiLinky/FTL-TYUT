# 跑通需要改一些东西！！！dw发的教程是不能完全跑通的

unsloth==2025.2.15
unsloth-zoo==2025.2.7
trl==0.15.2

如果直接pip install unsloth会报错，因为新版本的unsloth-zoo定义compute_dtype时赋值有问题
其余的跟着教程练就行

## 关于loss为0的问题

loss 为 0 可以确定是 GRPO 本身的特性

GRPO的loss代码：
```python
per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
per_token_loss = -(per_token_loss - self.beta * per_token_kl)
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

由于.detach()只是返回一个共享存储位置但没有梯度的tensor，所以per_token_logps - per_token_logps.detach()为0，torch.exp(per_token_logps - per_token_logps.detach())等于1，因此，此时的per_token_loss等于advantages

只不过如果计算这一步的梯度的话，per_token_logps.detach()就要被看做常数C了，所以整体是有per_token_logps梯度的

### 第一步的KL为什么是0？
```python
with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
    prompt_completion_ids = unwrapped_model.generate(
        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
    )
ref_per_token_logps = self._get_per_token_logps(
    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
)
...
input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
# Compute the KL divergence between the model and the reference model
ref_per_token_logps = inputs["ref_per_token_logps"]
per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
```

在第一步的时候，actor模型此时权重与参考模型ref_model一致，所以per_token_logps = ref_per_token_logps ，代入公式中，所以KL=0

### 第一步的advantages为什么是0？
```python
# Sum the rewards from all reward functions
rewards = rewards_per_func.sum(dim=1)

# Compute grouped-wise rewards
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

# Normalize the rewards to compute the advantages
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

# Slice to keep only the local part of the data
process_slice = slice(
    self.accelerator.process_index * len(prompts),
    (self.accelerator.process_index + 1) * len(prompts),
)
advantages = advantages[process_slice]

...

# x - x.detach() allows for preserving gradients from x
advantages = inputs["advantages"]
per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
per_token_loss = -(per_token_loss - self.beta * per_token_kl)
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

第4步是对组内每个样本的reward进行标准化，第5步时对组内的标准化后的reward求和。那么对于标准化公式(ri - mean) / std 求和，就正好分子为0了

换而言之，其实GRPO Loss就等于βKL。(https://github.com/huggingface/trl/issues/2703#issuecomment-2625274839)

只不过advantages可以在梯度计算中保留。用loss计算梯度，loss为0不代表梯度也为0

# Datawhale-R1 复现文件

# 单机多卡复现

对应文件：
- Datawhale-R1.yaml
- train_Datawhale-R1.py
- deepspeed_zero3.yaml
- requirements.txt
- train_Datawhale-R1.sh

对应文章：[DeepSeek R1 Zero中文复现教程来了！
](https://mp.weixin.qq.com/s?__biz=MzIyNjM2MzQyNg==&mid=2247700308&idx=1&sn=aa6324d30cc6d054c1dbb238b013b9b5&chksm=e98d1841220df3bd906ebe92682d4ff32dfa2fbd382fc714c8c457631de484068775c1c26846&mpshare=1&scene=2&srcid=0206JNv8uw29ECf9inhhzaxg&sharer_shareinfo=a3c5178266c37875a63b36d4a96bde91&sharer_shareinfo_first=5cd0c564850ed06c98ad41d8c06b256f#rd)

> [!CAUTION]
> 更正说明
>
> 1. 本文并不是严谨复现的 DeepSeek-R1-Zero，如果需要尽可能贴近原文，请使用 [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) 而不是 [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。但是请注意，这可能需要更长的训练步长才能达到理想效果。
> 2. **请删除思考长度奖励函数 `thought_len_reward_func`，并将 `GRPOTrainer` 的 `reward_funcs` 参数的值改为 `[format_reward_func, equation_reward_func]`**，本仓库提供的代码已经修改。思考长度奖励函数由 [骆师傅](https://github.com/anine09) 对 DeepSeek-R1-Zero 训练方法的错误理解而引入，经过与其他同学的讨论，它会影响模型性能，详见下一条分析， **请立刻更新你的训练代码！**
> 3. 由于思考长度奖励函数带来的影响，请谨慎评估本文的 **训练结果解读** 部分，思考长度奖励函数可能造成模型过分追求长输出，而导致的 Token 重复问题。更长的思考长度与更深度、细致的思考没有必然的因果关系，由文章报告的结果也能看出，模型后期放弃追求思考长度奖励，而回归一个稳定的输出长度。其他大部分同学的复现报告观察到“输出长度随着问题困难度增加而自然增长”、“输出长度有先降低后增加的趋势”。思考长度奖励函数是由于在训练初期观察到输出长度不断降低，从而引入这个奖励试图对抗长度降低趋势，但是这是一个错误设计，关于文章中任何提到思考长度奖励函数的部分都应该被删除，包括：介绍、代码、举例、示意图、训练曲线。
> 4. 我们在文章中推荐大家使用的 [TinyZero](https://github.com/Jiayi-Pan/TinyZero) 项目没有这个错误。
> 5. 关于 Aha Moment 的判断大家当笑话看看就好，仅为 [骆师傅](https://github.com/anine09) 个人观点，仍需更多严谨研究验证。
> 6. 忘了开启 `flash-attn`，已在 https://github.com/datawhalechina/unlock-deepseek/pull/25 修复，感谢 [@LinglingGreat](https://github.com/LinglingGreat) 同学的贡献。
> 7. 我们同时更新一版训练流程示意图。
> ![image](https://github.com/user-attachments/assets/8f4d576c-55cd-49bf-91a9-f9a86724ef04)


# 单机单卡复现

对应文件：
- Datawhale-R1_unsloth.yaml
- train_Datawhale-R1_unsloth.py
- train_Datawhale-R1_unsloth.py
- train_Datawhale-R1_unsloth.sh
