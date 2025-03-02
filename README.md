# FTL-TYUT
## For the learners AND To My Hustle-Fueled Odyssey Toward Graduate Studies
# AI全栈技术导航库 🚀
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/reponame?style=social)](https://github.com/yourusername/reponame)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://via.placeholder.com/800x300?text=AI+Learning+Path" alt="Banner">
</p>

## 🌟 项目愿景
为深度学习自学者打造**模块化技术栈导航**，通过七大核心领域的经典/前沿模型实现，构建从理论到实践的认知闭环。
每个项目均包含：
- 🧠 模型架构解读
- ⚡ 极简实现代码
- 📈 训练/推理脚本
未包含：
- 📚 数据集下载(涉及比赛数据和未公开数据)
- 📦 环境配置(时间跨度大，环境过多)

## 📂 目录架构
### 🔍 数据挖掘
- **数据**：`Pandas`/`Seaborn`/`ydata-profiling`
- **预测模型**：
  - 电动汽车充电需求预测（`LightGBM`/`KFold`/`LSTM`/`ARIMA`）
  - 锂离子电池温度预测（`FNN`/`XGBoost`/`Stacking`）
  - 量子化学分子属性预测（`Joblib`/`TF-IDF`/`CatBoost`）
- **分类任务**：
  - 糖尿病风险预测（`CatBoost`）
  - 中枢神经系统药物预测（`QSAR`/`LightGBM`）

### 🤖 机器学习
- **回归预测**：
  - 房屋价格预测（`Linear Regression`）
  - 太原暖气供应预测（`SVR`）
- **分类实战**：
  - 泰坦尼克生存预测（`Decision Tree`/`Random Forest`）
  - 广告分类（`PCA`/`XGBoost`）
- **聚类分析**：
  - 航空客户分群（`KMeans++`/`DBSCAN`）
- **图像处理**：
  - 图像去噪（`Markov`）

### ⏳ 时序预测
- **金融时序**：
  - 股票预测（`LSTM`/`GRU`）
  - 二手车价格预测(`CNN-LSTM`)
- **文本生成**：
  - 藏头诗生成（`LSTM`）
  - 莎士比亚诗风生成（`LSTM`）
- **环境预测**：
  - 天气预测（`BiLSTM`）

### 📝 NLP实战
- **预训练模型**：
  - 语法纠错/情感分析（`BERT`/`HuggingFace`）
- **传统方法**：
  - 推特情感识别（`n-gram`）
  - 莎士比亚文本挖掘（`Word2Vec`）
- **分类任务**：
  - 新闻分类（`ELMo`）

### 👁️ 计算机视觉
- **医疗影像**：
  - 脑PET疾病预测（`ResNet`/`VGG`）
  - 皮肤病分割（`UNet`/`UNet++`）
- **目标检测**：
  - 烟火检测（`YOLOv5`）
  - 红外小目标检测（`YOLOv8`）
- **图像生成**：
  - Fashion MNIST修复（`VAE`）
  - 风格迁移（`VGG`）
- **其余模型解读**

### 🌐 多模态
- **跨模态检索**：
  - 智能驾驶视频理解（`CLIP`）
  - 图像检索系统（`CLIP`/`Milvus`）
- **生成模型**：
  - Stable Diffusion分析（`BLIP2`）

### 🧠 大模型
- **教育应用**：
  - AI助教（`LangChain`/`RAG`/`CoT`）
  - 论文助手（`LoRA`/`vLLM`）
- **行业落地**：
  - 数字文旅（`LazyGraphRAG`/`CogVideoX`）
  - 文旅推荐(`RAG`/`Tools`)
  - 文旅剧本杀（`MetaGPT`）
- **模型优化**：
  - DeepSeek-R1单卡4090部署（`Unsloth`/`LoRA`）
  - 手搓Llama2
- **AIoT**：
  - 智能家居（`RAG`/`CoT`/`blockly`）
- **模型解读**：
  - Qwen（`Qwen2`/`Qwen2.5`）
  - GPT（`GPT1-4`）
  - Llama（`Llama3`/`Llava`）
  - DeepSeek（`DeepSeek v3`/`DeepSeek R1`）
  - 其他

### 🧰 工具箱
- **生成模型**：
  - CIFAR10训练（`Diffusion`）
  - GAN实战
- **语音合成**：
  - ChatTTS语音生成
  - EchoMimic数字人部署
- **图网络**：
  - GCN分子属性预测