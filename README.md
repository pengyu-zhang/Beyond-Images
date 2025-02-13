# Beyond-Images
Beyond Images - Is a Thousand Words Worth One Picture? A Framework for Multi-Modal Knowledge Graph Dataset Enrichment

<br><br>
<div align="center">
<img src="fig.png" width="800" />
</div>
<br><br>

Multi-Modal Knowledge Graphs (MMKGs) enhance entity representations by incorporating text, images, audio, and video, offering a more comprehensive understanding of each entity. Among these modalities, images are especially valuable due to their rich content and the ease of large-scale collection. However, many images are semantically unclear, making it challenging for the models to effectively use them to enhance entity representations. To address this, we present the Beyond Images framework, which generates textual descriptions for entity images, which more effectively capture their semantic relevance to the associated entity. By adding textual descriptions, we achieve up to 5\% improvement in Hits@1 for link prediction across three MMKG datasets. Furthermore, our scalable framework reduces the need for manual construction by automatically extending three MMKG datasets with additional images and their descriptions. Our work highlights the importance of textual descriptions for MMKGs.

## Usage

Please follow the instructions next to reproduce our experiments, and to train a model with your own data.

### 1. Install the requirements

Creating a new environment (e.g. with `conda`) is recommended. Use `requirements.txt` to install the dependencies:

```
conda create -n beyondimages311 -y python=3.11 && conda activate beyondimages311
pip install -r requirements.txt
```

### 2. Download the data

| Download link                                                | Size |
| ------------------------------------------------------------ | ----------------- |
| Our full dataset (Due to the anonymity requirement of the review process, the full dataset will be made available soon.) | 127 MB            |
| Small example dataset: img_text_summary.zip | 25 MB            |

### 3. Reproduce the experiments

```
bash run.sh
```
<br><br>
<div align="center">
<img src="fig2.png" width="700" />
</div>
<br><br>

