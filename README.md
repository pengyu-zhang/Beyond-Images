# Beyond-Images
Beyond Images - Is a Thousand Words Worth One Picture? A Framework for Multi-Modal Knowledge Graph Dataset Enrichment

<br><br>
<div align="center">
<img src="fig.png" width="800" />
</div>
<br><br>

Multi-Modal Knowledge Graphs (MMKGs) enhance entity representations by incorporating text, images, audio, and video, offering a more comprehensive understanding of each entity. Images are particularly valuable among these modalities due to their rich content and the ease of large-scale collection. However, many images are noisy or weakly related to their entities, leading to semantic information loss. Sparse-semantic images (e.g., brand logos) provide limited meaningful features, while rich-semantic images (e.g., abstract artwork) contain complex semantics that are difficult to capture, raising concerns about whether images consistently improve performance.

To address this, we generate meaningful textual descriptions from images to better capture the semantic relationship between the image and its entity. By incorporating these text-based image representations, we reduce dependence on low-quality visuals and achieve a 2%–5% improvement in Hit@1 for link prediction across three MMKG datasets. Additionally, our automated framework enables large-scale image retrieval while minimizing the need for labor-intensive creation and manual filtering. In some cases, replacing images with textual descriptions leads to more efficient and compact representations, highlighting the important role of language in MMKGs.

## Usage

Please follow the instructions next to reproduce our experiments, and to train a model with your own data.

### 1. Install the requirements

Creating a new environment (e.g. with `conda`) is recommended. Use `requirements.txt` to install the dependencies:

```
conda create -n beyondimages311 -y python=3.11 && conda activate beyondimagesl311
pip install -r requirements.txt
```

### 2. Download the data

| Download link                                                | Size |
| ------------------------------------------------------------ | ----------------- |
| [Our Dataset(Due to the anonymity requirement of the review process, the dataset will be made available after the review is complete.)]() | 127 MB            |

### 3. Reproduce the experiments

```
bash run.sh
```
<br><br>
<div align="center">
<img src="fig2.png" width="700" />
</div>
<br><br>

