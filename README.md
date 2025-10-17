# Beyond-Images
Are a Thousand Words Better Than a Single Picture?

Beyond Images - A Framework for Multi-Modal Knowledge Graph Dataset Enrichment

# Supplementary Material

[Supplementary Material](https://github.com/pengyu-zhang/Beyond-Images/blob/main/Supplementary_Material.pdf)

# Video Demo

[Video Demo](https://github.com/pengyu-zhang/Beyond-Images/blob/main/video%20demo/video%20demo.mp4)

<div align="center">
  <!-- 把 HEIGHT 换成图像像素高度，例如 500；rx=16 为圆角半径 -->
  <svg width="800" viewBox="0 0 800 HEIGHT" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <clipPath id="rounded">
        <rect x="0" y="0" width="800" height="HEIGHT" rx="16" ry="16"/>
      </clipPath>
    </defs>
    <image href="fig.png"
           width="800" height="HEIGHT"
           clip-path="url(#rounded)"
           preserveAspectRatio="xMidYMid slice"/>
    <rect x="0" y="0" width="800" height="HEIGHT" rx="16" ry="16"
          fill="none" stroke="#e5e7eb" stroke-width="1"/>
  </svg>
</div>



Multi-Modal Knowledge Graphs (MMKGs) enrich entity representations by incorporating diverse modalities such as text and images. While images offer valuable semantic information, many are semantically ambiguous, making it difficult to align them with the corresponding entities. To address this, we propose Beyond Images, an automated framework that enhances MMKGs by generating textual descriptions from entity-linked images and summarizing them using LLMs. Our framework includes: (1) automatic retrieval of additional images, (2) image-to-text models that convert ambiguous visual content into informative descriptions, and (3) LLM-based fusion to summarize multiple descriptions and filter out irrelevant or noisy semantic content. Experiments on three public MMKG datasets using four representative models demonstrate consistent improvements, with up to a 7% gain in Hits@1 for link prediction. These results highlight the value of language as a semantic bridge in MMKGs, particularly when visual inputs are noisy.

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
| Our full datasets contain many images (~23 GB) and cannot be hosted on GitHub; we will release them after the paper is accepted, and have temporarily uploaded a sample folder with a small subset. | 23 GB (includes raw images)            |
| Small example dataset: img_text_summary.zip | 4.16 MB            |

### 3. Reproduce the experiments

```
bash run.sh
```
<br><br>
<div align="center">
<img src="fig2.png" width="700" />
</div>
<br><br>

