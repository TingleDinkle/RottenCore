# RottenCore

<p align="center">
  <img src="https://github.com/TingleDinkle/RottenCore/blob/27f5a28ad5a2aab7a0ef7370a1dc0527d094b40c/logo.jpg"/>
</p>

RottenCore is a powerful command-line interface (CLI) tool designed to transform conventional videos into stylized "glyph" art animations. This project is an independent effort to create a general-purpose video-to-glyph converter, inspired by existing microcontroller-focused projects. RottenCore compresses video content by converting each frame into a sequence of discrete, small (e.g., 8x8 pixel) graphical blocks (glyphs). 

The tool offers two compression tiers:
1.  **Standard Mode (`.rc`):** High-fidelity blocks (8-bit grayscale), suitable for general playback.
2.  **Extreme Mode (`.rcx`):** Ultra-low bitrate (2-bit quantization, Huffman coding), ideal for microcontrollers or retro hardware constraints.

## Functionality

RottenCore provides flexible pipelines for generation and rendering:

1.  **Machine Learning Optimization (`optimize` command):** Utilizes an LPIPS-based neural network to learn the optimal set of 256 (or custom number) glyphs that best represent the video's visual content. This method aims for high perceptual quality.

2.  **Fast K-Means Clustering (`optimize_kmeans` command):** Employs an optimized K-Means algorithm (accelerated with Numba) to quickly derive a set of representative glyphs. This mode is significantly faster and suitable for processing long videos.

3.  **Compression Modes:**
    *   **Standard (Default):** Saves as `.rc`. Uses LZMA compression on full-precision block data. Best for quality.
    *   **Extreme (`--extreme`):** Saves as `.rcx`. Forces 2-bit color depth (4 colors per block) and uses Huffman coding for the frame sequence. This results in incredibly small files (often 60KB-100KB for short clips) at the cost of some visual fidelity.

4.  **Universal Rendering (`render` command):** Automatically detects `.rc` or `.rcx` formats and reconstructs the video into standard MP4 or GIF. Supports upscaling to preserve the pixel-art aesthetic on high-res screens.

## Acknowledgements

This project draws inspiration from the innovative work in video compression and glyph-based rendering, particularly from early microcontroller-focused demos like the "Bad Apple!!" implementations. Concepts for data structures and algorithmic approaches were informed by reviewing open-source projects in this domain.

https://github.com/cnlohr/badderapple

## Installation

Assuming you have obtained the RottenCore project files (e.g., by downloading a release archive or cloning your own fork/repository):

1.  **Navigate to the project directory:**
    ```bash
    cd /path/to/RottenCore
    ```
2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **(Optional) Install as an editable package:**
    If you plan to develop RottenCore or want to use the `rottencore` command directly from your shell, you can install it in "editable" mode:
    ```bash
    pip install -e .
    ```
    This allows you to run `rottencore <command>` from any directory.

## Usage

### 1. Optimize (Train Glyphs with Machine Learning)

Trains a neural network to find the best glyphs.

**Standard Mode:**
```bash
python rottencore.py optimize input.mp4 --out project.rc --width 64 --height 48 --epochs 500
```

**Extreme Mode:**
```bash
python rottencore.py optimize input.mp4 --out project.rcx --width 64 --height 48 --extreme
```

*   `input.mp4`: Path to your input video file.
*   `--out`: Output filename. Use `.rc` for standard or `.rcx` for extreme.
*   `--extreme`: **(Optional)** Enables 2-bit quantization and Huffman coding.
*   `--width`, `--height`: Target resolution (e.g., 64x48).
*   `--epochs`: Training duration (default: 500).
*   `--device`: `cuda` or `cpu`.

### 2. Optimize K-Means (Fast Generation)

Quickly clusters video patches to find glyphs.

**Standard Mode:**
```bash
python rottencore.py optimize_kmeans input.mp4 --out project_kmeans.rc --width 64 --height 48
```

**Extreme Mode:**
```bash
python rottencore.py optimize_kmeans input.mp4 --out project_kmeans.rcx --width 64 --height 48 --extreme
```

*   `--extreme`: **(Optional)** Enables 2-bit quantization and Huffman coding.
*   `--glyphs`: Number of glyphs (default: 256).
*   `--block-size`: Block dimensions (default: `8 8`).

### 3. Render (Export to Video/GIF)

Converts any project file back to a viewable format.

```bash
python rottencore.py render project.rc --out output.mp4 --scale 4
```
*   `project.rc`: Can be a standard `.rc` file OR an extreme `.rcx` file. The renderer detects the format automatically.
*   `--out`: Output path (`.mp4` or `.gif`).
*   `--scale`: Upscaling factor (e.g., `4` turns 64x48 into 256x192).
