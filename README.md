# RottenCore

<p align="center">
  <img src="https://github.com/TingleDinkle/RottenCore/blob/27f5a28ad5a2aab7a0ef7370a1dc0527d094b40c/logo.jpg"/>
</p>

RottenCore is a powerful command-line interface (CLI) tool designed to transform conventional videos into stylized "glyph" art animations. This project is an independent effort to create a general-purpose video-to-glyph converter, inspired by existing microcontroller-focused projects. RottenCore compresses video content by converting each frame into a sequence of discrete, small (e.g., 8x8 pixel) graphical blocks (glyphs). This process drastically reduces video data while maintaining a recognizable, albeit "rotten," aesthetic, ideal for low-bandwidth applications, artistic expression, or retro-style displays. The output is saved in a highly compressed, custom binary `.rc` or `.rcx` format.

## Functionality

RottenCore provides two primary modes for glyph generation, an "Extreme Compression" mode, and a robust rendering engine:

1.  **Machine Learning Optimization (`optimize` command):** Utilizes an LPIPS-based neural network to learn the optimal set of 256 (or custom number) glyphs that best represent the video's visual content. This method aims for high perceptual quality but can be computationally intensive.

2.  **Fast K-Means Clustering (`optimize_kmeans` command):** Employs an optimized K-Means algorithm (adapted from existing C implementations and accelerated with Numba) to quickly derive a set of representative glyphs. This mode is significantly faster and suitable for users who prioritize speed or are working with simpler visual content.

3.  **Extreme Compression Mode (`--extreme` flag):** When enabled, the output uses a highly specialized `.rcx` binary format. This mode utilizes 2-bit block quantization (4 colors per block) and Huffman coding for the frame sequence to achieve extremely small file sizes (typically ~60KB-100KB for short clips), ideal for microcontrollers or ultra-low bandwidth scenarios.

4.  **Flexible Rendering (`render` command):** Reconstructs the glyph-based video from a compressed `.rc` or `.rcx` project file into standard video formats like MP4 or GIF. It supports customizable scaling factors, allowing users to upscale the "rotten" output to higher resolutions while preserving the crisp, pixel-art look.

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

### Optimize (Train Glyphs with Machine Learning)

To train glyphs for a video using the ML (LPIPS) method:

```bash
python rottencore.py optimize input.mp4 --out project.rc --width 64 --height 48 --epochs 500
```

*   `input.mp4`: Path to your input video file.
*   `--out project.rc`: Output path for the **highly compressed RottenCore project file**.
*   `--extreme`: **(Optional)** Use Extreme Compression mode. Generates an `.rcx` file with 2-bit quantization and Huffman coding.
*   `--width`, `--height`: Target resolution for the video frames before glyph extraction.
*   `--glyphs`: Number of glyphs to generate (default: 256).
*   `--block-size`: Size of the square blocks [height width] (default: `8 8`). Example: `--block-size 8 8`.
*   `--epochs`: Number of training epochs (default: 500).
*   `--lr`: Learning rate (default: 0.002).
*   `--batch-size`: Batch size for training (default: 48).
*   `--device`: Device to use for training (`cuda` or `cpu`).

### Optimize K-Means (Fast Glyphs)

To generate glyphs for a video using the faster C-ported K-Means method:

```bash
python rottencore.py optimize_kmeans input.mp4 --out project_kmeans.rc --width 64 --height 48
```

*   `input.mp4`: Path to your input video file.
*   `--out project_kmeans.rc`: Output path for the **highly compressed RottenCore project file**.
*   `--extreme`: **(Optional)** Use Extreme Compression mode. Generates an `.rcx` file with 2-bit quantization and Huffman coding.
*   `--width`, `--height`: Target resolution for the video frames before glyph extraction.
*   `--glyphs`: Number of glyphs to generate (default: 256).
*   `--block-size`: Size of the square blocks [height width] (default: `8 8`). Example: `--block-size 8 8`.
*   `--device`: Device to use for K-Means (default: `cpu`). Note that Numba is CPU-bound here.

### Render (Export Video from Project File)

To render a video from a previously generated `.rc` or `.rcx` project file:

```bash
python rottencore.py render project.rc --out output.mp4 --scale 4 --fps 30
```

*   `project.rc`: Path to your project file (supports both `.rc` and `.rcx`).
*   `--out output.mp4`: Output path for the rendered video. The extension determines the output format (`.mp4` or `.gif`).
*   `--fps`: Frames per second for the output video. If not specified, it will use the original FPS of the input video that was used to create the project file.
*   `--scale`: Upscaling factor for the output video (e.g., `4` for 4x upscaling, turning a 64x48 video into 256x192). Defaults to `1` (no scaling).
*   `--gif`: (Optional) If your output file ends in `.gif`, it will automatically be rendered as a GIF. The `--gif` flag is legacy and no longer strictly required if the filename is explicit.