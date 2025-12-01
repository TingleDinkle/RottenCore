import argparse
import os
import torch
import pickle

from RottenCore.src.core import train_glyphs
from RottenCore.src.video_utils import get_video_properties
from RottenCore.src.compression import RottenCompressor
from RottenCore.src.extreme_compression import ExtremeCompressor
from RottenCore.src.fast_kmeans import generate_glyphs_kmeans
from RottenCore.src.renderer import RottenRenderer
from RottenCore.src.config import (
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_GLYPHS,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_OPTIMIZE_EPOCHS,
    DEFAULT_OPTIMIZE_LR,
    DEFAULT_OPTIMIZE_BATCH_SIZE,
    DEFAULT_FPS
)

def main():
    parser = argparse.ArgumentParser(description="RottenCore: Video-to-Glyph Converter Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Optimize (ML) command
    optimize_parser = subparsers.add_parser("optimize", help="Runs ML training to generate glyphs for a video.")
    optimize_parser.add_argument("input_video", type=str, help="Path to the input video file (e.g., input.mp4).")
    optimize_parser.add_argument("--out", type=str, required=True, help="Output path for the RottenCore project file (e.g., project.rc).")
    optimize_parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Target width for video frames (default: {DEFAULT_WIDTH}).")
    optimize_parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Target height for video frames (default: {DEFAULT_HEIGHT}).")
    optimize_parser.add_argument("--glyphs", type=int, default=DEFAULT_GLYPHS, help=f"Number of glyphs to generate (default: {DEFAULT_GLYPHS}).")
    optimize_parser.add_argument("--block-size", type=int, nargs=2, default=DEFAULT_BLOCK_SIZE, 
                                 help=f"Size of the blocks [height width] (default: {DEFAULT_BLOCK_SIZE[0]} {DEFAULT_BLOCK_SIZE[1]}). Example: --block-size 8 8")
    optimize_parser.add_argument("--epochs", type=int, default=DEFAULT_OPTIMIZE_EPOCHS, help=f"Number of training epochs (default: {DEFAULT_OPTIMIZE_EPOCHS}).")
    optimize_parser.add_argument("--lr", type=float, default=DEFAULT_OPTIMIZE_LR, help=f"Learning rate for optimization (default: {DEFAULT_OPTIMIZE_LR}).")
    optimize_parser.add_argument("--batch-size", type=int, default=DEFAULT_OPTIMIZE_BATCH_SIZE, help=f"Batch size for training (default: {DEFAULT_OPTIMIZE_BATCH_SIZE}).")
    optimize_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                                 help="Device to use for training (e.g., 'cuda', 'cpu').")
    optimize_parser.add_argument("--extreme", action="store_true", help="Enable Extreme Compression mode (2-bit quantization, Huffman coding).")


    # Optimize K-Means (fast) command
    optimize_kmeans_parser = subparsers.add_parser("optimize_kmeans", help="Runs fast K-Means clustering to generate glyphs for a video.")
    optimize_kmeans_parser.add_argument("input_video", type=str, help="Path to the input video file (e.g., input.mp4).")
    optimize_kmeans_parser.add_argument("--out", type=str, required=True, help="Output path for the RottenCore project file (e.g., project.rc).")
    optimize_kmeans_parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Target width for video frames (default: {DEFAULT_WIDTH}).")
    optimize_kmeans_parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Target height for video frames (default: {DEFAULT_HEIGHT}).")
    optimize_kmeans_parser.add_argument("--glyphs", type=int, default=DEFAULT_GLYPHS, help=f"Number of glyphs to generate (default: {DEFAULT_GLYPHS}).")
    optimize_kmeans_parser.add_argument("--block-size", type=int, nargs=2, default=DEFAULT_BLOCK_SIZE, 
                                        help=f"Size of the blocks [height width] (default: {DEFAULT_BLOCK_SIZE[0]} {DEFAULT_BLOCK_SIZE[1]}). Example: --block-size 8 8")
    optimize_kmeans_parser.add_argument("--device", type=str, default="cpu", # K-Means is CPU-bound with Numba
                                        help="Device to use for K-Means (e.g., 'cpu').")
    optimize_kmeans_parser.add_argument("--extreme", action="store_true", help="Enable Extreme Compression mode (2-bit quantization, Huffman coding).")


    # Render command
    render_parser = subparsers.add_parser("render", help="Exports a compressed project back to a viewable video.")
    render_parser.add_argument("project_file", type=str, help="Path to the RottenCore project file (e.g., project.rc).")
    render_parser.add_argument("--out", type=str, required=True, help="Output path for the rendered video (e.g., output.mp4).")
    render_parser.add_argument("--fps", type=int, default=None, 
                               help="Frames per second for the output video. If not specified, uses original FPS from project file.")
    render_parser.add_argument("--scale", type=int, default=1, 
                               help="Upscaling factor for the output video (e.g., 2 for 2x upscaling).")
    render_parser.add_argument("--gif", action="store_true", help="Output as GIF (will be converted from MP4).")


    args = parser.parse_args()

    if args.command == "optimize":
        if not os.path.exists(args.input_video):
            print(f"Error: Input video file not found at {args.input_video}")
            exit(1)
        
        output_dir = os.path.dirname(args.out)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        _, _, original_fps_rational = get_video_properties(args.input_video)
        original_fps = float(original_fps_rational) if original_fps_rational else DEFAULT_FPS

        train_glyphs(
            video_path=args.input_video,
            output_rc_path=args.out,
            width=args.width,
            height=args.height,
            num_glyphs=args.glyphs,
            block_size=tuple(args.block_size),
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
            original_fps=original_fps,
            use_extreme_mode=args.extreme
        )
    elif args.command == "optimize_kmeans":
        if not os.path.exists(args.input_video):
            print(f"Error: Input video file not found at {args.input_video}")
            exit(1)
        
        output_dir = os.path.dirname(args.out)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Starting K-Means optimization for video: {args.input_video}")
        print(f"Target resolution: {args.width}x{args.height}, Glyphs: {args.glyphs}, Block size: {args.block_size}")
        
        _, _, original_fps_rational = get_video_properties(args.input_video)
        original_fps = float(original_fps_rational) if original_fps_rational else DEFAULT_FPS

        generate_glyphs_kmeans(
            video_path=args.input_video,
            output_rc_path=args.out,
            width=args.width,
            height=args.height,
            num_glyphs=args.glyphs,
            block_size=tuple(args.block_size),
            original_fps=original_fps,
            use_extreme_mode=args.extreme
        )
        
    elif args.command == "render":
        if not os.path.exists(args.project_file):
            print(f"Error: Project file not found at {args.project_file}")
            exit(1)

        output_dir = os.path.dirname(args.out)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        renderer = RottenRenderer(args.project_file)
        
        fps_to_use = args.fps if args.fps is not None else renderer.original_fps
        if fps_to_use is None:
            print(f"Warning: FPS not specified and not found in project file. Defaulting to {DEFAULT_FPS} FPS.")
            fps_to_use = DEFAULT_FPS

        renderer.render_video(
            output_filepath=args.out,
            scale=args.scale,
            fps=fps_to_use
        )
        
        if args.gif:
            # Legacy flag support, though renderer handles .gif extension natively now
            if not args.out.lower().endswith('.gif'):
                 print("Warning: --gif flag used but output filename does not end in .gif. Renderer might output MP4.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()