import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_registered_image_names(images_txt_path):
    """Return image names from COLMAP text images.txt."""
    image_names = []
    expect_pose_line = True

    with open(images_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if expect_pose_line:
                parts = line.split()
                if len(parts) < 10:
                    raise ValueError(f"Invalid COLMAP image pose line: {line}")
                image_names.append(parts[9])

            expect_pose_line = not expect_pose_line

    return image_names


def count_points3d(points3d_txt_path):
    """Count non-comment point records in COLMAP text points3D.txt."""
    if not points3d_txt_path.exists():
        return 0

    count = 0
    with open(points3d_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def export_3dgs_scene(image_dir, colmap_dir, output_dir):
    """Package registered SfM outputs into a 3DGS-compatible scene directory."""
    image_dir = Path(image_dir)
    colmap_dir = Path(colmap_dir)
    output_dir = Path(output_dir)

    images_txt = colmap_dir / "images.txt"
    cameras_txt = colmap_dir / "cameras.txt"
    points3d_txt = colmap_dir / "points3D.txt"

    for required_path in (images_txt, cameras_txt, points3d_txt):
        if not required_path.exists():
            raise FileNotFoundError(f"Required COLMAP file not found: {required_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_images_dir = output_dir / "images"
    output_sparse_dir = output_dir / "sparse" / "0"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_sparse_dir.mkdir(parents=True, exist_ok=True)

    registered_names = parse_registered_image_names(images_txt)
    missing_images = []

    for image_name in registered_names:
        src = image_dir / image_name
        dst = output_images_dir / Path(image_name).name
        if not src.exists():
            missing_images.append(image_name)
            logger.warning(f"Registered image not found in source directory: {image_name}")
            continue
        shutil.copy2(src, dst)

    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(colmap_dir / filename, output_sparse_dir / filename)

    registered_list_path = output_dir / "registered_images.txt"
    with open(registered_list_path, "w", encoding="utf-8") as f:
        for image_name in registered_names:
            f.write(f"{image_name}\n")

    stats = {
        "source_image_dir": str(image_dir),
        "source_colmap_dir": str(colmap_dir),
        "output_dir": str(output_dir),
        "num_registered_images": len(registered_names),
        "num_copied_images": len(registered_names) - len(missing_images),
        "num_missing_images": len(missing_images),
        "missing_images": missing_images,
        "num_points3d": count_points3d(points3d_txt),
    }

    stats_path = output_dir / "reconstruction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(
        "3DGS scene export complete: "
        f"{stats['num_copied_images']}/{stats['num_registered_images']} images copied, "
        f"{stats['num_points3d']} points, output={output_dir}"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Export a COLMAP text model as a 3DGS scene directory.")
    parser.add_argument("--image-dir", required=True, help="Directory containing the original input images.")
    parser.add_argument("--colmap-dir", required=True, help="Directory containing cameras.txt, images.txt, and points3D.txt.")
    parser.add_argument("--output-dir", required=True, help="Output scene directory for 3DGS.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    export_3dgs_scene(args.image_dir, args.colmap_dir, args.output_dir)


if __name__ == "__main__":
    main()
