from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    catalog_dir = project_dir / "data" / "catalog"
    image_files = sorted(
        path
        for path in catalog_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_files:
        raise FileNotFoundError(f"No JPG, JPEG, or PNG images found in {catalog_dir}")

    model = SentenceTransformer("clip-ViT-B-32")
    embeddings = []
    image_paths = []
    batch_size = 32

    for start in tqdm(range(0, len(image_files), batch_size), desc="Extracting embeddings"):
        batch_paths = image_files[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        batch_embeddings = model.encode(
            images,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings.append(batch_embeddings)
        image_paths.extend(
            path.relative_to(project_dir).as_posix() for path in batch_paths
        )

    np.save(project_dir / "data" / "embeddings.npy", np.concatenate(embeddings))
    np.save(project_dir / "data" / "image_paths.npy", np.asarray(image_paths))


if __name__ == "__main__":
    main()
