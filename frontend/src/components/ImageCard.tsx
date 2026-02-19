import { useState } from 'react';
import type { ImageSearchResult } from '../api/client';
import { deleteImage, imageUrl } from '../api/client';
import './ImageCard.css';

interface Props {
  image: ImageSearchResult;
  onDeleted: (id: number) => void;
}

export function ImageCard({ image, onDeleted }: Props) {
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleDelete() {
    if (!window.confirm(`Delete "${image.original_filename}"?`)) return;
    setDeleting(true);
    setError(null);
    try {
      await deleteImage(image.id);
      onDeleted(image.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed');
      setDeleting(false);
    }
  }

  return (
    <article className="image-card">
      <div className="image-card__thumb">
        <img
          src={imageUrl(image.filename)}
          alt={image.description ?? image.original_filename}
          loading="lazy"
        />
      </div>
      <div className="image-card__body">
        <p className="image-card__filename" title={image.original_filename}>
          {image.original_filename}
        </p>
        {image.description && (
          <p className="image-card__desc">{image.description}</p>
        )}
        <p className="image-card__score">
          Score: <strong>{image.score.toFixed(3)}</strong>
        </p>
        {error && <p className="image-card__error">{error}</p>}
        <button
          className="image-card__delete"
          onClick={handleDelete}
          disabled={deleting}
          aria-label={`Delete ${image.original_filename}`}
        >
          {deleting ? 'Deleting…' : 'Delete'}
        </button>
      </div>
    </article>
  );
}
