import { useRef, useState } from 'react';
import type { ImageUploadResponse } from '../api/client';
import { uploadImage } from '../api/client';
import './UploadPage.css';

export function UploadPage() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [description, setDescription] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<ImageUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  function selectFile(file: File) {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) selectFile(file);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) selectFile(file);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedFile) return;
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const res = await uploadImage(selectedFile, description);
      setResult(res);
      setSelectedFile(null);
      setPreviewUrl(null);
      setDescription('');
      if (fileInput.current) fileInput.current.value = '';
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="upload-page">
      <h2>Upload Image</h2>
      <form onSubmit={handleSubmit} className="upload-form">
        <div
          className={`drop-zone${dragOver ? ' drop-zone--active' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInput.current?.click()}
          role="button"
          aria-label="Drop zone – click or drag an image here"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && fileInput.current?.click()}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Preview" className="drop-zone__preview" />
          ) : (
            <span className="drop-zone__hint">
              Click or drag &amp; drop an image here
            </span>
          )}
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            aria-hidden="true"
          />
        </div>

        {selectedFile && (
          <p className="upload-form__filename">{selectedFile.name}</p>
        )}

        <label className="upload-form__label" htmlFor="description">
          Description <span className="upload-form__optional">(optional – AI generates one if blank)</span>
        </label>
        <textarea
          id="description"
          className="upload-form__textarea"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe the image…"
          rows={3}
        />

        <button
          type="submit"
          className="upload-form__submit"
          disabled={!selectedFile || uploading}
        >
          {uploading ? 'Uploading…' : 'Upload'}
        </button>
      </form>

      {error && <p className="upload-page__error">{error}</p>}

      {result && (
        <div className="upload-page__success" role="status">
          <p>✅ {result.message}</p>
          <p>
            <strong>ID:</strong> {result.image.id} &nbsp;|&nbsp;
            <strong>File:</strong> {result.image.original_filename}
          </p>
          {result.image.description && (
            <p><strong>Description:</strong> {result.image.description}</p>
          )}
        </div>
      )}
    </div>
  );
}
