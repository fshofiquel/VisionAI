/**
 * VisionAI API client.
 * All requests go through Vite's dev-server proxy to http://localhost:8000.
 */

const BASE = '/api/v1/images';

export interface ImageResponse {
  id: number;
  filename: string;
  original_filename: string;
  filepath: string;
  content_type: string;
  file_size: number;
  description: string | null;
  created_at: string;
}

export interface ImageUploadResponse {
  message: string;
  image: ImageResponse;
}

export interface ImageSearchResult {
  id: number;
  filename: string;
  original_filename: string;
  filepath: string;
  description: string | null;
  score: number;
}

export interface SearchResponse {
  query: string | null;
  results: ImageSearchResult[];
  total: number;
}

/** Upload an image file with an optional description. */
export async function uploadImage(
  file: File,
  description?: string,
): Promise<ImageUploadResponse> {
  const form = new FormData();
  form.append('file', file);
  if (description?.trim()) {
    form.append('description', description.trim());
  }

  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? 'Upload failed');
  }
  return res.json();
}

/** Search images using a natural language query. */
export async function searchImages(
  q: string,
  limit = 20,
  minScore = 0.0,
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q,
    limit: String(limit),
    min_score: String(minScore),
  });
  const res = await fetch(`${BASE}/search/text?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? 'Search failed');
  }
  return res.json();
}

/** Get a single image by ID. */
export async function getImage(id: number): Promise<ImageResponse> {
  const res = await fetch(`${BASE}/${id}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? 'Image not found');
  }
  return res.json();
}

/** Delete an image by ID. */
export async function deleteImage(id: number): Promise<void> {
  const res = await fetch(`${BASE}/${id}`, { method: 'DELETE' });
  if (!res.ok && res.status !== 204) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? 'Delete failed');
  }
}

/** Build the URL to serve an uploaded image thumbnail. */
export function imageUrl(filename: string): string {
  return `/static/uploads/${filename}`;
}
