import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { uploadImage, searchImages, getImage, deleteImage, imageUrl } from '../api/client';

describe('API client', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('imageUrl', () => {
    it('builds the static upload URL', () => {
      expect(imageUrl('abc123.jpg')).toBe('/static/uploads/abc123.jpg');
    });
  });

  describe('searchImages', () => {
    it('sends a GET request with the correct query params', async () => {
      const mockResponse = { query: 'cat', results: [], total: 0 };
      vi.mocked(fetch).mockResolvedValueOnce(
        new Response(JSON.stringify(mockResponse), { status: 200 }),
      );

      const result = await searchImages('cat', 10, 0.5);

      expect(fetch).toHaveBeenCalledOnce();
      const url = vi.mocked(fetch).mock.calls[0][0] as string;
      expect(url).toContain('q=cat');
      expect(url).toContain('limit=10');
      expect(url).toContain('min_score=0.5');
      expect(result).toEqual(mockResponse);
    });

    it('throws on a non-OK response', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Bad request' }), { status: 400 }),
      );

      await expect(searchImages('cat')).rejects.toThrow('Bad request');
    });
  });

  describe('getImage', () => {
    it('fetches the correct URL', async () => {
      const mockImage = { id: 1, filename: 'test.jpg' };
      vi.mocked(fetch).mockResolvedValueOnce(
        new Response(JSON.stringify(mockImage), { status: 200 }),
      );

      const result = await getImage(1);
      expect(fetch).toHaveBeenCalledWith('/api/v1/images/1');
      expect(result).toEqual(mockImage);
    });

    it('throws when image is not found', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Image not found' }), { status: 404 }),
      );

      await expect(getImage(99)).rejects.toThrow('Image not found');
    });
  });

  describe('deleteImage', () => {
    it('sends a DELETE request', async () => {
      vi.mocked(fetch).mockResolvedValueOnce(new Response(null, { status: 204 }));

      await expect(deleteImage(5)).resolves.toBeUndefined();
      expect(fetch).toHaveBeenCalledWith('/api/v1/images/5', { method: 'DELETE' });
    });
  });

  describe('uploadImage', () => {
    it('sends a POST with FormData containing the file', async () => {
      const mockRes = { message: 'ok', image: { id: 1 } };
      vi.mocked(fetch).mockResolvedValueOnce(
        new Response(JSON.stringify(mockRes), { status: 200 }),
      );

      const file = new File(['data'], 'img.jpg', { type: 'image/jpeg' });
      const result = await uploadImage(file);

      expect(fetch).toHaveBeenCalledOnce();
      const [url, options] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
      expect(url).toBe('/api/v1/images/upload');
      expect(options.method).toBe('POST');
      expect(options.body).toBeInstanceOf(FormData);
      expect(result).toEqual(mockRes);
    });
  });
});
