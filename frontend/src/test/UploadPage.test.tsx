import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { UploadPage } from '../pages/UploadPage';
import type { ImageUploadResponse } from '../api/client';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    uploadImage: vi.fn(),
  };
});

import { uploadImage } from '../api/client';

const mockUploadResponse: ImageUploadResponse = {
  message: 'Image uploaded and indexed successfully.',
  image: {
    id: 42,
    filename: 'abc123.jpg',
    original_filename: 'photo.jpg',
    filepath: 'uploads/abc123.jpg',
    content_type: 'image/jpeg',
    file_size: 12345,
    description: 'A nice photograph.',
    created_at: '2024-01-01T12:00:00Z',
  },
};

describe('UploadPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders drop zone and submit button', () => {
    render(<UploadPage />);
    expect(screen.getByRole('button', { name: /drop zone/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /upload/i })).toBeInTheDocument();
  });

  it('submit button is disabled when no file is selected', () => {
    render(<UploadPage />);
    expect(screen.getByRole('button', { name: /^upload$/i })).toBeDisabled();
  });

  it('shows success message after upload', async () => {
    vi.mocked(uploadImage).mockResolvedValueOnce(mockUploadResponse);

    render(<UploadPage />);

    // Simulate file selection via the hidden input
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['content'], 'photo.jpg', { type: 'image/jpeg' });
    Object.defineProperty(fileInput, 'files', { value: [file], configurable: true });
    fireEvent.change(fileInput);

    await waitFor(() => expect(screen.getByRole('button', { name: /^upload$/i })).not.toBeDisabled());

    fireEvent.submit(screen.getByRole('button', { name: /^upload$/i }).closest('form')!);

    await waitFor(() =>
      expect(screen.getByText(/image uploaded and indexed successfully/i)).toBeInTheDocument(),
    );
  });

  it('shows error on upload failure', async () => {
    vi.mocked(uploadImage).mockRejectedValueOnce(new Error('File too large'));

    render(<UploadPage />);

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['content'], 'big.jpg', { type: 'image/jpeg' });
    Object.defineProperty(fileInput, 'files', { value: [file], configurable: true });
    fireEvent.change(fileInput);

    await waitFor(() => expect(screen.getByRole('button', { name: /^upload$/i })).not.toBeDisabled());

    fireEvent.submit(screen.getByRole('button', { name: /^upload$/i }).closest('form')!);

    await waitFor(() => expect(screen.getByText('File too large')).toBeInTheDocument());
  });
});
