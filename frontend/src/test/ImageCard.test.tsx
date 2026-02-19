import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ImageCard } from '../components/ImageCard';
import type { ImageSearchResult } from '../api/client';

// Mock the API client module
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    deleteImage: vi.fn(),
    imageUrl: (filename: string) => `/static/uploads/${filename}`,
  };
});

import { deleteImage } from '../api/client';

const mockImage: ImageSearchResult = {
  id: 1,
  filename: 'abc123.jpg',
  original_filename: 'cat.jpg',
  filepath: 'uploads/abc123.jpg',
  description: 'A cute cat sitting on a mat.',
  score: 0.876,
};

describe('ImageCard', () => {
  const onDeleted = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders filename, description and score', () => {
    render(<ImageCard image={mockImage} onDeleted={onDeleted} />);
    expect(screen.getByText('cat.jpg')).toBeInTheDocument();
    expect(screen.getByText('A cute cat sitting on a mat.')).toBeInTheDocument();
    expect(screen.getByText(/0\.876/)).toBeInTheDocument();
  });

  it('renders the image with correct src and alt', () => {
    render(<ImageCard image={mockImage} onDeleted={onDeleted} />);
    const img = screen.getByRole('img') as HTMLImageElement;
    expect(img.src).toContain('/static/uploads/abc123.jpg');
    expect(img.alt).toBe('A cute cat sitting on a mat.');
  });

  it('calls onDeleted after successful delete', async () => {
    vi.mocked(deleteImage).mockResolvedValueOnce(undefined);
    window.confirm = vi.fn(() => true);

    render(<ImageCard image={mockImage} onDeleted={onDeleted} />);
    fireEvent.click(screen.getByRole('button', { name: /delete cat\.jpg/i }));

    await waitFor(() => expect(onDeleted).toHaveBeenCalledWith(1));
  });

  it('shows error message when delete fails', async () => {
    vi.mocked(deleteImage).mockRejectedValueOnce(new Error('Server error'));
    window.confirm = vi.fn(() => true);

    render(<ImageCard image={mockImage} onDeleted={onDeleted} />);
    fireEvent.click(screen.getByRole('button', { name: /delete cat\.jpg/i }));

    await waitFor(() => expect(screen.getByText('Server error')).toBeInTheDocument());
    expect(onDeleted).not.toHaveBeenCalled();
  });

  it('does not delete when user cancels confirmation', async () => {
    window.confirm = vi.fn(() => false);

    render(<ImageCard image={mockImage} onDeleted={onDeleted} />);
    fireEvent.click(screen.getByRole('button', { name: /delete cat\.jpg/i }));

    expect(deleteImage).not.toHaveBeenCalled();
    expect(onDeleted).not.toHaveBeenCalled();
  });
});
