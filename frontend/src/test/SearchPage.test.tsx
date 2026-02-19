import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SearchPage } from '../pages/SearchPage';
import type { SearchResponse } from '../api/client';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    searchImages: vi.fn(),
    deleteImage: vi.fn(),
    imageUrl: (filename: string) => `/static/uploads/${filename}`,
  };
});

import { searchImages } from '../api/client';

const mockResponse: SearchResponse = {
  query: 'cat',
  total: 2,
  results: [
    {
      id: 1,
      filename: 'cat1.jpg',
      original_filename: 'my_cat.jpg',
      filepath: 'uploads/cat1.jpg',
      description: 'A fluffy cat.',
      score: 0.9,
    },
    {
      id: 2,
      filename: 'cat2.jpg',
      original_filename: 'another_cat.jpg',
      filepath: 'uploads/cat2.jpg',
      description: 'A black cat.',
      score: 0.75,
    },
  ],
};

describe('SearchPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the search input and button', () => {
    render(<SearchPage />);
    expect(screen.getByPlaceholderText(/describe what you're looking for/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
  });

  it('disables search button when query is empty', () => {
    render(<SearchPage />);
    expect(screen.getByRole('button', { name: /search/i })).toBeDisabled();
  });

  it('enables search button when query has text', () => {
    render(<SearchPage />);
    const input = screen.getByPlaceholderText(/describe what you're looking for/i);
    fireEvent.change(input, { target: { value: 'dog' } });
    expect(screen.getByRole('button', { name: /search/i })).not.toBeDisabled();
  });

  it('displays results after successful search', async () => {
    vi.mocked(searchImages).mockResolvedValueOnce(mockResponse);

    render(<SearchPage />);
    const input = screen.getByPlaceholderText(/describe what you're looking for/i);
    fireEvent.change(input, { target: { value: 'cat' } });
    fireEvent.submit(screen.getByRole('button', { name: /search/i }).closest('form')!);

    await waitFor(() => expect(screen.getByText('2 results found.')).toBeInTheDocument());
    expect(screen.getByText('my_cat.jpg')).toBeInTheDocument();
    expect(screen.getByText('another_cat.jpg')).toBeInTheDocument();
  });

  it('shows "No results found" when search returns empty', async () => {
    vi.mocked(searchImages).mockResolvedValueOnce({ query: 'xyz', total: 0, results: [] });

    render(<SearchPage />);
    const input = screen.getByPlaceholderText(/describe what you're looking for/i);
    fireEvent.change(input, { target: { value: 'xyz' } });
    fireEvent.submit(screen.getByRole('button', { name: /search/i }).closest('form')!);

    await waitFor(() => expect(screen.getByText('No results found.')).toBeInTheDocument());
  });

  it('shows error on API failure', async () => {
    vi.mocked(searchImages).mockRejectedValueOnce(new Error('Network error'));

    render(<SearchPage />);
    const input = screen.getByPlaceholderText(/describe what you're looking for/i);
    fireEvent.change(input, { target: { value: 'cat' } });
    fireEvent.submit(screen.getByRole('button', { name: /search/i }).closest('form')!);

    await waitFor(() => expect(screen.getByText('Network error')).toBeInTheDocument());
  });
});
