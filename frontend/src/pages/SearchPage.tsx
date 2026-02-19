import { useState } from 'react';
import type { ImageSearchResult } from '../api/client';
import { searchImages } from '../api/client';
import { ImageCard } from '../components/ImageCard';
import './SearchPage.css';

export function SearchPage() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ImageSearchResult[]>([]);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      const res = await searchImages(q);
      setResults(res.results);
      setSearched(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }

  function handleDeleted(id: number) {
    setResults((prev) => prev.filter((r) => r.id !== id));
  }

  return (
    <div className="search-page">
      <h2>Search Images</h2>
      <form onSubmit={handleSearch} className="search-form">
        <input
          type="search"
          className="search-form__input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe what you're looking for…"
          aria-label="Search query"
        />
        <button
          type="submit"
          className="search-form__btn"
          disabled={!query.trim() || loading}
        >
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>

      {error && <p className="search-page__error">{error}</p>}

      {searched && !loading && (
        <p className="search-page__count">
          {results.length === 0
            ? 'No results found.'
            : `${results.length} result${results.length !== 1 ? 's' : ''} found.`}
        </p>
      )}

      {results.length > 0 && (
        <div className="search-results">
          {results.map((img) => (
            <ImageCard key={img.id} image={img} onDeleted={handleDeleted} />
          ))}
        </div>
      )}
    </div>
  );
}
