import { useState } from 'react';
import { UploadPage } from './pages/UploadPage';
import { SearchPage } from './pages/SearchPage';
import './App.css';

type Tab = 'search' | 'upload';

function App() {
  const [tab, setTab] = useState<Tab>('search');

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__logo" aria-hidden="true">🔍</span>
          <span className="app-header__title">VisionAI</span>
          <span className="app-header__tagline">AI-powered image search</span>
        </div>
        <nav className="app-nav" aria-label="Main navigation">
          <button
            className={`app-nav__tab${tab === 'search' ? ' app-nav__tab--active' : ''}`}
            onClick={() => setTab('search')}
            aria-current={tab === 'search' ? 'page' : undefined}
          >
            Search
          </button>
          <button
            className={`app-nav__tab${tab === 'upload' ? ' app-nav__tab--active' : ''}`}
            onClick={() => setTab('upload')}
            aria-current={tab === 'upload' ? 'page' : undefined}
          >
            Upload
          </button>
        </nav>
      </header>

      <main className="app-main">
        {tab === 'search' ? <SearchPage /> : <UploadPage />}
      </main>
    </div>
  );
}

export default App;
