import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/tokens.css';
import './styles/shell.css';
import './metal/metal.css';

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);

import './styles/interaction.css';
import './styles/comparison.css';
import './styles/release.css';

import './styles/case-context.css';
