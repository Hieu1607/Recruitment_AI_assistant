import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import ResumesPage from './pages/ResumesPage';
import JobDescriptionsPage from './pages/JobDescriptionsPage';
import ScoringPage from './pages/ScoringPage';
import ChatPage from './pages/ChatPage';
import ShortlistPage from './pages/ShortlistPage';
import OutreachPage from './pages/OutreachPage';
import InterviewQuestionsPage from './pages/InterviewQuestionsPage';

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/resumes', label: 'Resumes' },
  { to: '/job-descriptions', label: 'Job Descriptions' },
  { to: '/scoring', label: 'Scoring' },
  { to: '/chat', label: 'Chat' },
  { to: '/shortlists', label: 'Shortlists' },
  { to: '/outreach', label: 'Outreach' },
  { to: '/interview-questions', label: 'Interview Questions' },
];

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ display: 'flex', minHeight: '100vh' }}>
        <nav
          style={{
            width: 220,
            padding: '1rem',
            borderRight: '1px solid #e0e0e0',
            background: '#fafafa',
          }}
        >
          <h2 style={{ fontSize: '1rem', marginBottom: '1rem' }}>
            Recruitment AI
          </h2>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {navItems.map(({ to, label }) => (
              <li key={to} style={{ marginBottom: '0.5rem' }}>
                <NavLink
                  to={to}
                  end={to === '/'}
                  style={({ isActive }) => ({
                    textDecoration: 'none',
                    fontWeight: isActive ? 700 : 400,
                    color: isActive ? '#1976d2' : '#333',
                  })}
                >
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <main style={{ flex: 1, padding: '1.5rem' }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/resumes" element={<ResumesPage />} />
            <Route path="/job-descriptions" element={<JobDescriptionsPage />} />
            <Route path="/scoring" element={<ScoringPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/shortlists" element={<ShortlistPage />} />
            <Route path="/outreach" element={<OutreachPage />} />
            <Route
              path="/interview-questions"
              element={<InterviewQuestionsPage />}
            />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
