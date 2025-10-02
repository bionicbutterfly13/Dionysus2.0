import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Dashboard from '../../pages/Dashboard';

// Mock axios
jest.mock('axios');

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      {component}
    </QueryClientProvider>
  );
};

describe('Dashboard Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render dashboard title', () => {
    renderWithQueryClient(<Dashboard />);
    expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
  });

  it('should display stat cards', async () => {
    renderWithQueryClient(<Dashboard />);
    
    await waitFor(() => {
      expect(screen.getByText(/documents processed/i)).toBeInTheDocument();
      expect(screen.getByText(/concepts extracted/i)).toBeInTheDocument();
      expect(screen.getByText(/curiosity missions/i)).toBeInTheDocument();
    });
  });

  it('should handle loading state', async () => {
    renderWithQueryClient(<Dashboard />);
    // Should show mock data warning when API fails
    await waitFor(() => {
      expect(screen.getByText(/development mode/i) || screen.getByText(/simulated data/i)).toBeInTheDocument();
    });
  });

  it('should be accessible', () => {
    renderWithQueryClient(<Dashboard />);
    // Should have proper heading structure
    const mainHeading = screen.getByRole('heading', { level: 1 });
    expect(mainHeading).toBeInTheDocument();
  });
});