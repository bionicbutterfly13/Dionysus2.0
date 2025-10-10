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
  const mockFetch = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        documentsProcessed: 12,
        conceptsExtracted: 45,
        curiosityMissions: 7,
      }),
    } as Response);

    global.fetch = mockFetch as unknown as typeof fetch;
  });

  afterEach(() => {
    mockFetch.mockReset();
  });

  it('should render dashboard title', async () => {
    renderWithQueryClient(<Dashboard />);
    const heading = await screen.findByRole('heading', { level: 1, name: /dashboard/i });
    expect(heading).toBeInTheDocument();
  });

  it('should display stat cards', async () => {
    renderWithQueryClient(<Dashboard />);
    await waitFor(() => expect(mockFetch).toHaveBeenCalled());
    
    await waitFor(() => {
      expect(screen.getByText(/documents processed/i)).toBeInTheDocument();
      expect(screen.getByText(/concepts extracted/i)).toBeInTheDocument();
      expect(screen.getByText(/curiosity missions/i)).toBeInTheDocument();
    });
  });

  it('should handle loading state', async () => {
    renderWithQueryClient(<Dashboard />);
    await waitFor(() => expect(mockFetch).toHaveBeenCalled());
    await waitFor(() => {
      expect(screen.getByText(/development mode/i) || screen.getByText(/simulated data/i)).toBeInTheDocument();
    });
  });

  it('should be accessible', async () => {
    renderWithQueryClient(<Dashboard />);
    await waitFor(() => expect(mockFetch).toHaveBeenCalled());
    const mainHeading = await screen.findByRole('heading', { level: 1 });
    expect(mainHeading).toBeInTheDocument();
  });
});
