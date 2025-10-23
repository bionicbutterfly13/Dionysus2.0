/**
 * API Client Configuration
 *
 * Centralized HTTP client for backend communication.
 * Configured for desktop app with proper error handling.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// Get backend URL from environment
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:9127';

/**
 * Axios instance configured for Flux backend
 */
export const api: AxiosInstance = axios.create({
  baseURL: BACKEND_URL,
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Request interceptor
 * Add any auth tokens or custom headers here
 */
api.interceptors.request.use(
  (config) => {
    // Add timestamp to prevent caching
    config.params = {
      ...config.params,
      _t: Date.now(),
    };
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

/**
 * Response interceptor
 * Handle errors globally
 */
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error: No response from backend');
      console.error('Is the backend running on', BACKEND_URL, '?');
    } else {
      // Something else happened
      console.error('Request Error:', error.message);
    }
    return Promise.reject(error);
  }
);

/**
 * Check if backend is available
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await api.get('/health');
    return response.status === 200;
  } catch (error) {
    return false;
  }
}

/**
 * Get backend information
 */
export async function getBackendInfo(): Promise<{
  name: string;
  version: string;
  status: string;
}> {
  const response = await api.get('/api/info');
  return response.data;
}

export default api;
