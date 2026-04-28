import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
    // JWT placeholder — will be replaced when auth is implemented
    'X-User-Id': '00000000-0000-0000-0000-000000000000',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail =
      error.response?.data?.detail ?? error.message ?? 'Unknown error';
    console.error('[API Error]', detail);
    return Promise.reject(error);
  },
);

export default apiClient;
