import axios, { type AxiosInstance } from "axios";
import { ApiError, parseAxiosError } from "./errors";

const TOKEN_KEY = "recruitai.token";

const baseURL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export const client: AxiosInstance = axios.create({
  baseURL,
  timeout: 60_000,
  withCredentials: false,
  headers: { "Content-Type": "application/json" },
});

client.interceptors.request.use((config) => {
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

client.interceptors.response.use(
  (res) => res,
  (err) => {
    // SECURITY: Do NOT log request/response bodies (PII — resume PDFs,
    // candidate data, auth tokens). Log only the normalized ApiError shape.
    throw parseAxiosError(err);
  },
);

export function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError;
}
