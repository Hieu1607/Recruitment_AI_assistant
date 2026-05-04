import { client } from "../client";

export interface UserProfile {
  id: string;
  email: string;
  display_name: string;
}

export interface UpdateProfileRequest {
  display_name?: string;
  email?: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  display_name: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

const TOKEN_KEY = "recruitai.token";

export const authApi = {
  async login(body: LoginRequest): Promise<TokenResponse> {
    const { data } = await client.post<TokenResponse>("/auth/login", body);
    return data;
  },

  async register(body: RegisterRequest): Promise<TokenResponse> {
    const { data } = await client.post<TokenResponse>("/auth/register", body);
    return data;
  },

  storeToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token);
  },

  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  },

  clearToken(): void {
    localStorage.removeItem(TOKEN_KEY);
  },

  isAuthenticated(): boolean {
    return localStorage.getItem(TOKEN_KEY) !== null;
  },

  async me(): Promise<UserProfile> {
    const { data } = await client.get<UserProfile>("/auth/me");
    return data;
  },

  async updateProfile(body: UpdateProfileRequest): Promise<UserProfile> {
    const { data } = await client.patch<UserProfile>("/auth/me", body);
    return data;
  },

  getGoogleLoginUrl(redirect: string = "/dashboard"): string {
    const base = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
    const qs = new URLSearchParams({ redirect });
    return `${base}/auth/google/login?${qs.toString()}`;
  },
};
