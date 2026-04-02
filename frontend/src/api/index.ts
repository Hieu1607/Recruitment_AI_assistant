export const apiCall = async (endpoint: string, options?: RequestInit) => {
    const res = await fetch(`/api${endpoint}`, options);
    return res.json();
};