import { client } from "../client";
import type { ScoreRequest, ScoreResponse } from "../types";

export const scoringApi = {
  /**
   * Score candidates against a job description.
   * POST /score/
   * Synchronous — blocks while LLM evaluates candidates in batches.
   * Large candidate sets can take several minutes.
   */
  async score(body: ScoreRequest): Promise<ScoreResponse> {
    const { data } = await client.post<ScoreResponse>("/score/", body, {
      timeout: 600_000, // 10 min for large batches
    });
    return data;
  },
};
