import { expect, type APIRequestContext, type Page } from "@playwright/test";

const API_BASE_URL = process.env.E2E_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";
const TOKEN_KEY = "recruitai.token";
const SELECTED_JOB_KEY = "recruit_ai_selected_job_id";

type AuthSetup = {
  accessToken: string;
  jobId: string;
  publicApplyToken: string;
  jobDescriptionId?: string;
};

function randomEmail(prefix: string) {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}@example.com`;
}

function buildPdfBuffer(lines: string[]) {
  const escaped = lines.join("\\n").replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");
  const stream = `BT\n/F1 12 Tf\n72 720 Td\n(${escaped}) Tj\nET`;
  const objects = [
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
    "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
    `4 0 obj\n<< /Length ${Buffer.byteLength(stream, "utf8")} >>\nstream\n${stream}\nendstream\nendobj\n`,
    "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
  ];

  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  for (const object of objects) {
    offsets.push(Buffer.byteLength(pdf, "utf8"));
    pdf += object;
  }
  const xrefOffset = Buffer.byteLength(pdf, "utf8");
  pdf += `xref\n0 ${objects.length + 1}\n`;
  pdf += "0000000000 65535 f \n";
  for (let i = 1; i < offsets.length; i += 1) {
    pdf += `${offsets[i].toString().padStart(10, "0")} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Buffer.from(pdf, "utf8");
}

async function createAccountAndJob(request: APIRequestContext, title: string): Promise<AuthSetup> {
  const registerResponse = await request.post(`${API_BASE_URL}/auth/register`, {
    data: {
      email: randomEmail("playwright"),
      password: "Passw0rd!",
      display_name: "Playwright Smoke",
    },
  });
  expect(registerResponse.ok()).toBeTruthy();
  const { access_token: accessToken } = await registerResponse.json();

  const authHeaders = {
    Authorization: `Bearer ${accessToken}`,
  };

  const jobResponse = await request.post(`${API_BASE_URL}/jobs/`, {
    headers: authHeaders,
    data: {
      title,
      status: "active",
      public_apply_enabled: true,
    },
  });
  expect(jobResponse.ok()).toBeTruthy();
  const job = await jobResponse.json();

  const jdResponse = await request.post(`${API_BASE_URL}/jobs/${job.id}/job-description`, {
    headers: authHeaders,
    data: {
      title: `${title} JD`,
      jd_text: `${title} requires strong testing, Python, recruiter workflow, and browser automation experience.`,
      is_active: true,
    },
  });
  expect(jdResponse.ok()).toBeTruthy();
  const jd = await jdResponse.json();

  const linkResponse = await request.get(`${API_BASE_URL}/jobs/${job.id}/application-link`, {
    headers: authHeaders,
  });
  expect(linkResponse.ok()).toBeTruthy();
  const link = await linkResponse.json();
  const publicApplyToken = String(link.public_apply_url).split("/").pop()!;

  return {
    accessToken,
    jobId: job.id,
    publicApplyToken,
    jobDescriptionId: jd.id,
  };
}

export async function seedWorkspace(
  request: APIRequestContext,
  title: string,
  candidates: Array<{ fullName: string; email: string; lines: string[] }>,
): Promise<AuthSetup> {
  const setup = await createAccountAndJob(request, title);

  for (const candidate of candidates) {
    const response = await request.post(
      `${API_BASE_URL}/public/jobs/${setup.publicApplyToken}/resumes`,
      {
        multipart: {
          full_name: candidate.fullName,
          email: candidate.email,
          file: {
            name: `${candidate.fullName.replace(/\s+/g, "_").toLowerCase()}.pdf`,
            mimeType: "application/pdf",
            buffer: buildPdfBuffer(candidate.lines),
          },
        },
      },
    );
    expect(response.ok()).toBeTruthy();
  }

  const authHeaders = {
    Authorization: `Bearer ${setup.accessToken}`,
  };

  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    const candidateResponse = await request.get(`${API_BASE_URL}/jobs/${setup.jobId}/candidates`, {
      headers: authHeaders,
    });
    expect(candidateResponse.ok()).toBeTruthy();
    const payload = await candidateResponse.json();
    if (payload.total >= candidates.length) {
      return setup;
    }
    await new Promise((resolve) => setTimeout(resolve, 2_000));
  }

  throw new Error(`Timed out waiting for ${candidates.length} candidates to finish processing`);
}

export async function authenticatePage(page: Page, setup: AuthSetup) {
  await page.addInitScript(
    ([token, jobId]) => {
      localStorage.setItem("recruitai.token", token);
      localStorage.setItem("recruit_ai_selected_job_id", jobId);
    },
    [setup.accessToken, setup.jobId],
  );
}

export async function verifyNoConsoleErrors(page: Page) {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(message.text());
    }
  });
  return errors;
}

export { API_BASE_URL, SELECTED_JOB_KEY, TOKEN_KEY };
