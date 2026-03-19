# Demo Message Test Cho LLM Theo Tung Giai Doan

File nay tong hop cac message co the copy dan de test phan hoi cua LLM theo dung cac luong dang duoc backend su dung.

Luu y:
- Hien tai project chi goi LLM o 2 giai doan: semantic query va batch matching score.
- Interview questions va outreach trong code hien tai la rule-based, khong goi LLM.

## 1) Giai doan Query Semantic Search (LLM tool)

Muc tieu: LLM tra ve danh sach candidate ID phu hop cau hoi recruiter.

System message (copy):

```text
You are a precise recruiter search assistant. Output strict JSON.
```

User message template (copy):

```text
Given the recruiter question and candidate snippets, return JSON only with key matchedCandidateIds as an ordered array of IDs that best satisfy the query. Return at most 10 IDs. Question: Tim ung vien phu hop cho vai tro Data Analyst, manh SQL va dashboard, co kinh nghiem retail analytics
Candidates: [
	{
		"id": "11111111-1111-1111-1111-111111111111",
		"name": "Nguyen Van A",
		"summary": "Data Analyst 4 years in FMCG and retail analytics",
		"skills": "SQL, Power BI, Python",
		"experience": "Built sales forecasting and store performance dashboards"
	},
	{
		"id": "22222222-2222-2222-2222-222222222222",
		"name": "Tran Thi B",
		"summary": "Backend Engineer",
		"skills": "Java, Spring Boot, Kafka",
		"experience": "Built microservices for payment systems"
	},
	{
		"id": "33333333-3333-3333-3333-333333333333",
		"name": "Le Van C",
		"summary": "BI Developer in e-commerce",
		"skills": "SQL, Tableau, ETL",
		"experience": "Created executive KPI dashboards and ad-hoc analytics"
	}
]
```

Expected response hop le (example):

```json
{
	"matchedCandidateIds": [
		"11111111-1111-1111-1111-111111111111",
		"33333333-3333-3333-3333-333333333333"
	]
}
```

Checklist pass:
- Phai la JSON hop le.
- Co key matchedCandidateIds.
- matchedCandidateIds la array cac string ID.
- Khong chen text thua ngoai JSON.

## 2) Giai doan Matching Batch Scoring (LLM scorer)

Muc tieu: LLM cham diem nhieu ung vien theo JD va scoring template.

User message template (copy):

```text
You are an objective recruitment scoring system. Return valid JSON only with the shape shown in responseFormat.

{
	"jobDescription": "Can tuyen Senior Data Analyst cho retail domain. Yeu cau manh SQL, dashboarding, stakeholder communication, uu tien co kinh nghiem forecasting.",
	"scoringPromptTemplate": "Score each candidate from 0-100. Weights: skills 0.4, education 0.3, experience 0.3. Explain evidence briefly.",
	"candidates": [
		{
			"candidateId": "11111111-1111-1111-1111-111111111111",
			"fullName": "Nguyen Van A",
			"currentJobTitle": "Senior Data Analyst",
			"education": "Bachelor of Statistics",
			"experience": "4 years in retail analytics, forecasting and dashboarding",
			"skills": "SQL, Power BI, Python",
			"summary": "Data analyst focusing on store performance and demand forecasting"
		},
		{
			"candidateId": "22222222-2222-2222-2222-222222222222",
			"fullName": "Tran Thi B",
			"currentJobTitle": "Backend Engineer",
			"education": "Bachelor of Computer Science",
			"experience": "5 years backend microservices",
			"skills": "Java, Spring Boot",
			"summary": "Backend engineer with strong system design"
		}
	],
	"responseFormat": {
		"scores": [
			{
				"candidateId": "uuid",
				"totalScore": 0,
				"passedThreshold": false,
				"rationale": "string",
				"componentScores": [
					{
						"criterionKey": "skills",
						"weight": 0.4,
						"score": 80,
						"weightedScore": 32,
						"evidenceSummary": "string"
					}
				]
			}
		]
	}
}
```

Expected response hop le (example):

```json
{
	"scores": [
		{
			"candidateId": "11111111-1111-1111-1111-111111111111",
			"totalScore": 88,
			"passedThreshold": true,
			"rationale": "Strong alignment with SQL, dashboarding, and retail forecasting requirements.",
			"componentScores": [
				{
					"criterionKey": "skills",
					"weight": 0.4,
					"score": 90,
					"weightedScore": 36,
					"evidenceSummary": "Direct SQL and BI tool match"
				},
				{
					"criterionKey": "education",
					"weight": 0.3,
					"score": 80,
					"weightedScore": 24,
					"evidenceSummary": "Relevant quantitative degree"
				},
				{
					"criterionKey": "experience",
					"weight": 0.3,
					"score": 93.33,
					"weightedScore": 28,
					"evidenceSummary": "Hands-on retail analytics and forecasting"
				}
			]
		},
		{
			"candidateId": "22222222-2222-2222-2222-222222222222",
			"totalScore": 42,
			"passedThreshold": false,
			"rationale": "Good engineering profile but weak relevance to analytics-focused JD.",
			"componentScores": [
				{
					"criterionKey": "skills",
					"weight": 0.4,
					"score": 35,
					"weightedScore": 14,
					"evidenceSummary": "Missing SQL/BI depth"
				},
				{
					"criterionKey": "education",
					"weight": 0.3,
					"score": 60,
					"weightedScore": 18,
					"evidenceSummary": "General technical background"
				},
				{
					"criterionKey": "experience",
					"weight": 0.3,
					"score": 33.33,
					"weightedScore": 10,
					"evidenceSummary": "Experience not centered on analytics use cases"
				}
			]
		}
	]
}
```

Checklist pass:
- JSON hop le.
- Cho phep 2 dang top-level: array hoac object co key scores la array.
- Moi item can co: candidateId, totalScore, passedThreshold, rationale, componentScores.

## 3) Kich ban test fallback de doi chieu he thong

Muc tieu: Co tinh tao output sai de backend buoc phai fallback heuristic.

### 3.1 Sai JSON (se fallback)

Tra loi nhu sau (co tinh sai):

```text
Candidate A is better than candidate B. I recommend A.
```

Ky vong tren he thong:
- Khong parse duoc JSON.
- Backend fallback sang heuristic scoring.
- Rationale trong ket qua co chuoi:
	Fallback heuristic applied because structured LLM output was unavailable.

### 3.2 Dung JSON nhung sai schema (se fallback)

Tra loi nhu sau (co tinh sai schema):

```json
{
	"result": "candidate A"
}
```

Ky vong tren he thong:
- JSON parse duoc, nhung khong phai list va khong co scores la list.
- Backend fallback sang heuristic scoring voi rationale fallback nhu tren.

## 4) Prompt ngan de test nhanh tren UI LLM

Neu ban muon test nhanh trong playground ma khong can payload lon:

Semantic quick test:

```text
System: You are a precise recruiter search assistant. Output strict JSON.
User: Return JSON only: {"matchedCandidateIds": [...]} for question "Top candidates strong in SQL and Power BI" from these IDs ["id-1","id-2","id-3"].
```

Scoring quick test:

```text
User: Return valid JSON only with key "scores". Score 2 candidates against a Data Analyst JD with fields candidateId, totalScore, passedThreshold, rationale, componentScores.
```

## 5) Cach doi chieu ket qua API sau khi test prompt

Sau khi ban da test prompt voi model, doi chieu voi API response:
- Query flow: answer, matchedCount, matchedCandidateIds, routingStrategy.
- Matching flow: scores[].rationale va componentScores.
- Neu thay rationale fallback, xem lai output JSON cua model theo dung schema o tren.
