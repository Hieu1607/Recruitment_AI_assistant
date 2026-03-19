from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from src.models.candidate import ExtractionTrace
from src.services.parsing.resume_extractor import ExtractedBlock


class ExtractionTraceService:
    def persist_blocks(
        self,
        session: Session,
        *,
        resume_document_id: uuid.UUID,
        candidate_profile_id: uuid.UUID,
        blocks: list[ExtractedBlock],
    ) -> list[ExtractionTrace]:
        traces: list[ExtractionTrace] = []
        for index, block in enumerate(blocks):
            mapped_field = "summary_text" if index < 2 else "raw_text_block"
            trace = ExtractionTrace(
                resume_document_id=resume_document_id,
                candidate_profile_id=candidate_profile_id,
                source_page=block.page,
                source_bbox=block.bbox,
                source_text_snippet=block.text[:1000],
                mapped_field_name=mapped_field,
                confidence_score=0.95 if index < 2 else 0.8,
            )
            session.add(trace)
            traces.append(trace)
        session.flush()
        return traces


extraction_trace_service = ExtractionTraceService()
