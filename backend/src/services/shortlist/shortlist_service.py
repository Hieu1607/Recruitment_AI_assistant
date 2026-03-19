from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.api.errors import AppError
from src.models.candidate import CandidateProfile
from src.models.engagement import ShortlistCollection, ShortlistItem


@dataclass
class ShortlistWithItems:
    collection: ShortlistCollection
    candidate_ids: list[str]


class ShortlistService:
    def create_shortlist(
        self,
        session: Session,
        name: str,
        created_by_user_id: uuid.UUID,
        candidate_ids: list[uuid.UUID],
        source_query_turn_id: uuid.UUID | None,
    ) -> ShortlistWithItems:
        existing_candidates = list(
            session.scalars(select(CandidateProfile.id).where(CandidateProfile.id.in_(candidate_ids)))
        )
        if len(existing_candidates) != len(set(candidate_ids)):
            raise AppError(
                code="invalid_candidate_ids",
                message="One or more candidate IDs do not exist",
                status_code=404,
            )

        collection = ShortlistCollection(
            name=name,
            created_by_user_id=created_by_user_id,
            source_query_turn_id=source_query_turn_id,
        )
        session.add(collection)
        session.flush()

        for candidate_id in candidate_ids:
            session.add(
                ShortlistItem(
                    shortlist_collection_id=collection.id,
                    candidate_profile_id=candidate_id,
                )
            )

        try:
            session.flush()
        except IntegrityError as exc:
            raise AppError(
                code="shortlist_duplicate_candidate",
                message="A candidate was added more than once to the same shortlist",
                status_code=409,
            ) from exc

        return ShortlistWithItems(collection=collection, candidate_ids=[str(item) for item in candidate_ids])

    def list_shortlists_for_user(
        self,
        session: Session,
        created_by_user_id: uuid.UUID,
        limit: int = 100,
    ) -> list[ShortlistWithItems]:
        collections = list(
            session.scalars(
                select(ShortlistCollection)
                .where(ShortlistCollection.created_by_user_id == created_by_user_id)
                .order_by(ShortlistCollection.created_at.desc())
                .limit(max(1, min(limit, 200)))
            )
        )

        results: list[ShortlistWithItems] = []
        for collection in collections:
            candidate_ids = list(
                session.scalars(
                    select(ShortlistItem.candidate_profile_id)
                    .where(ShortlistItem.shortlist_collection_id == collection.id)
                    .order_by(ShortlistItem.added_at.asc())
                )
            )
            results.append(
                ShortlistWithItems(
                    collection=collection,
                    candidate_ids=[str(candidate_id) for candidate_id in candidate_ids],
                )
            )

        return results


shortlist_service = ShortlistService()
